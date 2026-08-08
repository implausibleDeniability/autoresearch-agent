import json
import os
import subprocess
import sys
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.cost_metering.accounting import CostReport, ModelUsage
from src.cost_metering.proxy import MeteringProxy
from src.evaluation.cli import (
    _count_source_tokens,
    _parse_worker_result,
    _run_solution,
    _solution_environment,
)


def test_source_tokens_count_each_original_document_once():
    texts = {"first": "John Smith", "second": "Jane Doe"}

    result = _count_source_tokens(texts)

    assert result == 4


def test_solution_environment_replaces_openai_credentials():
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url="http://127.0.0.1:1") as meter:

        # operate
        environment = _solution_environment(
            meter,
            source={"PATH": "/usr/bin", "OPENAI_API_KEY": "real-key", "OPENAI_ADMIN_KEY": "admin-key"},
        )

    # check
    assert environment["OPENAI_API_KEY"] == meter.run_token
    assert environment["OPENAI_BASE_URL"] == meter.base_url
    assert "OPENAI_ADMIN_KEY" not in environment


def test_solution_environment_removes_every_evaluator_credential():
    # setup
    sensitive = {
        "AZURE_OPENAI_API_KEY": "azure",
        "OPENAI_ADMIN_KEY": "admin",
        "OPENAI_API_KEY": "real",
        "OPENAI_ORG_ID": "org",
        "OPENAI_ORGANIZATION": "organization",
        "OPENAI_PROJECT": "project",
        "OPENAI_PROJECT_ID": "project-id",
        "OPENAI_UPSTREAM_BASE_URL": "https://example.test",
    }
    with MeteringProxy(api_key="real-key", upstream_base_url="http://127.0.0.1:1") as meter:

        # operate
        environment = _solution_environment(meter, source={"PATH": "/usr/bin", **sensitive})

    # check
    assert not (sensitive.keys() - {"OPENAI_API_KEY"}) & environment.keys()
    assert environment["OPENAI_API_KEY"] == meter.run_token


def test_whitespace_solution_runs_in_subprocess_without_api_calls():
    # setup
    texts = {"empty": " \n\t"}
    with MeteringProxy(api_key="real-key", upstream_base_url="http://127.0.0.1:1") as meter:

        # operate
        predictions = _run_solution(texts, module="solution", meter=meter, timeout=10.0)
        report = meter.seal_and_report()

    # check
    assert predictions == {"empty": []}
    assert report.total_usd == Decimal("0")


def test_worker_result_allows_logs_around_single_record():
    output = 'log before\nEVALUATION_RESULT={"doc": [{"first_name": ["John"]}]}\nlog after\n'

    result = _parse_worker_result(output)

    assert result["doc"][0].first_name == ("John",)


@pytest.mark.parametrize(
    "output",
    ["no result", "EVALUATION_RESULT={}\nEVALUATION_RESULT={}"],
)
def test_worker_result_requires_exactly_one_record(output):
    with pytest.raises(RuntimeError, match="expected exactly one"):
        _parse_worker_result(output)


def test_solution_subprocess_failure_is_reported_without_api_spend():
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url="http://127.0.0.1:1") as meter:

        # operate
        with pytest.raises(RuntimeError, match="solution failed with exit code"):
            _run_solution({"doc": "text"}, module="missing_solution_module", meter=meter, timeout=10.0)
        report = meter.seal_and_report()

    # check
    assert report.total_usd == Decimal("0")


def test_cost_report_normalizes_by_source_tokens():
    report = CostReport((ModelUsage("gpt-4o-mini-2024-07-18", 1_000_000, 0, 0),))

    result = report.cost_per_million_source_tokens(2_000_000)

    assert result == Decimal("0.075")


def test_evaluator_cli_reports_quality_and_immediate_cost(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path)
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _CliUpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    host, port = upstream.server_address
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "real-key"
    environment["OPENAI_UPSTREAM_BASE_URL"] = f"http://{host}:{port}"
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [sys.executable, "-m", "src.evaluation.cli", "--dataset", "debug"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )
    upstream.shutdown()
    upstream.server_close()
    thread.join()

    # check
    assert completed.returncode == 0, completed.stderr
    assert "people_precision=1.000000" in completed.stdout
    assert "api_cost_usd=0.00032500" in completed.stdout
    assert "cost_usd_per_million_source_tokens=" in completed.stdout


class _CliUpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        content_length = int(self.headers["Content-Length"])
        self.rfile.read(content_length)
        content = json.dumps(
            {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "ok", "refusal": None, "role": "assistant"},
                    }
                ],
                "created": 0,
                "model": "gpt-4o-2024-08-06",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string, *args) -> None:
        pass


def _write_cli_fixture(directory: Path) -> None:
    text_directory = directory / "data" / "debug" / "texts"
    text_directory.mkdir(parents=True)
    (text_directory / "doc.txt").write_text("John")
    ground_truth = {"doc": [{"first_name": ["John"]}]}
    (directory / "data" / "debug" / "ground_truth.json").write_text(json.dumps(ground_truth))
    (directory / "solution.py").write_text("""from openai import OpenAI
from src.evaluation.models import PIIItem


def extract_pii(text):
    OpenAI().chat.completions.create(model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": text}])
    return [PIIItem(first_name=(text,))]
""")
