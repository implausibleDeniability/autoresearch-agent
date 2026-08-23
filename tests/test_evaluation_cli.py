import json
import os
import subprocess
import sys
import threading
import time
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.cost_metering.accounting import CostReport, ModelUsage
from src.cost_metering.proxy import MeteringProxy
from src.evaluation.cli import (
    Dataset,
    _count_source_tokens,
    _dataset_description,
    _parse_arguments,
    _parse_worker_result,
    _run_solution,
    _solution_environment,
)
from src.evaluation.worker import DEFAULT_MAX_CONCURRENT_DOCUMENTS


@pytest.mark.parametrize("dataset", Dataset.all())
def test_cli_accepts_each_development_dataset(dataset):
    parsed = _parse_arguments(("--dataset", dataset))

    assert parsed.dataset == dataset


def test_cli_uses_eight_cent_default_limit():
    parsed = _parse_arguments(("--dataset", "debug"))

    assert parsed.cents_limit == Decimal("8")


def test_cli_accepts_intentional_cent_limit_override():
    parsed = _parse_arguments(("--dataset", "debug", "--cents-limit", "20"))

    assert parsed.cents_limit == Decimal("20")


def test_cli_uses_three_minute_timeout_by_default_and_accepts_the_limit():
    default = _parse_arguments(("--dataset", "debug"))
    explicit = _parse_arguments(("--dataset", "debug", "--timeout", "180"))

    assert default.timeout == 180.0
    assert explicit.timeout == 180.0


@pytest.mark.parametrize("value", ["180.0001", "600"])
def test_cli_rejects_timeout_above_three_minutes(value):
    with pytest.raises(SystemExit):
        _parse_arguments(("--dataset", "debug", "--timeout", value))


def test_cli_defaults_to_fifty_concurrent_documents_and_accepts_override():
    default = _parse_arguments(("--dataset", "debug"))
    overridden = _parse_arguments(("--dataset", "debug", "--max-concurrent-documents", "7"))

    assert default.max_concurrent_documents == DEFAULT_MAX_CONCURRENT_DOCUMENTS == 50
    assert overridden.max_concurrent_documents == 7


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_cli_rejects_invalid_concurrent_document_limits(value):
    with pytest.raises(SystemExit):
        _parse_arguments(("--dataset", "debug", "--max-concurrent-documents", value))


def test_cli_accepts_diagnostics_path():
    parsed = _parse_arguments(("--dataset", "debug", "--diagnostics", "diagnostics.json"))

    assert parsed.diagnostics == Path("diagnostics.json")


def test_cli_accepts_exact_response_replay_cache_for_development():
    parsed = _parse_arguments(("--dataset", "debug", "--cache"))

    assert parsed.cache


def test_cli_accepts_dataset_description_mode():
    parsed = _parse_arguments(("--dataset", "dev-202k", "--describe-dataset"))

    assert parsed.describe_dataset


def test_cli_accepts_dynamic_blind_test_name_with_frozen_commit():
    parsed = _parse_arguments(("--dataset", "test-private-v2", "--frozen-commit", "abc1234"))

    assert parsed.dataset == "test-private-v2"
    assert parsed.frozen_commit == "abc1234"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--dataset", "test-private-v2"),
        (
            "--dataset",
            "test-private-v2",
            "--frozen-commit",
            "abc1234",
            "--diagnostics",
            "diagnostics.json",
        ),
        ("--dataset", "dev-19k", "--frozen-commit", "abc1234"),
        ("--dataset", "dev-19k", "--describe-dataset", "--diagnostics", "diagnostics.json"),
        ("--dataset", "test-private-v2", "--describe-dataset", "--frozen-commit", "abc1234"),
        ("--dataset", "test-private-v2", "--frozen-commit", "abc1234", "--cache"),
        ("--dataset", "dev-19k", "--describe-dataset", "--cache"),
        ("--worker", "--module", "solution", "--cache"),
        ("--dataset", "test-private/ground_truth.json", "--frozen-commit", "abc1234"),
    ],
)
def test_cli_rejects_invalid_blind_test_options(arguments):
    with pytest.raises(SystemExit):
        _parse_arguments(arguments)


def test_source_tokens_count_each_original_document_once():
    texts = {"first": "John Smith", "second": "Jane Doe"}

    result = _count_source_tokens(texts)

    assert result == 4


def test_dataset_description_reports_document_token_distribution():
    texts = {
        "one": "John",
        "two": "John Smith",
        "three": "John Michael Smith",
        "four": "John Michael Adam Smith",
    }

    result = _dataset_description(texts)

    assert result.source_encoding == "o200k_base"
    assert result.documents == 4
    assert result.source_tokens == 10
    assert result.min_document_tokens == 1
    assert result.median_document_tokens == 2.5
    assert result.p95_document_tokens == 4
    assert result.max_document_tokens == 4


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


def test_worker_extracts_documents_concurrently(tmp_path: Path):
    # setup
    (tmp_path / "concurrent_solution.py").write_text("""import threading

from src.evaluation.models import PIIItem

barrier = threading.Barrier(2)


def extract_pii(text):
    barrier.wait(timeout=2.0)
    return [PIIItem(first_name=(text,))]
""")
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(repository), str(tmp_path)))

    # operate
    completed = subprocess.run(
        [sys.executable, "-m", "src.evaluation.cli", "--worker", "--module", "concurrent_solution"],
        cwd=tmp_path,
        env=environment,
        input=json.dumps({"first": "Alice", "second": "Bob"}),
        text=True,
        capture_output=True,
        timeout=10.0,
        check=False,
    )

    # check
    assert completed.returncode == 0, completed.stderr
    predictions = _parse_worker_result(completed.stdout)
    assert predictions["first"][0].first_name == ("Alice",)
    assert predictions["second"][0].first_name == ("Bob",)


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
        with pytest.raises(RuntimeError, match="solution failed for documents"):
            _run_solution({"doc": "text"}, module="missing_solution_module", meter=meter, timeout=10.0)
        report = meter.seal_and_report()

    # check
    assert report.total_usd == Decimal("0")


def test_cost_report_normalizes_by_source_tokens():
    report = CostReport((ModelUsage("gpt-4o-mini-2024-07-18", 1_000_000, 0, 0),))

    result = report.cost_per_million_source_tokens(2_000_000)

    assert result == Decimal("0.075")


def test_evaluator_cli_reports_quality_cost_and_duration(tmp_path: Path):
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
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "debug",
            "--diagnostics",
            "diagnostics.json",
        ],
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
    assert "result_schema_version=5" in completed.stdout
    assert "result_status=complete" in completed.stdout
    assert "score_is_final=true" in completed.stdout
    assert "f_score=1.000000" in completed.stdout
    assert "precision=1.000000" in completed.stdout
    assert "recall=1.000000" in completed.stdout
    assert "true_positive=1" in completed.stdout
    assert "false_positive=0" in completed.stdout
    assert "false_negative=0" in completed.stdout
    assert not any(
        name in completed.stdout
        for name in (
            "people_precision=",
            "people_recall=",
            "people_f1=",
            "entity_f_score=",
            "entity_precision=",
            "entity_recall=",
            "entity_f1=",
            "document_accuracy=",
        )
    )
    assert "api_cost_usd=0.00000015" in completed.stdout
    assert "cost_usd_per_million_source_tokens=" in completed.stdout
    fields = [line.partition("=")[0] for line in completed.stdout.splitlines()]
    assert fields == [
        "result_schema_version",
        "result_status",
        "score_is_final",
        "termination_category",
        "evaluation_mode",
        "cache_hits",
        "cache_misses",
        "openai_live_requests",
        "cache_writes",
        "cache_write_errors",
        "cache_errors",
        "f_score",
        "precision",
        "recall",
        "true_positive",
        "false_positive",
        "false_negative",
        "documents_total",
        "documents_completed",
        "documents_failed",
        "documents_not_attempted",
        "source_tokens",
        "completed_source_tokens",
        "pricing_version",
        "api_cost_usd",
        "cost_status",
        "cost_usd_per_million_source_tokens",
        "duration_seconds",
        "document_results_json",
    ]
    assert "diagnostics written: diagnostics.json (1 documents, schema v5)" in completed.stderr
    diagnostics_duration = next(
        line.removeprefix("diagnostics_duration_seconds=")
        for line in completed.stderr.splitlines()
        if line.startswith("diagnostics_duration_seconds=")
    )
    assert float(diagnostics_duration) >= 0
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["documents"][0]["person_matches"] == [{"prediction_index": 0, "ground_truth_index": 0}]
    document_results = json.loads(
        next(
            line.removeprefix("document_results_json=")
            for line in completed.stdout.splitlines()
            if line.startswith("document_results_json=")
        )
    )
    assert document_results[0]["prompt_tokens"] == 1
    assert document_results[0]["completion_tokens"] == 0
    assert document_results[0]["latency_seconds"] > 0
    duration_seconds = next(
        line.removeprefix("duration_seconds=")
        for line in completed.stdout.splitlines()
        if line.startswith("duration_seconds=")
    )
    assert float(duration_seconds) > 0


def test_evaluator_cli_replays_cached_response_without_credentials_or_upstream(tmp_path: Path):
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

    live = subprocess.run(
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
    environment.pop("OPENAI_API_KEY")
    cached = subprocess.run(
        [sys.executable, "-m", "src.evaluation.cli", "--dataset", "debug", "--cache"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    assert live.returncode == 0, live.stderr
    assert "evaluation_mode=live" in live.stdout
    assert "openai_live_requests=1" in live.stdout
    assert "cache_writes=1" in live.stdout
    assert cached.returncode == 0, cached.stderr
    assert "evaluation_mode=cached" in cached.stdout
    assert "cache_hits=1" in cached.stdout
    assert "cache_misses=0" in cached.stdout
    assert "openai_live_requests=0" in cached.stdout
    assert "api_cost_usd=0.00000000" in cached.stdout
    assert (tmp_path / ".openai-response-cache").is_dir()


def test_evaluator_cli_cache_miss_is_partial_without_credentials_or_network(tmp_path: Path):
    _write_cli_fixture(tmp_path)
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    environment["OPENAI_UPSTREAM_BASE_URL"] = "http://127.0.0.1:1"
    environment["PYTHONPATH"] = str(repository)

    completed = subprocess.run(
        [sys.executable, "-m", "src.evaluation.cli", "--dataset", "debug", "--cache"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert "result_status=partial" in completed.stdout
    assert "termination_category=cache_miss" in completed.stdout
    assert "evaluation_mode=cached" in completed.stdout
    assert "cache_misses=1" in completed.stdout
    assert "openai_live_requests=0" in completed.stdout
    assert not (tmp_path / ".openai-response-cache").exists()


def test_evaluator_cli_reports_partial_results_and_continues_after_document_failure(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path)
    text_directory = tmp_path / "data" / "debug" / "texts"
    (text_directory / "second.txt").write_text("FAIL")
    (text_directory / "third.txt").write_text("Jane")
    ground_truth = {
        "doc": [{"first_name": ["John"]}],
        "second": [{"first_name": ["FAIL"]}],
        "third": [{"first_name": ["Jane"]}],
    }
    (tmp_path / "data" / "debug" / "ground_truth.json").write_text(json.dumps(ground_truth))
    (tmp_path / "solution.py").write_text("""from src.evaluation.models import PIIItem


def extract_pii(text):
    if text == "FAIL":
        raise RuntimeError("secret document content")
    return [PIIItem(first_name=(text,))]
""")
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "real-key"
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "debug",
            "--diagnostics",
            "diagnostics.json",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    # check
    assert completed.returncode == 2, completed.stderr
    assert "result_schema_version=5" in completed.stdout
    assert "result_status=partial" in completed.stdout
    assert "score_is_final=false" in completed.stdout
    assert "documents_completed=2" in completed.stdout
    assert "documents_failed=1" in completed.stdout
    assert "partial_true_positive=2" in completed.stdout
    assert "\nf_score=" not in completed.stdout
    assert "observed_api_cost_usd=0.00000000" in completed.stdout
    document_results = json.loads(
        next(
            line.removeprefix("document_results_json=")
            for line in completed.stdout.splitlines()
            if line.startswith("document_results_json=")
        )
    )
    assert [document["status"] for document in document_results] == ["completed", "failed", "completed"]
    assert document_results[1]["source_tokens"] > 0
    assert document_results[1]["failure_category"] == "solution_error"
    assert "secret document content" not in json.dumps(document_results)
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["schema_version"] == 5
    assert diagnostics["lifecycle_status"] == "terminal"
    assert diagnostics["result_status"] == "partial"
    assert diagnostics["completed_document_count"] == 2
    assert len(diagnostics["documents"]) == 2


def test_development_evaluation_runs_concurrently_and_preserves_document_order(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path)
    text_directory = tmp_path / "data" / "debug" / "texts"
    documents = {"doc": "John", "fourth": "Dave", "second": "Jane", "third": "Alex"}
    for document_id, text in documents.items():
        (text_directory / f"{document_id}.txt").write_text(text)
    ground_truth = {document_id: [{"first_name": [text]}] for document_id, text in documents.items()}
    (tmp_path / "data" / "debug" / "ground_truth.json").write_text(json.dumps(ground_truth))
    (tmp_path / "solution.py").write_text("""import time
from pathlib import Path

from src.evaluation.models import PIIItem

DELAYS = {"John": 0.3, "Jane": 0.2, "Alex": 0.1, "Dave": 0.0}


def extract_pii(text):
    Path(f"started-{text}").touch()
    deadline = time.monotonic() + 2.0
    while len(tuple(Path(".").glob("started-*"))) < 4:
        if time.monotonic() >= deadline:
            raise TimeoutError("documents did not run concurrently")
        time.sleep(0.01)
    time.sleep(DELAYS[text])
    return [PIIItem(first_name=(text,))]
""")
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "real-key"
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "debug",
            "--diagnostics",
            "diagnostics.json",
            "--max-concurrent-documents",
            "4",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10.0,
        check=False,
    )

    # check
    assert completed.returncode == 0, completed.stderr
    assert "f_score=1.000000" in completed.stdout
    results = json.loads(
        next(
            line.removeprefix("document_results_json=")
            for line in completed.stdout.splitlines()
            if line.startswith("document_results_json=")
        )
    )
    assert [result["document_id"] for result in results] == list(documents)


def test_first_document_timeout_writes_terminal_ledger_and_kills_process_group(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path)
    marker = tmp_path / "grandchild-survived"
    child_script = f"import pathlib,time; time.sleep(0.6); pathlib.Path({str(marker)!r}).write_text('alive')"
    (tmp_path / "solution.py").write_text(f"""import subprocess
import sys
import time


def extract_pii(text):
    subprocess.Popen([sys.executable, "-c", {child_script!r}])
    time.sleep(10)
""")
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "real-key"
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "debug",
            "--diagnostics",
            "diagnostics.json",
            "--timeout",
            "0.2",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    time.sleep(0.7)

    # check
    assert completed.returncode == 2, completed.stderr
    assert "result_status=partial" in completed.stdout
    assert "termination_category=dataset_deadline" in completed.stdout
    assert not marker.exists()
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["lifecycle_status"] == "terminal"
    assert diagnostics["coverage"] == {
        "total": 1,
        "completed": 0,
        "failed": 1,
        "not_attempted": 0,
    }
    assert diagnostics["document_results"][0]["failure_category"] == "dataset_deadline"


def test_dataset_description_requires_no_credentials_or_api_call(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path)
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "debug",
            "--describe-dataset",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    # check
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "dataset=debug",
        "source_encoding=o200k_base",
        "documents=1",
        "source_tokens=1",
        "min_document_tokens=1",
        "median_document_tokens=1",
        "p95_document_tokens=1",
        "max_document_tokens=1",
    ]
    assert completed.stderr == ""


def test_dataset_description_rejects_blind_dataset_without_emitting_metadata(tmp_path: Path):
    # setup
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "test-private-v2",
            "--describe-dataset",
            "--frozen-commit",
            "abc1234",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    # check
    assert completed.returncode != 0
    assert "--describe-dataset is not allowed with blind test datasets" in completed.stderr
    assert not any(
        field in completed.stdout
        for field in ("source_tokens=", "documents=", "min_document_tokens=", "max_document_tokens=")
    )


def test_blind_test_reports_only_permitted_aggregates_for_frozen_solution(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path, dataset="test-private-v2")
    frozen_commit = _initialize_fixture_repository(tmp_path)
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
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "test-private-v2",
            "--frozen-commit",
            frozen_commit,
        ],
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
    fields = [line.partition("=")[0] for line in completed.stdout.splitlines()]
    assert fields == [
        "f_score",
        "precision",
        "recall",
        "api_cost_usd",
        "duration_seconds",
    ]
    assert completed.stderr == ""
    assert not (tmp_path / ".openai-response-cache").exists()


def test_blind_test_extracts_documents_concurrently(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path, dataset="test-private-v2")
    text_directory = tmp_path / "data" / "test-private-v2" / "texts"
    (text_directory / "second.txt").write_text("Jane")
    ground_truth = {
        "doc": [{"first_name": ["John"]}],
        "second": [{"first_name": ["Jane"]}],
    }
    (tmp_path / "data" / "test-private-v2" / "ground_truth.json").write_text(json.dumps(ground_truth))
    (tmp_path / "solution.py").write_text("""import threading

from src.evaluation.models import PIIItem

barrier = threading.Barrier(2)


def extract_pii(text):
    barrier.wait(timeout=2.0)
    return [PIIItem(first_name=(text,))]
""")
    frozen_commit = _initialize_fixture_repository(tmp_path)
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "real-key"
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "test-private-v2",
            "--frozen-commit",
            frozen_commit,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10.0,
        check=False,
    )

    # check
    assert completed.returncode == 0, completed.stderr
    assert "f_score=1.000000" in completed.stdout
    assert completed.stderr == ""


def test_blind_test_rejects_solution_changed_after_frozen_commit(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path, dataset="test-private-v2")
    frozen_commit = _initialize_fixture_repository(tmp_path)
    (tmp_path / "solution.py").write_text("def extract_pii(text):\n    return []\n")
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "test-private-v2",
            "--frozen-commit",
            frozen_commit,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    # check
    assert completed.returncode != 0
    assert "solution.py differs from frozen commit" in completed.stderr
    assert "OPENAI_API_KEY" not in completed.stderr


def test_blind_test_withholds_solution_failure_details(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path, dataset="test-private-v2")
    (tmp_path / "solution.py").write_text("""import sys


def extract_pii(text):
    print("private document and prediction", file=sys.stderr)
    raise RuntimeError("private label and error")
""")
    frozen_commit = _initialize_fixture_repository(tmp_path)
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "real-key"
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "test-private-v2",
            "--frozen-commit",
            frozen_commit,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    # check
    assert completed.returncode != 0
    assert "blind test evaluation failed; details are withheld" in completed.stderr
    assert "private document" not in completed.stderr
    assert "private label" not in completed.stderr


def test_blind_test_rejects_solution_modified_during_evaluation(tmp_path: Path):
    # setup
    _write_cli_fixture(tmp_path, dataset="test-private-v2")
    (tmp_path / "solution.py").write_text("""from pathlib import Path

from src.evaluation.models import PIIItem


def extract_pii(text):
    Path("solution.py").write_text("def extract_pii(text): return []")
    return [PIIItem(first_name=(text,))]
""")
    frozen_commit = _initialize_fixture_repository(tmp_path)
    repository = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "real-key"
    environment["PYTHONPATH"] = str(repository)

    # operate
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--dataset",
            "test-private-v2",
            "--frozen-commit",
            frozen_commit,
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    # check
    assert completed.returncode != 0
    assert "solution.py differs from frozen commit" in completed.stderr
    assert "f_score=" not in completed.stdout


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
                "model": "gpt-4o-mini-2024-07-18",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 0,
                    "total_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
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


def _write_cli_fixture(directory: Path, *, dataset: str = "debug") -> None:
    text_directory = directory / "data" / dataset / "texts"
    text_directory.mkdir(parents=True)
    (text_directory / "doc.txt").write_text("John")
    ground_truth = {"doc": [{"first_name": ["John"]}]}
    (directory / "data" / dataset / "ground_truth.json").write_text(json.dumps(ground_truth))
    (directory / "solution.py").write_text("""from openai import OpenAI
from src.evaluation.models import PIIItem


def extract_pii(text):
    OpenAI().chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": text}],
    )
    return [PIIItem(first_name=(text,))]
""")


def _initialize_fixture_repository(directory: Path) -> str:
    subprocess.run(["git", "init", "--quiet"], cwd=directory, check=True)
    subprocess.run(["git", "config", "user.name", "Evaluator Test"], cwd=directory, check=True)
    subprocess.run(["git", "config", "user.email", "evaluator@example.test"], cwd=directory, check=True)
    subprocess.run(["git", "add", "solution.py"], cwd=directory, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "Freeze solution"], cwd=directory, check=True)
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()
