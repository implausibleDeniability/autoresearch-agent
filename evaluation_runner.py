import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Sequence, Tuple

import tiktoken

from evaluation import EvaluationResult, evaluate
from pii_item import PIIItem
from src.cost_metering.accounting import CostReport, PRICE_TABLE_VERSION
from src.cost_metering.proxy import MeteringProxy

DATA_DIRECTORY = Path("data")
SOURCE_ENCODING = "o200k_base"
SOLUTION_MODULE = "solution"
WORKER_RESULT_PREFIX = "EVALUATION_RESULT="
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_UPSTREAM_BASE_URL = "https://api.openai.com"
UPSTREAM_BASE_URL_ENVIRONMENT = "OPENAI_UPSTREAM_BASE_URL"
SENSITIVE_CHILD_ENVIRONMENT = {
    "AZURE_OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
    UPSTREAM_BASE_URL_ENVIRONMENT,
}


class Dataset:
    DEBUG = "debug"
    DEV_5K = "dev-5k"
    DEV_50K = "dev-50k"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.DEBUG, cls.DEV_5K, cls.DEV_50K


DatasetName = Literal["debug", "dev-5k", "dev-50k"]


def main(arguments: Sequence[str] = ()) -> int:
    parsed = _parse_arguments(arguments or sys.argv[1:])
    if parsed.worker:
        return _run_worker(parsed.module)
    return _run_evaluation(parsed)


def _run_evaluation(arguments: argparse.Namespace) -> int:
    texts = _load_texts(arguments.dataset)
    source_tokens = _count_source_tokens(texts)
    api_key = _required_environment("OPENAI_API_KEY")
    upstream_base_url = os.environ.get(UPSTREAM_BASE_URL_ENVIRONMENT, DEFAULT_UPSTREAM_BASE_URL)
    with MeteringProxy(api_key=api_key, upstream_base_url=upstream_base_url) as meter:
        predictions = _run_solution(
            texts,
            module=SOLUTION_MODULE,
            meter=meter,
            timeout=arguments.timeout,
        )
        cost = meter.seal_and_report()
    result = evaluate(predictions, ground_truth_path=DATA_DIRECTORY / arguments.dataset / "ground_truth.json")
    _print_result(result, cost=cost, source_tokens=source_tokens)
    return 0


def _load_texts(dataset: str) -> Dict[str, str]:
    text_directory = DATA_DIRECTORY / dataset / "texts"
    texts = {path.stem: path.read_text() for path in sorted(text_directory.glob("*.txt"))}
    if not texts:
        raise RuntimeError(f"dataset {dataset!r} contains no text files in {text_directory}")
    return texts


def _count_source_tokens(texts: Mapping[str, str]) -> int:
    encoding = tiktoken.get_encoding(SOURCE_ENCODING)
    return sum(len(encoding.encode(text)) for text in texts.values())


def _run_solution(
    texts: Mapping[str, str],
    *,
    module: str,
    meter: MeteringProxy,
    timeout: float,
) -> Dict[str, List[PIIItem]]:
    command = [sys.executable, "-m", "evaluation_runner", "--worker", "--module", module]
    completed = subprocess.run(
        command,
        input=json.dumps(texts),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=_solution_environment(meter),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"solution failed with exit code {completed.returncode}:\n{completed.stderr[-4000:]}"
        )
    return _parse_worker_result(completed.stdout)


def _solution_environment(meter: MeteringProxy, *, source: Mapping[str, str] = os.environ) -> Dict[str, str]:
    environment = {key: value for key, value in source.items() if key not in SENSITIVE_CHILD_ENVIRONMENT}
    environment["OPENAI_API_KEY"] = meter.run_token
    environment["OPENAI_BASE_URL"] = meter.base_url
    return environment


def _parse_worker_result(output: str) -> Dict[str, List[PIIItem]]:
    result_lines = [line for line in output.splitlines() if line.startswith(WORKER_RESULT_PREFIX)]
    if len(result_lines) != 1:
        raise RuntimeError(f"solution produced {len(result_lines)} result records; expected exactly one")
    serialized = json.loads(result_lines[0][len(WORKER_RESULT_PREFIX) :])
    return {
        document_id: [
            PIIItem(**{field: tuple(values) for field, values in person.items()}) for person in people
        ]
        for document_id, people in serialized.items()
    }


def _run_worker(module_name: str) -> int:
    texts = json.load(sys.stdin)
    extract_pii = importlib.import_module(module_name).extract_pii
    predictions = {document_id: extract_pii(text) for document_id, text in texts.items()}
    serialized = {
        document_id: [asdict(person) for person in people] for document_id, people in predictions.items()
    }
    print(f"{WORKER_RESULT_PREFIX}{json.dumps(serialized)}")
    return 0


def _print_result(result: EvaluationResult, *, cost: CostReport, source_tokens: int) -> None:
    print(f"people_precision={result.people.precision:.6f}")
    print(f"people_recall={result.people.recall:.6f}")
    print(f"people_f1={result.people.f1:.6f}")
    print(f"entity_precision={result.entities.precision:.6f}")
    print(f"entity_recall={result.entities.recall:.6f}")
    print(f"entity_f1={result.entities.f1:.6f}")
    print(f"document_accuracy={result.document_accuracy:.6f}")
    print(f"source_tokens={source_tokens}")
    print(f"pricing_version={PRICE_TABLE_VERSION}")
    print(f"api_cost_usd={cost.total_usd:.8f}")
    print(f"cost_usd_per_million_source_tokens={cost.cost_per_million_source_tokens(source_tokens):.6f}")


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PII extraction solution")
    parser.add_argument("--dataset", choices=Dataset.all(), required=True)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module", default="", help=argparse.SUPPRESS)
    parsed = parser.parse_args(arguments)
    if parsed.worker and not parsed.module:
        parser.error("--worker requires --module")
    return parsed


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
