import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import tiktoken

from src.cost_metering.accounting import CostReport, PRICE_TABLE_VERSION
from src.cost_metering.proxy import DEFAULT_SPENDING_LIMIT_USD, MeteringProxy
from src.evaluation.metrics import EntityMetrics, evaluate
from src.evaluation.models import PIIItem

DATA_DIRECTORY = Path("data")
SOURCE_ENCODING = "o200k_base"
SOLUTION_MODULE = "solution"
WORKER_RESULT_PREFIX = "EVALUATION_RESULT="
MAX_TIMEOUT_SECONDS = 180.0
USD_PER_CENT = Decimal("0.01")
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
    DEV_19K = "dev-19k"
    DEV_87K = "dev-87k"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.DEBUG, cls.DEV_19K, cls.DEV_87K


def main(arguments: Sequence[str] = ()) -> int:
    parsed = _parse_arguments(arguments or sys.argv[1:])
    if parsed.worker:
        return _run_worker(parsed.module)
    return _run_evaluation(parsed)


def _run_evaluation(arguments: argparse.Namespace) -> int:
    started_at = time.monotonic()
    texts = _load_texts(arguments.dataset)
    source_tokens = _count_source_tokens(texts)
    api_key = _required_environment("OPENAI_API_KEY")
    upstream_base_url = os.environ.get(UPSTREAM_BASE_URL_ENVIRONMENT, DEFAULT_UPSTREAM_BASE_URL)
    spending_limit_usd = arguments.cents_limit * USD_PER_CENT
    predictions, cost = _run_metered_solution(
        texts,
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=spending_limit_usd,
        timeout=arguments.timeout,
    )
    metrics = evaluate(
        predictions,
        ground_truth_path=DATA_DIRECTORY / arguments.dataset / "ground_truth.json",
    )
    duration_seconds = time.monotonic() - started_at
    _print_result(metrics, cost=cost, source_tokens=source_tokens, duration_seconds=duration_seconds)
    return 0


def _run_metered_solution(
    texts: Mapping[str, str],
    *,
    api_key: str,
    upstream_base_url: str,
    spending_limit_usd: Decimal,
    timeout: float,
) -> Tuple[Dict[str, List[PIIItem]], CostReport]:
    with MeteringProxy(
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=spending_limit_usd,
    ) as meter:
        try:
            predictions = _run_solution(texts, module=SOLUTION_MODULE, meter=meter, timeout=timeout)
        except Exception:
            meter.seal_and_report()
            raise
        return predictions, meter.seal_and_report()


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
    command = [sys.executable, "-m", "src.evaluation.cli", "--worker", "--module", module]
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


def _print_result(
    metrics: EntityMetrics,
    *,
    cost: CostReport,
    source_tokens: int,
    duration_seconds: float,
) -> None:
    print(f"f_score={metrics.f_score:.6f}")
    print(f"precision={metrics.precision:.6f}")
    print(f"recall={metrics.recall:.6f}")
    print(f"source_tokens={source_tokens}")
    print(f"pricing_version={PRICE_TABLE_VERSION}")
    print(f"api_cost_usd={cost.total_usd:.8f}")
    print(f"cost_usd_per_million_source_tokens={cost.cost_per_million_source_tokens(source_tokens):.6f}")
    print(f"duration_seconds={duration_seconds:.6f}")


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PII extraction solution")
    parser.add_argument("--dataset", choices=Dataset.all())
    parser.add_argument("--timeout", type=_timeout_seconds, default=MAX_TIMEOUT_SECONDS)
    parser.add_argument(
        "--cents-limit",
        type=_positive_decimal,
        default=DEFAULT_SPENDING_LIMIT_USD / USD_PER_CENT,
        help="absolute API spending limit in cents (default: 8)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module", default="", help=argparse.SUPPRESS)
    parsed = parser.parse_args(arguments)
    if parsed.worker and not parsed.module:
        parser.error("--worker requires --module")
    if not parsed.worker and not parsed.dataset:
        parser.error("--dataset is required")
    return parsed


def _timeout_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"timeout must be greater than 0 and at most {MAX_TIMEOUT_SECONDS:g} seconds"
        )
    return parsed


def _positive_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}") from error
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
