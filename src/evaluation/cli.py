import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import tiktoken

from src.cost_metering.accounting import CostReport, CostStatus, MeteringOutcome, PRICE_TABLE_VERSION
from src.cost_metering.proxy import DEFAULT_SPENDING_LIMIT_USD, MeteringProxy
from src.evaluation.diagnostics import (
    SCHEMA_VERSION,
    preflight_diagnostics_path,
    serialize_document_execution,
    write_diagnostics,
)
from src.evaluation.metrics import EntityMetrics, evaluate_completed_trace
from src.evaluation.models import PIIItem
from src.evaluation.run_results import (
    DocumentExecution,
    DocumentStatus,
    EvaluationRun,
    LifecycleStatus,
    ResultStatus,
)
from src.evaluation.worker import run_solution_documents, run_worker

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

    @classmethod
    def is_blind_test(cls, name: str) -> bool:
        return name.startswith("test-") and Path(name).name == name


def main(arguments: Sequence[str] = ()) -> int:
    parsed = _parse_arguments(arguments or sys.argv[1:])
    if parsed.worker:
        return _run_worker(parsed.module)
    return _run_evaluation(parsed)


def _run_evaluation(arguments: argparse.Namespace) -> int:
    started_at = time.monotonic()
    blind_test = Dataset.is_blind_test(arguments.dataset)
    _preflight_evaluation(arguments, blind_test=blind_test)
    run = _evaluate_with_blind_boundary(arguments, blind_test=blind_test)
    _report_evaluation(
        run,
        arguments=arguments,
        blind_test=blind_test,
        duration_seconds=time.monotonic() - started_at,
    )
    return 0 if run.result_status == ResultStatus.COMPLETE else 2


def _preflight_evaluation(arguments: argparse.Namespace, *, blind_test: bool) -> None:
    if blind_test:
        _validate_frozen_solution(arguments.frozen_commit)
    if arguments.diagnostics:
        preflight_diagnostics_path(arguments.diagnostics)


def _evaluate_with_blind_boundary(
    arguments: argparse.Namespace,
    *,
    blind_test: bool,
) -> EvaluationRun:
    try:
        run = _evaluate_dataset(arguments)
        if blind_test and run.result_status != ResultStatus.COMPLETE:
            raise RuntimeError("blind evaluation did not complete")
        return run
    except Exception:
        if blind_test:
            raise RuntimeError("blind test evaluation failed; details are withheld") from None
        raise


def _report_evaluation(
    run: EvaluationRun,
    *,
    arguments: argparse.Namespace,
    blind_test: bool,
    duration_seconds: float,
) -> None:
    if blind_test:
        _validate_frozen_solution(arguments.frozen_commit)
        _print_blind_test_result(run.trace.metrics, cost=run.cost.report, duration_seconds=duration_seconds)
    else:
        _print_development_result(run, duration_seconds=duration_seconds)
    if arguments.diagnostics:
        _write_development_diagnostics(run, arguments=arguments)


def _write_development_diagnostics(
    run: EvaluationRun,
    *,
    arguments: argparse.Namespace,
) -> None:
    write_diagnostics(
        arguments.diagnostics,
        trace=run.trace,
        texts=run.texts,
        dataset=arguments.dataset,
        run=run,
    )
    print(
        f"diagnostics written: {arguments.diagnostics} "
        f"({len(run.trace.documents)} documents, schema v{SCHEMA_VERSION})",
        file=sys.stderr,
    )


def _evaluate_dataset(arguments: argparse.Namespace) -> EvaluationRun:
    texts = _load_texts(arguments.dataset)
    document_tokens = _count_document_tokens(texts)
    source_tokens = sum(document_tokens.values())
    api_key = _required_environment("OPENAI_API_KEY")
    upstream_base_url = os.environ.get(UPSTREAM_BASE_URL_ENVIRONMENT, DEFAULT_UPSTREAM_BASE_URL)
    run_id = uuid.uuid4().hex
    started_at = EvaluationRun.timestamp()
    documents = _not_attempted_documents(texts, document_tokens=document_tokens)
    initial = _make_run(
        run_id=run_id,
        dataset=arguments.dataset,
        texts=texts,
        source_tokens=source_tokens,
        documents=documents,
        cost=MeteringOutcome(CostReport(()), CostStatus.PENDING),
        lifecycle_status=LifecycleStatus.RUNNING,
        termination_category="none",
        started_at=started_at,
    )
    _checkpoint_diagnostics(initial, arguments=arguments)
    deadline = time.monotonic() + arguments.timeout
    with MeteringProxy(
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=arguments.cents_limit * USD_PER_CENT,
    ) as meter:

        def checkpoint(completed, outcome):
            ledger = _merge_document_ledger(completed, initial=initial.documents)
            running = _make_run(
                run_id=run_id,
                dataset=arguments.dataset,
                texts=texts,
                source_tokens=source_tokens,
                documents=ledger,
                cost=outcome,
                lifecycle_status=LifecycleStatus.RUNNING,
                termination_category="none",
                started_at=started_at,
            )
            _checkpoint_diagnostics(running, arguments=arguments)

        documents, termination_category = run_solution_documents(
            texts,
            module=SOLUTION_MODULE,
            meter=meter,
            deadline=deadline,
            environment=_solution_environment(meter),
            source_tokens=document_tokens,
            on_checkpoint=checkpoint,
        )
        cost = meter.finalize(timeout=max(deadline - time.monotonic(), 0.0))
    if termination_category == "none" and cost.status == CostStatus.INCOMPLETE:
        termination_category = "metering_incomplete"
    if termination_category == "none" and any(
        document.status == DocumentStatus.FAILED for document in documents
    ):
        termination_category = "document_failures"
    run = _make_run(
        run_id=run_id,
        dataset=arguments.dataset,
        texts=texts,
        source_tokens=source_tokens,
        documents=documents,
        cost=cost,
        lifecycle_status=LifecycleStatus.TERMINAL,
        termination_category=termination_category,
        started_at=started_at,
    )
    _checkpoint_diagnostics(run, arguments=arguments)
    return run


def _make_run(
    *,
    run_id: str,
    dataset: str,
    texts: Mapping[str, str],
    source_tokens: int,
    documents: Tuple[DocumentExecution, ...],
    cost: MeteringOutcome,
    lifecycle_status: str,
    termination_category: str,
    started_at: str,
) -> EvaluationRun:
    completed = tuple(document for document in documents if document.status == DocumentStatus.COMPLETED)
    predictions = {document.document_id: document.predictions for document in completed}
    trace = evaluate_completed_trace(
        predictions,
        document_ids=tuple(predictions),
        ground_truth_path=DATA_DIRECTORY / dataset / "ground_truth.json",
    )
    return EvaluationRun(
        run_id=run_id,
        dataset=dataset,
        texts=texts,
        source_tokens=source_tokens,
        documents=documents,
        trace=trace,
        cost=cost,
        lifecycle_status=lifecycle_status,
        termination_category=termination_category,
        started_at=started_at,
        updated_at=EvaluationRun.timestamp(),
    )


def _checkpoint_diagnostics(run: EvaluationRun, *, arguments: argparse.Namespace) -> None:
    if not arguments.diagnostics:
        return
    write_diagnostics(
        arguments.diagnostics,
        trace=run.trace,
        texts=run.texts,
        dataset=run.dataset,
        run=run,
    )


def _not_attempted_documents(
    texts: Mapping[str, str], *, document_tokens: Mapping[str, int]
) -> Tuple[DocumentExecution, ...]:
    return tuple(
        DocumentExecution(
            ordinal=ordinal,
            document_id=document_id,
            status=DocumentStatus.NOT_ATTEMPTED,
            source_tokens=document_tokens[document_id],
        )
        for ordinal, document_id in enumerate(texts)
    )


def _merge_document_ledger(
    completed: Sequence[DocumentExecution], *, initial: Sequence[DocumentExecution]
) -> Tuple[DocumentExecution, ...]:
    return tuple(completed) + tuple(initial[len(completed) :])


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
    return sum(_count_document_tokens(texts).values())


def _count_document_tokens(texts: Mapping[str, str]) -> Dict[str, int]:
    encoding = tiktoken.get_encoding(SOURCE_ENCODING)
    return {document_id: len(encoding.encode(text)) for document_id, text in texts.items()}


def _run_solution(
    texts: Mapping[str, str],
    *,
    module: str,
    meter: MeteringProxy,
    timeout: float,
) -> Dict[str, List[PIIItem]]:
    tokens = _count_document_tokens(texts)
    documents, termination = run_solution_documents(
        texts,
        module=module,
        meter=meter,
        deadline=time.monotonic() + timeout,
        environment=_solution_environment(meter),
        source_tokens=tokens,
        on_checkpoint=lambda documents, outcome: None,
    )
    failed = [document.document_id for document in documents if document.status != DocumentStatus.COMPLETED]
    if failed:
        raise RuntimeError(f"solution failed for documents {failed}; termination={termination}")
    return {document.document_id: list(document.predictions) for document in documents}


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
    return run_worker(module_name)


def _print_development_result(run: EvaluationRun, *, duration_seconds: float) -> None:
    status = run.result_status
    metrics = run.trace.metrics
    print(f"result_schema_version={SCHEMA_VERSION}")
    print(f"result_status={status}")
    print(f"score_is_final={'true' if status == ResultStatus.COMPLETE else 'false'}")
    print(f"termination_category={run.termination_category}")
    prefix = "" if status == ResultStatus.COMPLETE else "partial_"
    print(f"{prefix}f_score={metrics.f_score:.6f}")
    print(f"{prefix}precision={metrics.precision:.6f}")
    print(f"{prefix}recall={metrics.recall:.6f}")
    print(f"{prefix}true_positive={metrics.true_positive}")
    print(f"{prefix}false_positive={metrics.false_positive}")
    print(f"{prefix}false_negative={metrics.false_negative}")
    statuses = [document.status for document in run.documents]
    print(f"documents_total={len(run.documents)}")
    print(f"documents_completed={statuses.count(DocumentStatus.COMPLETED)}")
    print(f"documents_failed={statuses.count(DocumentStatus.FAILED)}")
    print(f"documents_not_attempted={statuses.count(DocumentStatus.NOT_ATTEMPTED)}")
    print(f"source_tokens={run.source_tokens}")
    print(f"completed_source_tokens={run.completed_source_tokens}")
    print(f"pricing_version={PRICE_TABLE_VERSION}")
    cost_key = "api_cost_usd" if status == ResultStatus.COMPLETE else "observed_api_cost_usd"
    print(f"{cost_key}={run.cost.report.total_usd:.8f}")
    print(f"cost_status={run.cost.status}")
    if status == ResultStatus.COMPLETE:
        normalized = run.cost.report.cost_per_million_source_tokens(run.source_tokens)
        print(f"cost_usd_per_million_source_tokens={normalized:.6f}")
    elif run.completed_source_tokens:
        normalized = run.cost.report.cost_per_million_source_tokens(run.completed_source_tokens)
        print(f"partial_cost_usd_per_million_completed_source_tokens={normalized:.6f}")
    print(f"duration_seconds={duration_seconds:.6f}")
    documents = [serialize_document_execution(document) for document in run.documents]
    print(f"document_results_json={json.dumps(documents, separators=(',', ':'))}")


def _print_blind_test_result(
    metrics: EntityMetrics,
    *,
    cost: CostReport,
    duration_seconds: float,
) -> None:
    print(f"f_score={metrics.f_score:.6f}")
    print(f"precision={metrics.precision:.6f}")
    print(f"recall={metrics.recall:.6f}")
    print(f"api_cost_usd={cost.total_usd:.8f}")
    print(f"duration_seconds={duration_seconds:.6f}")


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PII extraction solution")
    parser.add_argument("--dataset", type=_dataset_name)
    parser.add_argument("--diagnostics", type=Path, help="write detailed evaluation diagnostics as JSON")
    parser.add_argument(
        "--frozen-commit",
        help="current solution commit required for a final blind test",
    )
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
    _validate_arguments(parsed, parser=parser)
    return parsed


def _validate_arguments(parsed: argparse.Namespace, *, parser: argparse.ArgumentParser) -> None:
    if parsed.worker and not parsed.module:
        parser.error("--worker requires --module")
    if not parsed.worker and not parsed.dataset:
        parser.error("--dataset is required")
    if parsed.dataset and Dataset.is_blind_test(parsed.dataset):
        if parsed.diagnostics:
            parser.error("--diagnostics is not allowed with blind test datasets")
        if not parsed.frozen_commit:
            parser.error("test-* datasets require --frozen-commit")
    elif parsed.frozen_commit:
        parser.error("--frozen-commit is only allowed with test-* datasets")


def _dataset_name(value: str) -> str:
    if value in Dataset.all() or Dataset.is_blind_test(value):
        return value
    raise argparse.ArgumentTypeError(
        f"dataset must be one of {Dataset.all()!r} or have a name starting with 'test-', got {value!r}"
    )


def _validate_frozen_solution(commit: str) -> None:
    resolved_commit = _git_output("rev-parse", "--verify", f"{commit}^{{commit}}")
    head_commit = _git_output("rev-parse", "HEAD")
    if resolved_commit != head_commit:
        raise RuntimeError(
            f"frozen commit {resolved_commit!r} is not the current HEAD commit {head_commit!r}"
        )
    completed = subprocess.run(
        ["git", "diff", "--quiet", head_commit, "--", "solution.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 1:
        raise RuntimeError(f"solution.py differs from frozen commit {head_commit!r}")
    if completed.returncode:
        raise RuntimeError(
            f"git diff failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


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
