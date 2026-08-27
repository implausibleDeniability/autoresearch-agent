import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple

import tiktoken

from src.cost_metering.accounting import (
    CostReport,
    CostStatus,
    EvaluationMode,
    MeteringOutcome,
    PRICE_TABLE_VERSION,
    cost_is_comparable,
)
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
from src.evaluation.worker import (
    DEFAULT_MAX_CONCURRENT_DOCUMENTS,
    extract_documents,
    run_solution_documents,
    run_worker,
)

DATA_DIRECTORY = Path("data")
SOURCE_ENCODING = "o200k_base"
SOLUTION_MODULE = "solution"
WORKER_RESULT_PREFIX = "EVALUATION_RESULT="
MAX_TIMEOUT_SECONDS = 180.0
USD_PER_CENT = Decimal("0.01")
DEFAULT_UPSTREAM_BASE_URL = "https://api.openai.com"
UPSTREAM_BASE_URL_ENVIRONMENT = "OPENAI_UPSTREAM_BASE_URL"
RESPONSE_CACHE_DIRECTORY_NAME = ".openai-response-cache"
DEFAULT_DEVELOPMENT_SEED = 0
EVALUATION_SEED_ENVIRONMENT = "EVALUATION_SEED"
SENSITIVE_CHILD_ENVIRONMENT = {
    "AZURE_OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
    EVALUATION_SEED_ENVIRONMENT,
    UPSTREAM_BASE_URL_ENVIRONMENT,
}


class Dataset:
    DEBUG = "debug"
    DEV_19K = "dev-19k"
    DEV_87K = "dev-87k"
    DEV_202K = "dev-202k"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.DEBUG, cls.DEV_19K, cls.DEV_87K, cls.DEV_202K

    @classmethod
    def is_blind_test(cls, name: str) -> bool:
        return name.startswith("test-") and Path(name).name == name


@dataclass(frozen=True)
class DatasetDescription:
    source_encoding: str
    documents: int
    source_tokens: int
    min_document_tokens: int
    median_document_tokens: float
    p95_document_tokens: int
    max_document_tokens: int


def main(arguments: Sequence[str] = ()) -> int:
    parsed = _parse_arguments(arguments or sys.argv[1:])
    if parsed.worker:
        return _run_worker(parsed.module, max_concurrent_documents=parsed.max_concurrent_documents)
    if parsed.describe_dataset:
        return _describe_dataset(parsed.dataset)
    return _run_evaluation(parsed)


def _describe_dataset(dataset: str) -> int:
    description = _dataset_description(_load_texts(dataset))
    print(f"dataset={dataset}")
    for field, value in asdict(description).items():
        print(f"{field}={value:g}" if isinstance(value, float) else f"{field}={value}")
    return 0


def _run_evaluation(arguments: argparse.Namespace) -> int:
    started_at = time.monotonic()
    blind_test = Dataset.is_blind_test(arguments.dataset)
    _preflight_evaluation(arguments, blind_test=blind_test)
    if not blind_test:
        print(f"resolved_evaluation_mode={_development_evaluation_mode(arguments).value}", file=sys.stderr)
        print(f"resolved_evaluation_seed={_development_evaluation_seed(arguments)}", file=sys.stderr)
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
    started_at = time.monotonic()
    write_diagnostics(
        arguments.diagnostics,
        trace=run.trace,
        texts=run.texts,
        dataset=arguments.dataset,
        run=run,
    )
    duration_seconds = time.monotonic() - started_at
    print(
        f"diagnostics written: {arguments.diagnostics} "
        f"({len(run.trace.documents)} documents, schema v{SCHEMA_VERSION})",
        file=sys.stderr,
    )
    print(f"diagnostics_duration_seconds={duration_seconds:.3f}", file=sys.stderr)


def _evaluate_dataset(arguments: argparse.Namespace) -> EvaluationRun:
    texts = _load_texts(arguments.dataset)
    document_tokens = _count_document_tokens(texts)
    source_tokens = sum(document_tokens.values())
    blind_test = Dataset.is_blind_test(arguments.dataset)
    evaluation_mode = EvaluationMode.FRESH if blind_test else _development_evaluation_mode(arguments)
    api_key = (
        _required_environment("OPENAI_API_KEY")
        if evaluation_mode.requires_api_key_upfront
        else os.environ.get("OPENAI_API_KEY") or None
    )
    upstream_base_url = os.environ.get(UPSTREAM_BASE_URL_ENVIRONMENT, DEFAULT_UPSTREAM_BASE_URL)
    run_id = uuid.uuid4().hex
    started_at = EvaluationRun.timestamp()
    if blind_test:
        if api_key is None:
            raise RuntimeError("blind evaluation requires an OpenAI API key")
        return _evaluate_blind_dataset(
            arguments,
            texts=texts,
            document_tokens=document_tokens,
            source_tokens=source_tokens,
            api_key=api_key,
            upstream_base_url=upstream_base_url,
            run_id=run_id,
            started_at=started_at,
            max_concurrent_documents=arguments.max_concurrent_documents,
        )
    documents = _not_attempted_documents(texts, document_tokens=document_tokens)
    evaluation_seed = _development_evaluation_seed(arguments)
    initial = _make_run(
        run_id=run_id,
        dataset=arguments.dataset,
        texts=texts,
        source_tokens=source_tokens,
        documents=documents,
        cost=MeteringOutcome(
            CostReport(()),
            CostStatus.PENDING,
            evaluation_mode=evaluation_mode,
        ),
        lifecycle_status=LifecycleStatus.RUNNING,
        termination_category="none",
        evaluation_seed=evaluation_seed,
        started_at=started_at,
    )
    _checkpoint_diagnostics(initial, arguments=arguments)
    deadline = time.monotonic() + arguments.timeout
    with MeteringProxy(
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=arguments.cents_limit * USD_PER_CENT,
        evaluation_mode=evaluation_mode,
        cache_directory=Path.cwd() / RESPONSE_CACHE_DIRECTORY_NAME,
    ) as meter:

        def checkpoint(completed, outcome):
            if not arguments.diagnostics:
                return
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
                evaluation_seed=evaluation_seed,
                started_at=started_at,
            )
            _checkpoint_diagnostics(running, arguments=arguments)

        documents, termination_category = run_solution_documents(
            texts,
            module=SOLUTION_MODULE,
            meter=meter,
            deadline=deadline,
            environment=_solution_environment(meter, evaluation_seed=evaluation_seed),
            source_tokens=document_tokens,
            on_checkpoint=checkpoint,
            max_concurrent_documents=arguments.max_concurrent_documents,
        )
        cost = meter.finalize(timeout=max(deadline - time.monotonic(), 0.0))
    if evaluation_mode.reads_cache and cost.cache_errors:
        termination_category = "cache_error"
    elif evaluation_mode is EvaluationMode.CACHE_FILL and cost.cache_write_errors:
        termination_category = "cache_write_error"
    elif evaluation_mode is EvaluationMode.CACHE_FILL and any(
        "OPENAI_API_KEY is unavailable" in error for error in cost.errors
    ):
        termination_category = "cache_fill_requires_api_key"
    elif evaluation_mode is EvaluationMode.CACHE_FILL and any(
        "previous cache-fill attempt" in error for error in cost.errors
    ):
        termination_category = "cache_fill_failed"
    elif evaluation_mode is EvaluationMode.CACHE and cost.cache_misses:
        termination_category = "cache_miss"
    elif termination_category == "none" and cost.status == CostStatus.INCOMPLETE:
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
        evaluation_seed=evaluation_seed,
        started_at=started_at,
    )
    _checkpoint_diagnostics(run, arguments=arguments)
    return run


def _evaluate_blind_dataset(
    arguments: argparse.Namespace,
    *,
    texts: Mapping[str, str],
    document_tokens: Mapping[str, int],
    source_tokens: int,
    api_key: str,
    upstream_base_url: str,
    run_id: str,
    started_at: str,
    max_concurrent_documents: int,
) -> EvaluationRun:
    predictions, report = _run_metered_solution(
        texts,
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=arguments.cents_limit * USD_PER_CENT,
        timeout=arguments.timeout,
        max_concurrent_documents=max_concurrent_documents,
    )
    documents = tuple(
        DocumentExecution(
            ordinal=ordinal,
            document_id=document_id,
            status=DocumentStatus.COMPLETED,
            source_tokens=document_tokens[document_id],
            predictions=tuple(predictions[document_id]),
        )
        for ordinal, document_id in enumerate(texts)
    )
    return _make_run(
        run_id=run_id,
        dataset=arguments.dataset,
        texts=texts,
        source_tokens=source_tokens,
        documents=documents,
        cost=MeteringOutcome(report, CostStatus.COMPLETE),
        lifecycle_status=LifecycleStatus.TERMINAL,
        termination_category="none",
        evaluation_seed=None,
        started_at=started_at,
    )


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
    evaluation_seed: int | None,
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
        evaluation_seed=evaluation_seed,
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
    completed_by_ordinal = {document.ordinal: document for document in completed}
    return tuple(completed_by_ordinal.get(document.ordinal, document) for document in initial)


def _run_metered_solution(
    texts: Mapping[str, str],
    *,
    api_key: str,
    upstream_base_url: str,
    spending_limit_usd: Decimal,
    timeout: float,
    max_concurrent_documents: int,
) -> Tuple[Dict[str, List[PIIItem]], CostReport]:
    with MeteringProxy(
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=spending_limit_usd,
    ) as meter:
        try:
            predictions = _run_batch_solution(
                texts,
                module=SOLUTION_MODULE,
                meter=meter,
                timeout=timeout,
                max_concurrent_documents=max_concurrent_documents,
            )
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


def _dataset_description(texts: Mapping[str, str]) -> DatasetDescription:
    token_counts = sorted(_document_token_counts(texts))
    p95_index = math.ceil(0.95 * len(token_counts)) - 1
    return DatasetDescription(
        source_encoding=SOURCE_ENCODING,
        documents=len(token_counts),
        source_tokens=sum(token_counts),
        min_document_tokens=token_counts[0],
        median_document_tokens=median(token_counts),
        p95_document_tokens=token_counts[p95_index],
        max_document_tokens=token_counts[-1],
    )


def _document_token_counts(texts: Mapping[str, str]) -> List[int]:
    return list(_count_document_tokens(texts).values())


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
        environment=_solution_environment(meter, evaluation_seed=DEFAULT_DEVELOPMENT_SEED),
        source_tokens=tokens,
        on_checkpoint=lambda documents, outcome: None,
        max_concurrent_documents=DEFAULT_MAX_CONCURRENT_DOCUMENTS,
    )
    failed = [document.document_id for document in documents if document.status != DocumentStatus.COMPLETED]
    if failed:
        raise RuntimeError(f"solution failed for documents {failed}; termination={termination}")
    return {document.document_id: list(document.predictions) for document in documents}


def _run_batch_solution(
    texts: Mapping[str, str],
    *,
    module: str,
    meter: MeteringProxy,
    timeout: float,
    max_concurrent_documents: int,
) -> Dict[str, List[PIIItem]]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.cli",
            "--worker",
            "--module",
            module,
            "--max-concurrent-documents",
            str(max_concurrent_documents),
        ],
        input=json.dumps(texts),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=_solution_environment(meter),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"solution failed with exit code {completed.returncode}")
    return _parse_worker_result(completed.stdout)


def _solution_environment(
    meter: MeteringProxy,
    *,
    evaluation_seed: int | None = None,
    source: Mapping[str, str] = os.environ,
) -> Dict[str, str]:
    environment = {key: value for key, value in source.items() if key not in SENSITIVE_CHILD_ENVIRONMENT}
    environment["OPENAI_API_KEY"] = meter.run_token
    environment["OPENAI_BASE_URL"] = meter.base_url
    if evaluation_seed is not None:
        environment[EVALUATION_SEED_ENVIRONMENT] = str(evaluation_seed)
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


def _run_worker(module_name: str, *, max_concurrent_documents: int) -> int:
    if "EVALUATION_RESULT_FD" in os.environ:
        return run_worker(module_name)
    texts = json.load(sys.stdin)
    predictions = extract_documents(
        texts,
        module_name=module_name,
        max_concurrent_documents=max_concurrent_documents,
    )
    serialized = {
        document_id: [asdict(person) for person in people] for document_id, people in predictions.items()
    }
    print(f"{WORKER_RESULT_PREFIX}{json.dumps(serialized)}")
    return 0


def _print_development_result(run: EvaluationRun, *, duration_seconds: float) -> None:
    status = run.result_status
    metrics = run.trace.metrics
    print(f"result_schema_version={SCHEMA_VERSION}")
    print(f"result_status={status}")
    print(f"score_is_final={'true' if status == ResultStatus.COMPLETE else 'false'}")
    print(f"termination_category={run.termination_category}")
    print(f"evaluation_mode={run.cost.evaluation_mode}")
    print(f"evaluation_seed={run.evaluation_seed}")
    print(f"cache_hits={run.cost.cache_hits}")
    print(f"cache_misses={run.cost.cache_misses}")
    print(f"openai_live_requests={run.cost.live_requests}")
    print(f"cache_writes={run.cost.cache_writes}")
    print(f"cache_write_errors={run.cost.cache_write_errors}")
    print(f"cache_errors={run.cost.cache_errors}")
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
    comparable_cost = cost_is_comparable(
        run.cost,
        result_is_complete=status == ResultStatus.COMPLETE,
    )
    print(f"cost_is_comparable={'true' if comparable_cost else 'false'}")
    if status == ResultStatus.COMPLETE and comparable_cost:
        normalized = run.cost.report.cost_per_million_source_tokens(run.source_tokens)
        print(f"cost_usd_per_million_source_tokens={normalized:.6f}")
    elif status != ResultStatus.COMPLETE and run.completed_source_tokens and comparable_cost:
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
    parser.add_argument(
        "--describe-dataset",
        action="store_true",
        help="print aggregate development-dataset size statistics without running the solution",
    )
    parser.add_argument("--diagnostics", type=Path, help="write detailed evaluation diagnostics as JSON")
    parser.add_argument(
        "--seed",
        type=_non_negative_integer,
        help=f"development evaluation seed (default: {DEFAULT_DEVELOPMENT_SEED})",
    )
    parser.add_argument(
        "--frozen-commit",
        help="current solution commit required for a final blind test",
    )
    parser.add_argument("--timeout", type=_timeout_seconds, default=MAX_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-concurrent-documents",
        type=_positive_integer,
        default=DEFAULT_MAX_CONCURRENT_DOCUMENTS,
        help=f"maximum documents evaluated in parallel (default: {DEFAULT_MAX_CONCURRENT_DOCUMENTS})",
    )
    parser.add_argument(
        "--cents-limit",
        type=_positive_decimal,
        default=DEFAULT_SPENDING_LIMIT_USD / USD_PER_CENT,
        help="absolute API spending limit in cents (default: 8)",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--cache-fill",
        action="store_true",
        help=(
            "read cache; call and charge OpenAI on misses; save successful responses; " "development default"
        ),
    )
    cache_mode.add_argument(
        "--fresh",
        action="store_true",
        help="call and charge OpenAI; bypass cache reads and writes",
    )
    cache_mode.add_argument(
        "--cache",
        action="store_true",
        help="strict response replay; never call OpenAI; fail on a miss",
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
    if parsed.describe_dataset and parsed.diagnostics:
        parser.error("--describe-dataset cannot be combined with --diagnostics")
    if parsed.seed is not None and parsed.worker:
        parser.error("--seed is not allowed with --worker")
    if parsed.seed is not None and parsed.describe_dataset:
        parser.error("--seed is not allowed with --describe-dataset")
    selected_mode = next(
        (flag for flag in ("cache_fill", "fresh", "cache") if getattr(parsed, flag)),
        None,
    )
    if selected_mode and parsed.worker:
        parser.error("development cache modes are not allowed with --worker")
    if selected_mode and parsed.describe_dataset:
        parser.error("development cache modes are not allowed with --describe-dataset")
    if parsed.dataset and Dataset.is_blind_test(parsed.dataset):
        if parsed.describe_dataset:
            parser.error("--describe-dataset is not allowed with blind test datasets")
        if parsed.diagnostics:
            parser.error("--diagnostics is not allowed with blind test datasets")
        if selected_mode:
            parser.error("development cache modes are not allowed with blind test datasets")
        if parsed.seed is not None:
            parser.error("--seed is not allowed with blind test datasets")
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
        message = completed.stderr.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed with exit code {completed.returncode}: {message}"
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


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


def _development_evaluation_mode(arguments: argparse.Namespace) -> EvaluationMode:
    if arguments.fresh:
        return EvaluationMode.FRESH
    if arguments.cache:
        return EvaluationMode.CACHE
    return EvaluationMode.CACHE_FILL


def _development_evaluation_seed(arguments: argparse.Namespace) -> int:
    return DEFAULT_DEVELOPMENT_SEED if arguments.seed is None else arguments.seed


if __name__ == "__main__":
    raise SystemExit(main())
