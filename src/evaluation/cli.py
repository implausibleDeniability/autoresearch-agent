import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
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
    append_document_journal,
    preflight_diagnostics_path,
    serialize_document_execution,
    write_diagnostics,
)
from src.evaluation.evidence import EVIDENCE_SCHEMA_VERSION, evidence_path_for, write_evidence
from src.evaluation.metrics import EntityMetrics, evaluate_completed_trace
from src.evaluation.execution import (
    AdmissionStrategy,
    DEFAULT_ADMISSION_STRATEGY,
    DEFAULT_EXECUTION_MODE,
    DEFAULT_MAX_CONCURRENT_DOCUMENTS,
    DEFAULT_MAX_INFLIGHT_LIABILITY_CENTS,
    DEFAULT_SETTLEMENT_GRACE_SECONDS,
    ExecutionMode,
    MAX_CONCURRENT_DOCUMENTS,
    MAX_SETTLEMENT_GRACE_SECONDS,
)
from src.evaluation.models import PIIItem
from src.evaluation.provenance import (
    EvidenceContext,
    FinalProvenance,
    capture_evidence_context,
    snapshot_environment,
)
from src.evaluation.run_results import (
    DocumentExecution,
    DocumentStatus,
    EvaluationRun,
    LifecycleStatus,
    ResultStatus,
)
from src.evaluation.worker import (
    extract_documents,
    run_solution_documents,
    run_worker,
)
from src.evaluation.threaded_worker import run_threaded_worker

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
        if parsed.threaded_worker:
            return run_threaded_worker(
                parsed.module,
                run_id=parsed.worker_run_id,
                max_concurrent_documents=parsed.max_concurrent_documents,
                admission_strategy=parsed.admission_strategy,
            )
        return _run_worker(parsed.module, max_concurrent_documents=parsed.max_concurrent_documents)
    if parsed.describe_dataset:
        return _describe_dataset(parsed.dataset)
    if parsed.preflight:
        return _run_preflight(parsed)
    return _run_evaluation(parsed)


def _run_preflight(arguments: argparse.Namespace) -> int:
    blind_test = Dataset.is_blind_test(arguments.dataset)
    _preflight_evaluation(arguments, blind_test=blind_test)
    texts = _load_texts(arguments.dataset)
    payload = {
        "preflight": "passed",
        "dataset": arguments.dataset,
        "documents": len(texts),
        "execution_mode": arguments.execution_mode,
        "max_concurrent_documents": arguments.max_concurrent_documents,
        "max_upstream_requests": arguments.max_upstream_requests,
        "admission_strategy": arguments.admission_strategy,
        "settled_spend_limit_cents": str(arguments.settled_spend_limit_cents),
        "max_inflight_liability_cents": str(arguments.max_inflight_liability_cents),
        "maximum_api_cost_exposure_cents": str(
            arguments.settled_spend_limit_cents + arguments.max_inflight_liability_cents
        ),
        "settlement_grace_seconds": arguments.settlement_grace_seconds,
        "openai_requests_admitted": 0,
    }
    _print_payload(payload, output_format=arguments.output_format)
    return 0


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
        print(f"resolved_execution_mode={arguments.execution_mode}", file=sys.stderr)
        print(f"resolved_max_concurrent_documents={arguments.max_concurrent_documents}", file=sys.stderr)
        print(f"resolved_max_upstream_requests={arguments.max_upstream_requests}", file=sys.stderr)
        print(f"resolved_admission_strategy={arguments.admission_strategy}", file=sys.stderr)
        print(
            f"resolved_maximum_api_cost_exposure_cents="
            f"{arguments.settled_spend_limit_cents + arguments.max_inflight_liability_cents}",
            file=sys.stderr,
        )
    run = _evaluate_with_blind_boundary(arguments, blind_test=blind_test)
    _report_evaluation(
        run,
        arguments=arguments,
        blind_test=blind_test,
        duration_seconds=time.monotonic() - started_at,
    )
    if run.termination_category == "interrupted":
        return 130
    return 0 if run.result_status == ResultStatus.COMPLETE else 2


def _preflight_evaluation(arguments: argparse.Namespace, *, blind_test: bool) -> None:
    if blind_test:
        _validate_frozen_solution(arguments.frozen_commit)
    if arguments.diagnostics:
        preflight_diagnostics_path(arguments.diagnostics)
        preflight_diagnostics_path(evidence_path_for(arguments.diagnostics))


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
        _print_blind_test_result(
            run.trace.metrics,
            cost=run.cost.report,
            duration_seconds=duration_seconds,
            output_format=arguments.output_format,
        )
    else:
        _print_development_result(
            run,
            duration_seconds=duration_seconds,
            output_format=arguments.output_format,
        )
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
        dataset=run.dataset,
        run=run,
    )
    print(
        f"diagnostics written: {arguments.diagnostics} "
        f"({len(run.trace.documents)} documents, schema v{SCHEMA_VERSION})",
        file=sys.stderr,
    )
    print(f"diagnostics_duration_seconds={time.monotonic() - started_at:.3f}", file=sys.stderr)
    evidence_path = evidence_path_for(arguments.diagnostics)
    print(
        f"comparison evidence written: {evidence_path} (schema v{EVIDENCE_SCHEMA_VERSION})",
        file=sys.stderr,
    )


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
    with capture_evidence_context(
        run_id=run_id,
        dataset=arguments.dataset,
        upstream_base_url=upstream_base_url,
    ) as evidence_context:
        return _evaluate_development_dataset(
            arguments,
            texts=texts,
            document_tokens=document_tokens,
            source_tokens=source_tokens,
            api_key=api_key,
            upstream_base_url=upstream_base_url,
            run_id=run_id,
            started_at=started_at,
            evaluation_mode=evaluation_mode,
            evidence_context=evidence_context,
        )


def _evaluate_development_dataset(
    arguments: argparse.Namespace,
    *,
    texts: Mapping[str, str],
    document_tokens: Mapping[str, int],
    source_tokens: int,
    api_key: str | None,
    upstream_base_url: str,
    run_id: str,
    started_at: str,
    evaluation_mode: EvaluationMode,
    evidence_context: EvidenceContext,
) -> EvaluationRun:
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
        execution_mode=arguments.execution_mode,
        evaluation_seed=evaluation_seed,
        started_at=started_at,
    )
    _checkpoint_diagnostics(initial, arguments=arguments, evidence_context=evidence_context)
    deadline = time.monotonic() + arguments.timeout
    with MeteringProxy(
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=arguments.cents_limit * USD_PER_CENT,
        max_inflight_liability_usd=arguments.max_inflight_liability_cents * USD_PER_CENT,
        evaluation_mode=evaluation_mode,
        cache_directory=Path.cwd() / RESPONSE_CACHE_DIRECTORY_NAME,
        admission_deadline=deadline,
        max_upstream_requests=arguments.max_upstream_requests,
    ) as meter:
        journaled_ordinals = set()
        materialized_count = 0
        last_materialized_at = time.monotonic()

        def checkpoint(completed, outcome):
            nonlocal materialized_count, last_materialized_at
            if not arguments.diagnostics:
                return
            newly_settled = tuple(
                document for document in completed if document.ordinal not in journaled_ordinals
            )
            append_document_journal(arguments.diagnostics, newly_settled)
            journaled_ordinals.update(document.ordinal for document in newly_settled)
            now = time.monotonic()
            materialization_due = (
                len(completed) - materialized_count >= 10 or now - last_materialized_at >= 1.0
            )
            if not materialization_due:
                return
            ledger = _merge_document_ledger(completed, initial=initial.documents)
            running = replace(
                initial,
                documents=ledger,
                cost=outcome,
                updated_at=EvaluationRun.timestamp(),
            )
            _checkpoint_diagnostics(
                running,
                arguments=arguments,
                evidence_context=evidence_context,
            )
            materialized_count = len(completed)
            last_materialized_at = now

        documents, termination_category = run_solution_documents(
            texts,
            module=evidence_context.solution_module,
            meter=meter,
            deadline=deadline,
            environment=_solution_environment(
                meter,
                evaluation_seed=evaluation_seed,
                evidence_context=evidence_context,
            ),
            source_tokens=document_tokens,
            on_checkpoint=checkpoint,
            max_concurrent_documents=arguments.max_concurrent_documents,
            execution_mode=arguments.execution_mode,
            admission_strategy=arguments.admission_strategy,
            run_id=run_id,
        )
        cost = meter.finalize(timeout=arguments.settlement_grace_seconds)
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
        execution_mode=arguments.execution_mode,
        evaluation_seed=evaluation_seed,
        started_at=started_at,
    )
    final_provenance = evidence_context.finalize()
    if arguments.diagnostics:
        write_evidence(
            evidence_path_for(arguments.diagnostics),
            run=run,
            context=evidence_context,
            provenance=final_provenance,
        )
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
    deadline = time.monotonic() + arguments.timeout
    with MeteringProxy(
        api_key=api_key,
        upstream_base_url=upstream_base_url,
        spending_limit_usd=arguments.cents_limit * USD_PER_CENT,
        max_inflight_liability_usd=arguments.max_inflight_liability_cents * USD_PER_CENT,
        admission_deadline=deadline,
        max_upstream_requests=arguments.max_upstream_requests,
    ) as meter:
        documents, termination_category = run_solution_documents(
            texts,
            module=SOLUTION_MODULE,
            meter=meter,
            deadline=deadline,
            environment=_solution_environment(meter),
            source_tokens=document_tokens,
            on_checkpoint=lambda _documents, _outcome: None,
            max_concurrent_documents=max_concurrent_documents,
            execution_mode=ExecutionMode.THREADED,
            admission_strategy=arguments.admission_strategy,
            run_id=run_id,
        )
        cost = meter.finalize(timeout=arguments.settlement_grace_seconds)
    if termination_category == "none" and cost.status == CostStatus.INCOMPLETE:
        termination_category = "metering_incomplete"
    if termination_category == "none" and any(
        document.status == DocumentStatus.FAILED for document in documents
    ):
        termination_category = "document_failures"
    return _make_run(
        run_id=run_id,
        dataset=arguments.dataset,
        texts=texts,
        source_tokens=source_tokens,
        documents=documents,
        cost=cost,
        lifecycle_status=LifecycleStatus.TERMINAL,
        termination_category=termination_category,
        execution_mode=ExecutionMode.THREADED,
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
    execution_mode: str,
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
        execution_mode=execution_mode,
        evaluation_seed=evaluation_seed,
        started_at=started_at,
        updated_at=EvaluationRun.timestamp(),
    )


def _checkpoint_diagnostics(
    run: EvaluationRun,
    *,
    arguments: argparse.Namespace,
    evidence_context: EvidenceContext,
    provenance: FinalProvenance | None = None,
) -> None:
    if not arguments.diagnostics:
        return
    write_diagnostics(
        arguments.diagnostics,
        trace=run.trace,
        texts=run.texts,
        dataset=run.dataset,
        run=run,
    )
    write_evidence(
        evidence_path_for(arguments.diagnostics),
        run=run,
        context=evidence_context,
        provenance=provenance,
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
    evidence_context: EvidenceContext | None = None,
    source: Mapping[str, str] = os.environ,
) -> Dict[str, str]:
    environment = {key: value for key, value in source.items() if key not in SENSITIVE_CHILD_ENVIRONMENT}
    environment["OPENAI_API_KEY"] = meter.run_token
    environment["OPENAI_BASE_URL"] = meter.base_url
    if evaluation_seed is not None:
        environment[EVALUATION_SEED_ENVIRONMENT] = str(evaluation_seed)
    if evidence_context is not None:
        environment = snapshot_environment(evidence_context, environment)
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


def _print_development_result(
    run: EvaluationRun, *, duration_seconds: float, output_format: str = "text"
) -> None:
    _print_payload(
        _development_result_payload(run, duration_seconds=duration_seconds),
        output_format=output_format,
    )


def _development_result_payload(run: EvaluationRun, *, duration_seconds: float) -> Dict[str, object]:
    status = run.result_status
    metrics = run.trace.metrics
    prefix = "" if status == ResultStatus.COMPLETE else "partial_"
    statuses = [document.status for document in run.documents]
    cost_key = "api_cost_usd" if status == ResultStatus.COMPLETE else "observed_api_cost_usd"
    comparable_cost = cost_is_comparable(
        run.cost,
        result_is_complete=status == ResultStatus.COMPLETE,
    )
    payload = {
        "result_schema_version": SCHEMA_VERSION,
        "result_status": status,
        "score_is_final": status == ResultStatus.COMPLETE,
        "termination_category": run.termination_category,
        "evaluation_mode": str(run.cost.evaluation_mode),
        "evaluation_seed": run.evaluation_seed,
        "execution_mode": run.execution_mode,
        "cost_is_final": run.cost.cost_is_final,
        "usage_attribution_status": run.usage_attribution_status,
        "cache_hits": run.cost.cache_hits,
        "cache_misses": run.cost.cache_misses,
        "openai_live_requests": run.cost.live_requests,
        "cache_writes": run.cost.cache_writes,
        "cache_write_errors": run.cost.cache_write_errors,
        "cache_errors": run.cost.cache_errors,
        f"{prefix}f_score": round(metrics.f_score, 6),
        f"{prefix}precision": round(metrics.precision, 6),
        f"{prefix}recall": round(metrics.recall, 6),
        f"{prefix}true_positive": metrics.true_positive,
        f"{prefix}false_positive": metrics.false_positive,
        f"{prefix}false_negative": metrics.false_negative,
        "documents_total": len(run.documents),
        "documents_completed": statuses.count(DocumentStatus.COMPLETED),
        "documents_failed": statuses.count(DocumentStatus.FAILED),
        "documents_not_attempted": statuses.count(DocumentStatus.NOT_ATTEMPTED),
        "source_tokens": run.source_tokens,
        "completed_source_tokens": run.completed_source_tokens,
        "pricing_version": PRICE_TABLE_VERSION,
        cost_key: _decimal_text(run.cost.report.total_usd, places=8),
        "reserved_api_cost_usd": _decimal_text(run.cost.reserved_api_cost_usd, places=8),
        "unknown_api_cost_liability_usd": _decimal_text(
            run.cost.unknown_api_cost_liability_usd,
            places=8,
        ),
        "maximum_api_cost_exposure_usd": _decimal_text(
            run.cost.maximum_api_cost_exposure_usd,
            places=8,
        ),
        "peak_reserved_api_cost_usd": _decimal_text(run.cost.peak_reserved_api_cost_usd, places=8),
        "peak_active_upstream_requests": run.cost.peak_active_upstream_requests,
        "reservation_wait_seconds": round(run.cost.reservation_wait_seconds, 6),
        "cost_status": run.cost.status,
        "cost_is_comparable": comparable_cost,
    }
    error = _operational_error(run)
    if error:
        payload.update(error)
    if status == ResultStatus.COMPLETE and comparable_cost:
        normalized = run.cost.report.cost_per_million_source_tokens(run.source_tokens)
        payload["cost_usd_per_million_source_tokens"] = _decimal_text(normalized, places=6)
    elif status != ResultStatus.COMPLETE and run.completed_source_tokens and comparable_cost:
        normalized = run.cost.report.cost_per_million_source_tokens(run.completed_source_tokens)
        payload["partial_cost_usd_per_million_completed_source_tokens"] = _decimal_text(
            normalized,
            places=6,
        )
    payload["duration_seconds"] = round(duration_seconds, 6)
    payload["document_results_json"] = [serialize_document_execution(document) for document in run.documents]
    return payload


def _print_blind_test_result(
    metrics: EntityMetrics,
    *,
    cost: CostReport,
    duration_seconds: float,
    output_format: str = "text",
) -> None:
    payload = {
        "f_score": round(metrics.f_score, 6),
        "precision": round(metrics.precision, 6),
        "recall": round(metrics.recall, 6),
        "api_cost_usd": _decimal_text(cost.total_usd, places=8),
        "duration_seconds": round(duration_seconds, 6),
    }
    _print_payload(payload, output_format=output_format)


def _print_payload(payload: Mapping[str, object], *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        return
    for key, value in payload.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, float):
            value = f"{value:.6f}"
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        print(f"{key}={value}")
    sys.stdout.flush()


def _decimal_text(value: Decimal, *, places: int) -> str:
    return f"{value:.{places}f}"


def _operational_error(run: EvaluationRun) -> Dict[str, str]:
    if run.result_status == ResultStatus.COMPLETE:
        return {}
    if run.termination_category == "interrupted":
        return _error_fields(
            code="E_INTERRUPTED",
            problem="evaluation interrupted after preserving settled results",
            fix="rerun the same command or inspect the diagnostics journal",
            anchor="#interrupts",
        )
    if any("liability" in error for error in run.cost.errors) or run.termination_category == "spending_limit":
        return _error_fields(
            code="E_LIABILITY_LIMIT",
            problem="configured API liability prevented a request from settling",
            fix="lower concurrency or pass an explicit larger --max-inflight-liability-cents value",
            anchor="#cost-safety",
        )
    if not run.cost.cost_is_final:
        return _error_fields(
            code="E_COST_UNSETTLED",
            problem="one or more admitted API requests did not produce final billing evidence",
            fix="use the maximum exposure value and inspect diagnostics before retrying",
            anchor="#cost-finality",
        )
    if run.execution_mode == ExecutionMode.THREADED:
        return _error_fields(
            code="E_THREADED_CONTRACT",
            problem="the solution did not complete under import-once concurrent execution",
            fix="rerun the same command with --execution-mode isolated",
            anchor="#execution-modes",
        )
    return {}


def _error_fields(*, code: str, problem: str, fix: str, anchor: str) -> Dict[str, str]:
    return {
        "error_code": code,
        "error_problem": problem,
        "error_fix": fix,
        "error_docs": f"research-runbook.md{anchor}",
    }


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PII extraction solution")
    parser.add_argument("--dataset", type=_dataset_name)
    parser.add_argument(
        "--describe-dataset",
        action="store_true",
        help="print aggregate development-dataset size statistics without running the solution",
    )
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--diagnostics", type=Path, help="write detailed diagnostics to PATH")
    diagnostics.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="write diagnostics under DIR using a collision-free file name",
    )
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
        "--execution-mode",
        choices=ExecutionMode.all(),
        default=None,
        help=f"solution isolation topology (default: {DEFAULT_EXECUTION_MODE})",
    )
    parser.add_argument(
        "--max-concurrent-documents",
        type=_positive_integer,
        default=DEFAULT_MAX_CONCURRENT_DOCUMENTS,
        help=(
            f"maximum documents evaluated in parallel, 1-{MAX_CONCURRENT_DOCUMENTS} "
            f"(default: {DEFAULT_MAX_CONCURRENT_DOCUMENTS})"
        ),
    )
    parser.add_argument(
        "--max-upstream-requests",
        type=_positive_integer,
        help="maximum simultaneous OpenAI requests (default: document concurrency)",
    )
    parser.add_argument(
        "--admission-strategy",
        choices=AdmissionStrategy.all(),
        default=DEFAULT_ADMISSION_STRATEGY,
        help=f"threaded health admission policy (default: {DEFAULT_ADMISSION_STRATEGY})",
    )
    spending_limit = parser.add_mutually_exclusive_group()
    spending_limit.add_argument(
        "--settled-spend-limit-cents",
        type=_positive_decimal,
        help="stop new paid requests after observed spend reaches this many cents (default: 8)",
    )
    spending_limit.add_argument(
        "--cents-limit",
        type=_positive_decimal,
        help="deprecated alias for --settled-spend-limit-cents",
    )
    parser.add_argument(
        "--max-inflight-liability-cents",
        type=_positive_decimal,
        default=Decimal(DEFAULT_MAX_INFLIGHT_LIABILITY_CENTS),
        help=(
            "maximum reserved or unknown-billing API liability in cents "
            f"(default: {DEFAULT_MAX_INFLIGHT_LIABILITY_CENTS})"
        ),
    )
    parser.add_argument(
        "--settlement-grace-seconds",
        type=_settlement_grace_seconds,
        default=DEFAULT_SETTLEMENT_GRACE_SECONDS,
        help=(
            "wait for admitted API requests after worker stop "
            f"(default: {DEFAULT_SETTLEMENT_GRACE_SECONDS:g})"
        ),
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
    parser.add_argument("--preflight", action="store_true", help="validate configuration without API calls")
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="final stdout format (default: text)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--threaded-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--module", default="", help=argparse.SUPPRESS)
    parsed = parser.parse_args(arguments)
    _resolve_argument_defaults(parsed, parser=parser)
    _validate_arguments(parsed, parser=parser)
    return parsed


def _resolve_argument_defaults(parsed: argparse.Namespace, *, parser: argparse.ArgumentParser) -> None:
    default_cents = DEFAULT_SPENDING_LIMIT_USD / USD_PER_CENT
    parsed.settled_spend_limit_cents = parsed.settled_spend_limit_cents or parsed.cents_limit or default_cents
    parsed.cents_limit = parsed.settled_spend_limit_cents
    parsed.max_upstream_requests = parsed.max_upstream_requests or parsed.max_concurrent_documents
    if parsed.execution_mode is None:
        parsed.execution_mode = (
            ExecutionMode.THREADED if Dataset.is_blind_test(parsed.dataset or "") else DEFAULT_EXECUTION_MODE
        )
    if parsed.diagnostics_dir:
        if not parsed.diagnostics_dir.is_dir():
            parser.error(f"--diagnostics-dir does not exist: {parsed.diagnostics_dir}")
        run_name = f"{parsed.dataset}-{uuid.uuid4().hex}.json"
        parsed.diagnostics = parsed.diagnostics_dir / run_name


def _validate_arguments(parsed: argparse.Namespace, *, parser: argparse.ArgumentParser) -> None:
    if parsed.worker and not parsed.module:
        parser.error("--worker requires --module")
    if parsed.threaded_worker and not parsed.worker:
        parser.error("--threaded-worker requires --worker")
    if parsed.threaded_worker and not parsed.worker_run_id:
        parser.error("--threaded-worker requires --worker-run-id")
    if not parsed.worker and not parsed.dataset:
        parser.error("--dataset is required")
    if Dataset.is_blind_test(parsed.dataset or "") and parsed.execution_mode != ExecutionMode.THREADED:
        parser.error("blind evaluations require --execution-mode threaded")
    if parsed.max_concurrent_documents > MAX_CONCURRENT_DOCUMENTS:
        parser.error(f"--max-concurrent-documents must be at most {MAX_CONCURRENT_DOCUMENTS}")
    if parsed.max_upstream_requests > MAX_CONCURRENT_DOCUMENTS:
        parser.error(f"--max-upstream-requests must be at most {MAX_CONCURRENT_DOCUMENTS}")
    if parsed.describe_dataset and parsed.diagnostics:
        parser.error("--describe-dataset cannot be combined with --diagnostics")
    if parsed.seed is not None and parsed.worker:
        parser.error("--seed is not allowed with --worker")
    if parsed.seed is not None and parsed.describe_dataset:
        parser.error("--seed is not allowed with --describe-dataset")
    if parsed.preflight and parsed.worker:
        parser.error("--preflight is not allowed with --worker")
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


def _settlement_grace_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > MAX_SETTLEMENT_GRACE_SECONDS:
        raise argparse.ArgumentTypeError(
            f"settlement grace must be between 0 and {MAX_SETTLEMENT_GRACE_SECONDS:g} seconds"
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
