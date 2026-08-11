import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from src.evaluation.results import (
    DocumentEvaluation,
    EntityMetrics,
    EvaluationTrace,
    FieldEvaluation,
    ValueReference,
)
from src.evaluation.run_results import DocumentExecution, EvaluationRun

SCHEMA_VERSION = 2
CONTEXT_RADIUS = 60
MAX_OCCURRENCES_PER_VALUE = 20


def preflight_diagnostics_path(path: Path) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise ValueError(f"diagnostics parent directory does not exist: {parent}")
    if path.exists() and not path.is_file():
        raise ValueError(f"diagnostics path is not a file: {path}")
    with tempfile.NamedTemporaryFile(dir=parent, prefix=f".{path.name}.") as file:
        file.write(b"{}")
        file.flush()


def write_diagnostics(
    path: Path,
    *,
    trace: EvaluationTrace,
    texts: Mapping[str, str],
    dataset: str,
    run: Optional[EvaluationRun] = None,
) -> None:
    serialized = _serialize_trace(trace, texts=texts, dataset=dataset, run=run)
    temporary_path = _write_temporary_file(path, serialized=serialized)
    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_temporary_file(path: Path, *, serialized: Dict[str, object]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as file:
            os.fchmod(file.fileno(), 0o600)
            json.dump(serialized, file, indent=2)
            file.write("\n")
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _serialize_trace(
    trace: EvaluationTrace,
    *,
    texts: Mapping[str, str],
    dataset: str,
    run: Optional[EvaluationRun],
) -> Dict[str, object]:
    serialized = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_document_count": len(texts),
        "document_count": len(trace.documents),
        "metrics": _serialize_metrics(trace.metrics),
        "field_metrics": {
            field_name: _serialize_metrics(metrics) for field_name, metrics in trace.field_metrics.items()
        },
        "documents": [
            _serialize_document(document, text=texts[document.document_id]) for document in trace.documents
        ],
    }
    if run is not None:
        serialized.update(_serialize_run(run))
    return serialized


def _serialize_run(run: EvaluationRun) -> Dict[str, object]:
    statuses = [document.status for document in run.documents]
    return {
        "run_id": run.run_id,
        "started_at": run.started_at,
        "updated_at": run.updated_at,
        "lifecycle_status": run.lifecycle_status,
        "result_status": run.result_status,
        "score_is_final": run.result_status == "complete",
        "termination_category": run.termination_category,
        "coverage": {
            "total": len(run.documents),
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "not_attempted": statuses.count("not_attempted"),
        },
        "completed_document_count": statuses.count("completed"),
        "failed_document_count": statuses.count("failed"),
        "not_attempted_document_count": statuses.count("not_attempted"),
        "source_tokens": run.source_tokens,
        "completed_source_tokens": run.completed_source_tokens,
        "cost_status": run.cost.status,
        "observed_api_cost_usd": str(run.cost.report.total_usd),
        "metering_error_count": len(run.cost.errors),
        "document_results": [serialize_document_execution(document) for document in run.documents],
    }


def serialize_document_execution(document: DocumentExecution) -> Dict[str, object]:
    usage = document.usage
    return {
        "ordinal": document.ordinal,
        "document_id": document.document_id,
        "status": document.status,
        "source_tokens": document.source_tokens,
        "request_count": len(usage.usages) if usage is not None else None,
        "prompt_tokens": usage.input_tokens if usage is not None else None,
        "cached_prompt_tokens": usage.cached_input_tokens if usage is not None else None,
        "completion_tokens": usage.output_tokens if usage is not None else None,
        "total_tokens": usage.input_tokens + usage.output_tokens if usage is not None else None,
        "observed_api_cost_usd": str(usage.total_usd) if usage is not None else None,
        "latency_seconds": document.latency_seconds,
        "usage_status": document.usage_status,
        "failure_category": document.failure_category or None,
        "error_message": document.error_message or None,
        "retryable": document.retryable,
    }


def _serialize_document(document: DocumentEvaluation, *, text: str) -> Dict[str, object]:
    return {
        "document_id": document.document_id,
        "predictions": [asdict(person) for person in document.predictions],
        "ground_truth": [asdict(person) for person in document.ground_truth],
        "person_matches": [
            {"prediction_index": prediction_index, "ground_truth_index": ground_index}
            for prediction_index, ground_index in document.person_matches
        ],
        "unmatched_prediction_indexes": list(document.unmatched_prediction_indexes),
        "unmatched_ground_truth_indexes": list(document.unmatched_ground_truth_indexes),
        "metrics": _serialize_metrics(document.metrics),
        "field_results": [_serialize_field(field, text=text) for field in document.fields],
    }


def _serialize_field(field: FieldEvaluation, *, text: str) -> Dict[str, object]:
    return {
        "field": field.field,
        "metrics": _serialize_metrics(field.metrics),
        "matches": [
            {
                "prediction": _serialize_value(match.prediction, text=text),
                "ground_truth": _serialize_value(match.ground_truth, text=text),
            }
            for match in field.matches
        ],
        "false_positives": [_serialize_value(value, text=text) for value in field.false_positives],
        "false_negatives": [_serialize_value(value, text=text) for value in field.false_negatives],
    }


def _serialize_value(reference: ValueReference, *, text: str) -> Dict[str, object]:
    occurrences, occurrence_count = _find_occurrences(text, value=reference.value)
    return {
        "person_index": reference.person_index,
        "value_index": reference.value_index,
        "value": reference.value,
        "occurrence_count": occurrence_count,
        "occurrences_truncated": occurrence_count > len(occurrences),
        "occurrences": occurrences,
    }


def _find_occurrences(text: str, *, value: str) -> tuple[List[Dict[str, object]], int]:
    if not value:
        return [], 0
    occurrences = []
    start = text.find(value)
    while start >= 0 and len(occurrences) < MAX_OCCURRENCES_PER_VALUE:
        end = start + len(value)
        context_start = max(0, start - CONTEXT_RADIUS)
        context_end = min(len(text), end + CONTEXT_RADIUS)
        occurrences.append(
            {
                "start": start,
                "end": end,
                "context_start": context_start,
                "context_end": context_end,
                "context": text[context_start:context_end],
            }
        )
        start = text.find(value, end)
    return occurrences, text.count(value)


def _serialize_metrics(metrics: EntityMetrics) -> Dict[str, object]:
    return {
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f_score": metrics.f_score,
    }
