import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping

from src.evaluation.results import (
    DocumentEvaluation,
    EntityMetrics,
    EvaluationTrace,
    FieldEvaluation,
    ValueReference,
)
from src.evaluation.source_evidence import SourceEvidence, SourceMatchKind
from src.evaluation.source_matching import (
    SourceTextMatcher,
    SourceValueRole,
    SourceValueRoleLiteral,
    source_matching_policy,
)

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
) -> None:
    serialized = _serialize_trace(trace, texts=texts, dataset=dataset)
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


def _serialize_trace(trace: EvaluationTrace, *, texts: Mapping[str, str], dataset: str) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_matching_policy": source_matching_policy(),
        "dataset": dataset,
        "document_count": len(trace.documents),
        "metrics": _serialize_metrics(trace.metrics),
        "field_metrics": {
            field_name: _serialize_metrics(metrics) for field_name, metrics in trace.field_metrics.items()
        },
        "documents": [
            _serialize_document(document, text=texts[document.document_id]) for document in trace.documents
        ],
    }


def _serialize_document(document: DocumentEvaluation, *, text: str) -> Dict[str, object]:
    source_matcher = SourceTextMatcher(text)
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
        "field_results": [
            _serialize_field(field, text=text, source_matcher=source_matcher) for field in document.fields
        ],
    }


def _serialize_field(
    field: FieldEvaluation, *, text: str, source_matcher: SourceTextMatcher
) -> Dict[str, object]:
    return {
        "field": field.field,
        "metrics": _serialize_metrics(field.metrics),
        "matches": [
            {
                "prediction": _serialize_value(
                    match.prediction,
                    text=text,
                    source_matcher=source_matcher,
                    role=SourceValueRole.PREDICTION,
                ),
                "ground_truth": _serialize_value(
                    match.ground_truth,
                    text=text,
                    source_matcher=source_matcher,
                    role=SourceValueRole.GROUND_TRUTH,
                ),
            }
            for match in field.matches
        ],
        "false_positives": [
            _serialize_value(
                value,
                text=text,
                source_matcher=source_matcher,
                role=SourceValueRole.PREDICTION,
            )
            for value in field.false_positives
        ],
        "false_negatives": [
            _serialize_value(
                value,
                text=text,
                source_matcher=source_matcher,
                role=SourceValueRole.GROUND_TRUTH,
            )
            for value in field.false_negatives
        ],
    }


def _serialize_value(
    reference: ValueReference,
    *,
    text: str,
    source_matcher: SourceTextMatcher,
    role: SourceValueRoleLiteral,
) -> Dict[str, object]:
    match_result = source_matcher.find(reference.value, role=role)
    evidence = match_result.evidence
    counts = Counter(item.match_kind for item in evidence)
    return {
        "person_index": reference.person_index,
        "value_index": reference.value_index,
        "value": reference.value,
        "source_evidence_count": len(evidence),
        "raw_occurrence_count": counts[SourceMatchKind.RAW_EXACT],
        "normalized_occurrence_count": counts[SourceMatchKind.NORMALIZED_EXACT],
        "fuzzy_occurrence_count": counts[SourceMatchKind.FUZZY],
        "fuzzy_search_complete": match_result.fuzzy_search_complete,
        "source_evidence_truncated": len(evidence) > MAX_OCCURRENCES_PER_VALUE,
        "source_evidence": [
            _serialize_source_evidence(item, text=text) for item in evidence[:MAX_OCCURRENCES_PER_VALUE]
        ],
    }


def _serialize_source_evidence(evidence: SourceEvidence, *, text: str) -> Dict[str, object]:
    context_start = max(0, evidence.start - CONTEXT_RADIUS)
    context_end = min(len(text), evidence.end + CONTEXT_RADIUS)
    return {
        "start": evidence.start,
        "end": evidence.end,
        "match_kind": evidence.match_kind,
        "similarity": evidence.similarity,
        "context_start": context_start,
        "context_end": context_end,
        "context": text[context_start:context_end],
    }


def _serialize_metrics(metrics: EntityMetrics) -> Dict[str, object]:
    return {
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f_score": metrics.f_score,
    }
