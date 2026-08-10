import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from src.evaluation.models import PIIItem
from src.evaluation.results import EntityMetrics, EvaluationTrace
from src.evaluation.trace import build_evaluation_trace

DEFAULT_GROUND_TRUTH_PATH = Path("data/dev-19k/ground_truth.json")
DocumentPII = Dict[str, List[PIIItem]]


def evaluate(
    predictions: Mapping[str, Sequence[PIIItem]],
    *,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> EntityMetrics:
    return evaluate_trace(predictions, ground_truth_path=ground_truth_path).metrics


def evaluate_trace(
    predictions: Mapping[str, Sequence[PIIItem]],
    *,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> EvaluationTrace:
    ground_truth = _load_ground_truth(ground_truth_path)
    _validate_prediction_documents(predictions, ground_truth=ground_truth)
    return build_evaluation_trace(predictions, ground_truth=ground_truth)


def _load_ground_truth(path: Path) -> DocumentPII:
    with path.open() as file:
        serialized = json.load(file)
    return {
        document_id: [_deserialize_pii_item(person) for person in people]
        for document_id, people in serialized.items()
    }


def _deserialize_pii_item(values: Mapping[str, Sequence[str]]) -> PIIItem:
    return PIIItem(**{field_name: tuple(field_values) for field_name, field_values in values.items()})


def _validate_prediction_documents(
    predictions: Mapping[str, Sequence[PIIItem]], *, ground_truth: DocumentPII
) -> None:
    unknown = sorted(set(predictions) - set(ground_truth))
    if unknown:
        raise ValueError(f"Predictions contain unknown document IDs: {unknown}")
