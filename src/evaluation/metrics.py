import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from src.evaluation.models import GroundTruthPIIItem, PIIItem
from src.evaluation.results import EntityMetrics, EvaluationTrace
from src.evaluation.trace import build_evaluation_trace

DEFAULT_GROUND_TRUTH_PATH = Path("data/dev-19k/ground_truth.json")
DocumentGroundTruth = Dict[str, List[GroundTruthPIIItem]]


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


def evaluate_completed_trace(
    predictions: Mapping[str, Sequence[PIIItem]],
    *,
    document_ids: Sequence[str],
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> EvaluationTrace:
    ground_truth = _load_ground_truth(ground_truth_path)
    _validate_prediction_documents(predictions, ground_truth=ground_truth)
    completed_ground_truth = {document_id: ground_truth[document_id] for document_id in document_ids}
    return build_evaluation_trace(predictions, ground_truth=completed_ground_truth)


def _load_ground_truth(path: Path) -> DocumentGroundTruth:
    with path.open() as file:
        serialized = json.load(file)
    return {
        document_id: [GroundTruthPIIItem.from_serialized(person) for person in people]
        for document_id, people in serialized.items()
    }


def _validate_prediction_documents(
    predictions: Mapping[str, Sequence[PIIItem]], *, ground_truth: DocumentGroundTruth
) -> None:
    unknown = sorted(set(predictions) - set(ground_truth))
    if unknown:
        raise ValueError(f"Predictions contain unknown document IDs: {unknown}")
