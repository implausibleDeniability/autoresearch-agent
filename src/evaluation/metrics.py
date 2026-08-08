import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set

from src.evaluation.matching import MatchIndexes, match_people, match_values
from src.evaluation.models import PIIItem

DEFAULT_GROUND_TRUTH_PATH = Path("data/dev-5k/ground_truth.json")
RECALL_WEIGHT = 5
DocumentPII = Dict[str, List[PIIItem]]


@dataclass(frozen=True)
class _Counts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def __add__(self, other: "_Counts") -> "_Counts":
        return _Counts(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
        )

    @property
    def f_score(self) -> float:
        weighted_true_positive = (1 + RECALL_WEIGHT) * self.true_positive
        total = weighted_true_positive + self.false_positive + RECALL_WEIGHT * self.false_negative
        return weighted_true_positive / total if total else 0.0


def evaluate(
    predictions: Mapping[str, Sequence[PIIItem]],
    *,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> float:
    ground_truth = _load_ground_truth(ground_truth_path)
    _validate_prediction_documents(predictions, ground_truth=ground_truth)
    counts = [_evaluate_document(predictions.get(key, ()), ground_truth[key]) for key in ground_truth]
    return _sum_counts(counts).f_score


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


def _sum_counts(counts: Sequence[_Counts]) -> _Counts:
    return sum(counts, _Counts())


def _evaluate_document(predictions: Sequence[PIIItem], ground_truth: Sequence[PIIItem]) -> _Counts:
    matches = match_people(predictions, ground_truth=ground_truth)
    return _pii_counts(predictions, ground_truth=ground_truth, matches=matches)


def _pii_counts(
    predictions: Sequence[PIIItem],
    *,
    ground_truth: Sequence[PIIItem],
    matches: MatchIndexes,
) -> _Counts:
    matched = _matched_pii_counts(predictions, ground_truth=ground_truth, matches=matches)
    unmatched_prediction = _unmatched_value_count(predictions, matched_indexes=set(matches))
    unmatched_ground = _unmatched_value_count(ground_truth, matched_indexes=set(matches.values()))
    return matched + _Counts(false_positive=unmatched_prediction, false_negative=unmatched_ground)


def _matched_pii_counts(
    predictions: Sequence[PIIItem],
    *,
    ground_truth: Sequence[PIIItem],
    matches: MatchIndexes,
) -> _Counts:
    counts = _Counts()
    for prediction_index, ground_index in matches.items():
        counts += _person_pii_counts(predictions[prediction_index], ground_truth[ground_index])
    return counts


def _person_pii_counts(prediction: PIIItem, ground_truth: PIIItem) -> _Counts:
    counts = _Counts()
    ground_values = asdict(ground_truth)
    for field_name, predicted_values in asdict(prediction).items():
        counts += _value_counts(predicted_values, ground_truth=ground_values[field_name])
    return counts


def _value_counts(predictions: Sequence[str], *, ground_truth: Sequence[str]) -> _Counts:
    matches = match_values(predictions, ground_truth=ground_truth)
    return _Counts(
        true_positive=len(matches),
        false_positive=len(predictions) - len(matches),
        false_negative=len(ground_truth) - len(matches),
    )


def _unmatched_value_count(people: Sequence[PIIItem], *, matched_indexes: Set[int]) -> int:
    return sum(
        _person_value_count(person) for index, person in enumerate(people) if index not in matched_indexes
    )


def _person_value_count(person: PIIItem) -> int:
    return sum(len(values) for values in asdict(person).values())
