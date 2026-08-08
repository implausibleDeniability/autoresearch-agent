from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, Set

from pii_ground_truth import DEFAULT_GROUND_TRUTH_PATH, DocumentPII, load_ground_truth
from pii_matching import MatchIndexes, match_people, match_values
from pii_item import PIIItem


@dataclass(frozen=True)
class Metrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
            true_negative=self.true_negative + other.true_negative,
        )

    @property
    def precision(self) -> float:
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positive + self.false_negative
        return self.true_positive / total if total else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.false_positive + self.false_negative + self.true_negative
        return (self.true_positive + self.true_negative) / total if total else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    people: Metrics
    entities: Metrics
    documents: Metrics

    def __add__(self, other: "EvaluationResult") -> "EvaluationResult":
        return EvaluationResult(
            people=self.people + other.people,
            entities=self.entities + other.entities,
            documents=self.documents + other.documents,
        )


def evaluate(
    predictions: Mapping[str, Sequence[PIIItem]],
    *,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
) -> EvaluationResult:
    ground_truth = load_ground_truth(ground_truth_path)
    _validate_prediction_documents(predictions, ground_truth=ground_truth)
    results = [_evaluate_document(predictions.get(key, ()), ground_truth[key]) for key in ground_truth]
    return _sum_results(results)


def _validate_prediction_documents(
    predictions: Mapping[str, Sequence[PIIItem]], *, ground_truth: DocumentPII
) -> None:
    unknown = sorted(set(predictions) - set(ground_truth))
    if unknown:
        raise ValueError(f"Predictions contain unknown document IDs: {unknown}")


def _sum_results(results: Sequence[EvaluationResult]) -> EvaluationResult:
    total = EvaluationResult(people=Metrics(), entities=Metrics(), documents=Metrics())
    for result in results:
        total += result
    return total


def _evaluate_document(predictions: Sequence[PIIItem], ground_truth: Sequence[PIIItem]) -> EvaluationResult:
    matches = match_people(predictions, ground_truth=ground_truth)
    return EvaluationResult(
        people=_people_metrics(predictions, ground_truth=ground_truth, matches=matches),
        entities=_entity_metrics(predictions, ground_truth=ground_truth, matches=matches),
        documents=_document_metrics(predictions, ground_truth=ground_truth),
    )


def _people_metrics(
    predictions: Sequence[PIIItem],
    *,
    ground_truth: Sequence[PIIItem],
    matches: MatchIndexes,
) -> Metrics:
    return Metrics(
        true_positive=len(matches),
        false_positive=len(predictions) - len(matches),
        false_negative=len(ground_truth) - len(matches),
    )


def _entity_metrics(
    predictions: Sequence[PIIItem],
    *,
    ground_truth: Sequence[PIIItem],
    matches: MatchIndexes,
) -> Metrics:
    matched = _matched_entity_metrics(predictions, ground_truth=ground_truth, matches=matches)
    unmatched_prediction = _unmatched_value_count(predictions, matched_indexes=set(matches))
    unmatched_ground = _unmatched_value_count(ground_truth, matched_indexes=set(matches.values()))
    return matched + Metrics(false_positive=unmatched_prediction, false_negative=unmatched_ground)


def _matched_entity_metrics(
    predictions: Sequence[PIIItem],
    *,
    ground_truth: Sequence[PIIItem],
    matches: MatchIndexes,
) -> Metrics:
    metrics = Metrics()
    for prediction_index, ground_index in matches.items():
        metrics += _person_entity_metrics(predictions[prediction_index], ground_truth[ground_index])
    return metrics


def _person_entity_metrics(prediction: PIIItem, ground_truth: PIIItem) -> Metrics:
    metrics = Metrics()
    ground_values = asdict(ground_truth)
    for field_name, predicted_values in asdict(prediction).items():
        metrics += _value_metrics(predicted_values, ground_truth=ground_values[field_name])
    return metrics


def _value_metrics(predictions: Sequence[str], *, ground_truth: Sequence[str]) -> Metrics:
    matches = match_values(predictions, ground_truth=ground_truth)
    return Metrics(
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


def _document_metrics(predictions: Sequence[PIIItem], *, ground_truth: Sequence[PIIItem]) -> Metrics:
    predicted = bool(predictions)
    expected = bool(ground_truth)
    return Metrics(
        true_positive=int(predicted and expected),
        false_positive=int(predicted and not expected),
        false_negative=int(not predicted and expected),
        true_negative=int(not predicted and not expected),
    )
