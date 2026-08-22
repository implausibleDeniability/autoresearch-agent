import itertools
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, Sequence, Tuple, TypeVar

from src.evaluation.models import GroundTruthPIIItem, GroundTruthValue, PIIItem

SIMILARITY_THRESHOLD = 0.65
MINIMUM_FUZZY_LENGTH = 3
MATCH = 1
NO_RESULT = 0
MISMATCH = -1

MatchIndexes = Dict[int, int]
Prediction = TypeVar("Prediction")
GroundTruth = TypeVar("GroundTruth")
MatchingFunction = Callable[[Prediction, GroundTruth], bool]


@dataclass(frozen=True)
class ValueComparison:
    normalized_exact: bool
    similarity: float
    result: int


def compare_values(prediction: str, *, ground_truth: str) -> ValueComparison:
    predicted = normalize_value(prediction)
    expected = normalize_value(ground_truth)
    if predicted == expected:
        return ValueComparison(normalized_exact=True, similarity=1.0, result=MATCH)
    similarity = SequenceMatcher(None, predicted, expected, autojunk=False).ratio()
    if len(predicted) < MINIMUM_FUZZY_LENGTH or len(expected) < MINIMUM_FUZZY_LENGTH:
        return ValueComparison(normalized_exact=False, similarity=similarity, result=NO_RESULT)
    result = MATCH if similarity >= SIMILARITY_THRESHOLD else MISMATCH
    return ValueComparison(normalized_exact=False, similarity=similarity, result=result)


def normalize_value(value: str) -> str:
    return value.lower().strip().rstrip(".")


def similarity_length_bounds(value: str) -> Tuple[int, int]:
    target_length = len(normalize_value(value))
    minimum = math.ceil(SIMILARITY_THRESHOLD * target_length / (2 - SIMILARITY_THRESHOLD))
    maximum = math.floor(target_length * (2 - SIMILARITY_THRESHOLD) / SIMILARITY_THRESHOLD)
    return minimum, maximum


def match_people(
    predictions: Sequence[PIIItem], *, ground_truth: Sequence[GroundTruthPIIItem]
) -> MatchIndexes:
    return _match_indexes(
        predictions,
        ground_truth=ground_truth,
        matching_functions=(_people_match_exactly, _people_match_approximately),
    )


def match_values(
    predictions: Sequence[str], *, ground_truth: Sequence[GroundTruthValue | str]
) -> MatchIndexes:
    labels = tuple(
        value if isinstance(value, GroundTruthValue) else GroundTruthValue(canonical=value)
        for value in ground_truth
    )
    return _match_indexes(
        predictions,
        ground_truth=labels,
        matching_functions=(_values_match_exactly, _values_match_approximately),
    )


def _match_indexes(
    predictions: Sequence[Prediction],
    *,
    ground_truth: Sequence[GroundTruth],
    matching_functions: Sequence[MatchingFunction[Prediction, GroundTruth]],
) -> MatchIndexes:
    unmatched_predictions = list(range(len(predictions)))
    unmatched_ground = list(range(len(ground_truth)))
    matches: MatchIndexes = {}
    for matching_function in matching_functions:
        current = _match_step(
            predictions,
            ground_truth=ground_truth,
            prediction_indexes=unmatched_predictions,
            ground_indexes=unmatched_ground,
            matching_function=matching_function,
        )
        matches.update(current)
        unmatched_predictions = [index for index in unmatched_predictions if index not in current]
        unmatched_ground = [index for index in unmatched_ground if index not in current.values()]
    return matches


def _match_step(
    predictions: Sequence[Prediction],
    *,
    ground_truth: Sequence[GroundTruth],
    prediction_indexes: Sequence[int],
    ground_indexes: Sequence[int],
    matching_function: MatchingFunction[Prediction, GroundTruth],
) -> MatchIndexes:
    matches: MatchIndexes = {}
    available_ground = set(ground_indexes)
    for prediction_index in prediction_indexes:
        candidates = [
            index
            for index in available_ground
            if matching_function(predictions[prediction_index], ground_truth[index])
        ]
        if len(candidates) == 1:
            matches[prediction_index] = candidates[0]
            available_ground.remove(candidates[0])
    return matches


def _people_match_exactly(prediction: PIIItem, ground_truth: GroundTruthPIIItem) -> bool:
    return all(
        _value_sets_match_exactly(predicted, expected)
        for predicted, expected in zip(
            _prediction_core_values(prediction),
            _ground_truth_core_values(ground_truth),
        )
    )


def _people_match_approximately(prediction: PIIItem, ground_truth: GroundTruthPIIItem) -> bool:
    comparisons = [
        _compare_value_sets_approximately(predicted, expected)
        for predicted, expected in zip(
            _prediction_core_values(prediction),
            _ground_truth_core_values(ground_truth),
        )
    ]
    return MATCH in comparisons and MISMATCH not in comparisons


def _prediction_core_values(person: PIIItem) -> Tuple[Sequence[str], ...]:
    return person.first_name, person.last_name, person.email


def _ground_truth_core_values(
    person: GroundTruthPIIItem,
) -> Tuple[Sequence[GroundTruthValue], ...]:
    return person.first_name, person.last_name, person.email


def _value_sets_match_exactly(predictions: Sequence[str], ground_truth: Sequence[GroundTruthValue]) -> bool:
    if not predictions and not ground_truth:
        return True
    return any(
        _values_match_exactly(prediction, expected)
        for prediction, expected in itertools.product(predictions, ground_truth)
    )


def _values_match_exactly(prediction: str, ground_truth: GroundTruthValue) -> bool:
    return any(
        compare_values(prediction, ground_truth=accepted).normalized_exact
        for accepted in ground_truth.accepted_values
    )


def _values_match_approximately(prediction: str, ground_truth: GroundTruthValue) -> bool:
    return _compare_value_approximately(prediction, ground_truth) == MATCH


def _compare_value_sets_approximately(
    predictions: Sequence[str], ground_truth: Sequence[GroundTruthValue]
) -> int:
    comparisons = [
        _compare_value_approximately(predicted, expected)
        for predicted, expected in itertools.product(predictions, ground_truth)
    ]
    if MATCH in comparisons:
        return MATCH
    return MISMATCH if MISMATCH in comparisons else NO_RESULT


def _compare_value_approximately(prediction: str, ground_truth: GroundTruthValue) -> int:
    comparisons = [
        compare_values(prediction, ground_truth=accepted).result for accepted in ground_truth.accepted_values
    ]
    if MATCH in comparisons:
        return MATCH
    return MISMATCH if MISMATCH in comparisons else NO_RESULT
