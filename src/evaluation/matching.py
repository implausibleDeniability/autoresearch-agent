import itertools
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, Sequence, Tuple, TypeVar

from src.evaluation.models import PIIItem

SIMILARITY_THRESHOLD = 0.65
MINIMUM_FUZZY_LENGTH = 3
MATCH = 1
NO_RESULT = 0
MISMATCH = -1

MatchIndexes = Dict[int, int]
T = TypeVar("T")
MatchingFunction = Callable[[T, T], bool]


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


def match_people(predictions: Sequence[PIIItem], *, ground_truth: Sequence[PIIItem]) -> MatchIndexes:
    return _match_indexes(
        predictions,
        ground_truth=ground_truth,
        matching_functions=(_people_match_exactly, _people_match_approximately),
    )


def match_values(predictions: Sequence[str], *, ground_truth: Sequence[str]) -> MatchIndexes:
    return _match_indexes(
        predictions,
        ground_truth=ground_truth,
        matching_functions=(_values_match_exactly, _values_match_approximately),
    )


def _match_indexes(
    predictions: Sequence[T],
    *,
    ground_truth: Sequence[T],
    matching_functions: Sequence[MatchingFunction[T]],
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
    predictions: Sequence[T],
    *,
    ground_truth: Sequence[T],
    prediction_indexes: Sequence[int],
    ground_indexes: Sequence[int],
    matching_function: MatchingFunction[T],
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


def _people_match_exactly(prediction: PIIItem, ground_truth: PIIItem) -> bool:
    return all(
        _value_sets_match_exactly(predicted, expected)
        for predicted, expected in zip(_core_values(prediction), _core_values(ground_truth))
    )


def _people_match_approximately(prediction: PIIItem, ground_truth: PIIItem) -> bool:
    comparisons = [
        _compare_value_sets_approximately(predicted, expected)
        for predicted, expected in zip(_core_values(prediction), _core_values(ground_truth))
    ]
    return MATCH in comparisons and MISMATCH not in comparisons


def _core_values(person: PIIItem) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
    return person.first_name, person.last_name, person.email


def _value_sets_match_exactly(predictions: Sequence[str], ground_truth: Sequence[str]) -> bool:
    if not predictions and not ground_truth:
        return True
    predicted = {normalize_value(value) for value in predictions}
    expected = {normalize_value(value) for value in ground_truth}
    return bool(predicted & expected)


def _values_match_exactly(prediction: str, ground_truth: str) -> bool:
    return compare_values(prediction, ground_truth=ground_truth).normalized_exact


def _values_match_approximately(prediction: str, ground_truth: str) -> bool:
    return compare_values(prediction, ground_truth=ground_truth).result == MATCH


def _compare_value_sets_approximately(predictions: Sequence[str], ground_truth: Sequence[str]) -> int:
    comparisons = [
        _compare_values_approximately(predicted, expected)
        for predicted, expected in itertools.product(predictions, ground_truth)
    ]
    if MATCH in comparisons:
        return MATCH
    return MISMATCH if MISMATCH in comparisons else NO_RESULT


def _compare_values_approximately(prediction: str, ground_truth: str) -> int:
    return compare_values(prediction, ground_truth=ground_truth).result
