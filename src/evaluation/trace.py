from dataclasses import fields
from typing import List, Mapping, Sequence, Tuple

from src.evaluation.matching import MatchIndexes, match_people, match_values
from src.evaluation.models import PIIItem
from src.evaluation.results import (
    DocumentEvaluation,
    EvaluationTrace,
    FieldEvaluation,
    ValueMatch,
    ValueReference,
)


def build_evaluation_trace(
    predictions: Mapping[str, Sequence[PIIItem]], *, ground_truth: Mapping[str, Sequence[PIIItem]]
) -> EvaluationTrace:
    documents = (
        _evaluate_document(
            document_id,
            predictions=predictions.get(document_id, ()),
            ground_truth=expected,
        )
        for document_id, expected in ground_truth.items()
    )
    return EvaluationTrace(tuple(documents))


def _evaluate_document(
    document_id: str, *, predictions: Sequence[PIIItem], ground_truth: Sequence[PIIItem]
) -> DocumentEvaluation:
    person_matches = match_people(predictions, ground_truth=ground_truth)
    prediction_indexes = tuple(index for index in range(len(predictions)) if index not in person_matches)
    matched_ground_indexes = set(person_matches.values())
    ground_indexes = tuple(index for index in range(len(ground_truth)) if index not in matched_ground_indexes)
    field_results = tuple(
        _evaluate_field(
            field.name,
            predictions=predictions,
            ground_truth=ground_truth,
            person_matches=person_matches,
        )
        for field in fields(PIIItem)
    )
    return DocumentEvaluation(
        document_id=document_id,
        predictions=tuple(predictions),
        ground_truth=tuple(ground_truth),
        person_matches=tuple(person_matches.items()),
        unmatched_prediction_indexes=prediction_indexes,
        unmatched_ground_truth_indexes=ground_indexes,
        fields=field_results,
    )


def _evaluate_field(
    field_name: str,
    *,
    predictions: Sequence[PIIItem],
    ground_truth: Sequence[PIIItem],
    person_matches: MatchIndexes,
) -> FieldEvaluation:
    matches, false_positives, false_negatives = _matched_people_values(
        field_name,
        predictions=predictions,
        ground_truth=ground_truth,
        person_matches=person_matches,
    )
    false_positives.extend(_unmatched_people_values(field_name, predictions, matched=set(person_matches)))
    false_negatives.extend(
        _unmatched_people_values(field_name, ground_truth, matched=set(person_matches.values()))
    )
    return FieldEvaluation(
        field=field_name,
        matches=tuple(matches),
        false_positives=tuple(false_positives),
        false_negatives=tuple(false_negatives),
    )


def _matched_people_values(
    field_name: str,
    *,
    predictions: Sequence[PIIItem],
    ground_truth: Sequence[PIIItem],
    person_matches: MatchIndexes,
) -> Tuple[List[ValueMatch], List[ValueReference], List[ValueReference]]:
    matches: List[ValueMatch] = []
    false_positives: List[ValueReference] = []
    false_negatives: List[ValueReference] = []
    for prediction_person_index, ground_person_index in person_matches.items():
        predicted_values = getattr(predictions[prediction_person_index], field_name)
        expected_values = getattr(ground_truth[ground_person_index], field_name)
        value_matches = match_values(predicted_values, ground_truth=expected_values)
        matches.extend(
            ValueMatch(
                prediction=_value_reference(prediction_person_index, predicted_values, prediction_index),
                ground_truth=_value_reference(ground_person_index, expected_values, ground_index),
            )
            for prediction_index, ground_index in value_matches.items()
        )
        false_positives.extend(
            _unmatched_values(prediction_person_index, predicted_values, matched=set(value_matches))
        )
        false_negatives.extend(
            _unmatched_values(ground_person_index, expected_values, matched=set(value_matches.values()))
        )
    return matches, false_positives, false_negatives


def _unmatched_people_values(
    field_name: str, people: Sequence[PIIItem], *, matched: set[int]
) -> List[ValueReference]:
    return [
        ValueReference(person_index=person_index, value_index=value_index, value=value)
        for person_index, person in enumerate(people)
        if person_index not in matched
        for value_index, value in enumerate(getattr(person, field_name))
    ]


def _unmatched_values(person_index: int, values: Sequence[str], *, matched: set[int]) -> List[ValueReference]:
    return [
        _value_reference(person_index, values, value_index)
        for value_index in range(len(values))
        if value_index not in matched
    ]


def _value_reference(person_index: int, values: Sequence[str], value_index: int) -> ValueReference:
    return ValueReference(person_index=person_index, value_index=value_index, value=values[value_index])
