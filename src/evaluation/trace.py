from dataclasses import dataclass, field, fields
from typing import List, Mapping, Sequence, Tuple

from src.evaluation.matching import MatchIndexes, match_people, match_value_groups
from src.evaluation.models import GroundTruthPIIItem, GroundTruthValue, PIIItem
from src.evaluation.results import (
    DocumentEvaluation,
    EvaluationTrace,
    FieldEvaluation,
    ValueMatch,
    ValueReference,
)

GroundTruthInput = GroundTruthPIIItem | PIIItem


@dataclass
class FieldValueLedger:
    matches: List[ValueMatch] = field(default_factory=list)
    ignored_optional_matches: List[ValueMatch] = field(default_factory=list)
    false_positives: List[ValueReference] = field(default_factory=list)
    false_negatives: List[ValueReference] = field(default_factory=list)
    unmatched_optional_values: List[ValueReference] = field(default_factory=list)


@dataclass(frozen=True)
class GroundTruthValuePartitions:
    required: Tuple[ValueReference, ...]
    optional: Tuple[ValueReference, ...]


def build_evaluation_trace(
    predictions: Mapping[str, Sequence[PIIItem]],
    *,
    ground_truth: Mapping[str, Sequence[GroundTruthInput]],
) -> EvaluationTrace:
    documents = (
        _evaluate_document(
            document_id,
            predictions=predictions.get(document_id, ()),
            ground_truth=_coerce_ground_truth(expected),
        )
        for document_id, expected in ground_truth.items()
    )
    return EvaluationTrace(tuple(documents))


def _coerce_ground_truth(people: Sequence[GroundTruthInput]) -> Tuple[GroundTruthPIIItem, ...]:
    return tuple(
        person if isinstance(person, GroundTruthPIIItem) else GroundTruthPIIItem.from_pii_item(person)
        for person in people
    )


def _evaluate_document(
    document_id: str,
    *,
    predictions: Sequence[PIIItem],
    ground_truth: Sequence[GroundTruthPIIItem],
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
    ground_truth: Sequence[GroundTruthPIIItem],
    person_matches: MatchIndexes,
) -> FieldEvaluation:
    ledger = _matched_people_values(
        field_name,
        predictions=predictions,
        ground_truth=ground_truth,
        person_matches=person_matches,
    )
    ledger.false_positives.extend(
        _unmatched_prediction_people_values(field_name, predictions, matched=set(person_matches))
    )
    unmatched = _unmatched_ground_truth_people_values(
        field_name,
        ground_truth,
        matched=set(person_matches.values()),
    )
    ledger.false_negatives.extend(unmatched.required)
    ledger.unmatched_optional_values.extend(unmatched.optional)
    return FieldEvaluation(
        field=field_name,
        matches=tuple(ledger.matches),
        ignored_optional_matches=tuple(ledger.ignored_optional_matches),
        false_positives=tuple(ledger.false_positives),
        false_negatives=tuple(ledger.false_negatives),
        unmatched_optional_values=tuple(ledger.unmatched_optional_values),
    )


def _matched_people_values(
    field_name: str,
    *,
    predictions: Sequence[PIIItem],
    ground_truth: Sequence[GroundTruthPIIItem],
    person_matches: MatchIndexes,
) -> FieldValueLedger:
    ledger = FieldValueLedger()
    for prediction_person_index, ground_person_index in person_matches.items():
        _add_matched_person_values(
            ledger,
            field_name=field_name,
            prediction_person_index=prediction_person_index,
            ground_person_index=ground_person_index,
            predictions=predictions,
            ground_truth=ground_truth,
        )
    return ledger


def _add_matched_person_values(
    ledger: FieldValueLedger,
    *,
    field_name: str,
    prediction_person_index: int,
    ground_person_index: int,
    predictions: Sequence[PIIItem],
    ground_truth: Sequence[GroundTruthPIIItem],
) -> None:
    predicted_values = getattr(predictions[prediction_person_index], field_name)
    expected_values = getattr(ground_truth[ground_person_index], field_name)
    value_matches = match_value_groups(predicted_values, ground_truth=expected_values)
    ledger.matches.extend(
        _value_matches(
            prediction_person_index=prediction_person_index,
            ground_person_index=ground_person_index,
            predicted_values=predicted_values,
            expected_values=expected_values,
            matches=value_matches.required,
        )
    )
    ledger.ignored_optional_matches.extend(
        _value_matches(
            prediction_person_index=prediction_person_index,
            ground_person_index=ground_person_index,
            predicted_values=predicted_values,
            expected_values=expected_values,
            matches=value_matches.optional,
        )
    )
    matched_predictions = set(value_matches.required) | set(value_matches.optional)
    ledger.false_positives.extend(
        _unmatched_prediction_values(
            prediction_person_index,
            predicted_values,
            matched=matched_predictions,
        )
    )
    matched_ground = set(value_matches.required.values()) | set(value_matches.optional.values())
    unmatched = _unmatched_ground_truth_values(
        ground_person_index,
        expected_values,
        matched=matched_ground,
    )
    ledger.false_negatives.extend(unmatched.required)
    ledger.unmatched_optional_values.extend(unmatched.optional)


def _value_matches(
    *,
    prediction_person_index: int,
    ground_person_index: int,
    predicted_values: Sequence[str],
    expected_values: Sequence[GroundTruthValue],
    matches: MatchIndexes,
) -> List[ValueMatch]:
    return [
        ValueMatch(
            prediction=_prediction_value_reference(
                prediction_person_index,
                value_index=prediction_index,
                value=predicted_values[prediction_index],
            ),
            ground_truth=_ground_truth_value_reference(
                ground_person_index,
                value_index=ground_index,
                value=expected_values[ground_index],
            ),
        )
        for prediction_index, ground_index in matches.items()
    ]


def _unmatched_prediction_people_values(
    field_name: str, people: Sequence[PIIItem], *, matched: set[int]
) -> List[ValueReference]:
    return [
        _prediction_value_reference(person_index, value_index=value_index, value=value)
        for person_index, person in enumerate(people)
        if person_index not in matched
        for value_index, value in enumerate(getattr(person, field_name))
    ]


def _unmatched_ground_truth_people_values(
    field_name: str, people: Sequence[GroundTruthPIIItem], *, matched: set[int]
) -> GroundTruthValuePartitions:
    references = [
        _ground_truth_value_reference(person_index, value_index=value_index, value=value)
        for person_index, person in enumerate(people)
        if person_index not in matched
        for value_index, value in enumerate(getattr(person, field_name))
    ]
    return _partition_ground_truth_references(references)


def _unmatched_prediction_values(
    person_index: int, values: Sequence[str], *, matched: set[int]
) -> List[ValueReference]:
    return [
        _prediction_value_reference(person_index, value_index=value_index, value=value)
        for value_index, value in enumerate(values)
        if value_index not in matched
    ]


def _unmatched_ground_truth_values(
    person_index: int, values: Sequence[GroundTruthValue], *, matched: set[int]
) -> GroundTruthValuePartitions:
    references = [
        _ground_truth_value_reference(person_index, value_index=value_index, value=value)
        for value_index, value in enumerate(values)
        if value_index not in matched
    ]
    return _partition_ground_truth_references(references)


def _partition_ground_truth_references(
    references: Sequence[ValueReference],
) -> GroundTruthValuePartitions:
    return GroundTruthValuePartitions(
        required=tuple(reference for reference in references if not reference.optional),
        optional=tuple(reference for reference in references if reference.optional),
    )


def _prediction_value_reference(person_index: int, *, value_index: int, value: str) -> ValueReference:
    return ValueReference(person_index=person_index, value_index=value_index, value=value)


def _ground_truth_value_reference(
    person_index: int, *, value_index: int, value: GroundTruthValue
) -> ValueReference:
    return ValueReference(
        person_index=person_index,
        value_index=value_index,
        value=value.canonical,
        variants=value.variants,
        optional=value.optional,
    )
