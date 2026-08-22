from dataclasses import dataclass
from typing import Dict, Tuple

from src.evaluation.models import GroundTruthPIIItem, PIIItem

RECALL_WEIGHT = 5


@dataclass(frozen=True)
class EntityMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def __add__(self, other: "EntityMetrics") -> "EntityMetrics":
        return EntityMetrics(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
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
    def f_score(self) -> float:
        weighted_true_positive = (1 + RECALL_WEIGHT) * self.true_positive
        total = weighted_true_positive + self.false_positive + RECALL_WEIGHT * self.false_negative
        return weighted_true_positive / total if total else 0.0


@dataclass(frozen=True)
class ValueReference:
    person_index: int
    value_index: int
    value: str
    variants: Tuple[str, ...] = ()
    optional: bool = False

    @property
    def accepted_values(self) -> Tuple[str, ...]:
        return self.value, *self.variants


@dataclass(frozen=True)
class ValueMatch:
    prediction: ValueReference
    ground_truth: ValueReference


@dataclass(frozen=True)
class FieldEvaluation:
    field: str
    matches: Tuple[ValueMatch, ...]
    ignored_optional_matches: Tuple[ValueMatch, ...]
    false_positives: Tuple[ValueReference, ...]
    false_negatives: Tuple[ValueReference, ...]
    unmatched_optional_values: Tuple[ValueReference, ...]

    @property
    def metrics(self) -> EntityMetrics:
        return EntityMetrics(
            true_positive=len(self.matches),
            false_positive=len(self.false_positives),
            false_negative=len(self.false_negatives),
        )


@dataclass(frozen=True)
class DocumentEvaluation:
    document_id: str
    predictions: Tuple[PIIItem, ...]
    ground_truth: Tuple[GroundTruthPIIItem, ...]
    person_matches: Tuple[Tuple[int, int], ...]
    unmatched_prediction_indexes: Tuple[int, ...]
    unmatched_ground_truth_indexes: Tuple[int, ...]
    fields: Tuple[FieldEvaluation, ...]

    @property
    def metrics(self) -> EntityMetrics:
        return sum((field.metrics for field in self.fields), EntityMetrics())


@dataclass(frozen=True)
class EvaluationTrace:
    documents: Tuple[DocumentEvaluation, ...]

    @property
    def metrics(self) -> EntityMetrics:
        return sum((document.metrics for document in self.documents), EntityMetrics())

    @property
    def field_metrics(self) -> Dict[str, EntityMetrics]:
        names = (field.field for document in self.documents for field in document.fields)
        return {
            name: sum(
                (
                    field.metrics
                    for document in self.documents
                    for field in document.fields
                    if field.field == name
                ),
                EntityMetrics(),
            )
            for name in dict.fromkeys(names)
        }
