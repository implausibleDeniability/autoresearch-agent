from dataclasses import dataclass, fields
from typing import Dict, Mapping, Sequence, Tuple

PIIValues = Tuple[str, ...]
SerializedGroundTruthValue = str | Mapping[str, object]
GROUND_TRUTH_VALUE_FIELDS = frozenset({"canonical", "variants"})


@dataclass(frozen=True)
class PIIItem:
    first_name: PIIValues = ()
    middle_name: PIIValues = ()
    last_name: PIIValues = ()
    age: PIIValues = ()
    birthdate: PIIValues = ()
    phone: PIIValues = ()
    email: PIIValues = ()
    social_network_identifier: PIIValues = ()
    location: PIIValues = ()
    ssn: PIIValues = ()


@dataclass(frozen=True)
class GroundTruthValue:
    canonical: str
    variants: PIIValues = ()

    def __post_init__(self) -> None:
        if not self.canonical:
            raise ValueError("ground-truth canonical value must not be empty")
        if any(not variant for variant in self.variants):
            raise ValueError(f"ground-truth variants must not be empty: {self.variants}")
        if len(set(self.accepted_values)) != len(self.accepted_values):
            raise ValueError(f"ground-truth values must be unique: {self.accepted_values}")

    @property
    def accepted_values(self) -> PIIValues:
        return (self.canonical, *self.variants)

    @classmethod
    def from_serialized(cls, serialized: SerializedGroundTruthValue) -> "GroundTruthValue":
        if isinstance(serialized, str):
            return cls(canonical=serialized)
        _validate_mapping(serialized, context="ground-truth value")
        _validate_known_fields(
            serialized,
            expected=GROUND_TRUTH_VALUE_FIELDS,
            context="ground-truth value",
        )
        canonical = serialized.get("canonical")
        variants = serialized.get("variants", [])
        if not isinstance(canonical, str) or not isinstance(variants, list):
            raise TypeError(f"invalid ground-truth value: {serialized}")
        if any(not isinstance(variant, str) for variant in variants):
            raise TypeError(f"ground-truth variants must be strings: {serialized}")
        return cls(canonical=canonical, variants=tuple(variants))

    def serialize(self) -> str | Dict[str, object]:
        if not self.variants:
            return self.canonical
        return {"canonical": self.canonical, "variants": list(self.variants)}


GroundTruthValues = Tuple[GroundTruthValue, ...]


@dataclass(frozen=True)
class GroundTruthPIIItem:
    first_name: GroundTruthValues = ()
    middle_name: GroundTruthValues = ()
    last_name: GroundTruthValues = ()
    age: GroundTruthValues = ()
    birthdate: GroundTruthValues = ()
    phone: GroundTruthValues = ()
    email: GroundTruthValues = ()
    social_network_identifier: GroundTruthValues = ()
    location: GroundTruthValues = ()
    ssn: GroundTruthValues = ()

    @classmethod
    def from_serialized(
        cls, serialized: Mapping[str, Sequence[SerializedGroundTruthValue]]
    ) -> "GroundTruthPIIItem":
        _validate_mapping(serialized, context="ground-truth person")
        expected_fields = frozenset(field.name for field in fields(cls))
        _validate_known_fields(serialized, expected=expected_fields, context="ground-truth person")
        values = {
            field.name: _deserialize_ground_truth_values(
                serialized.get(field.name, []), field_name=field.name
            )
            for field in fields(cls)
        }
        return cls(**values)

    @classmethod
    def from_pii_item(cls, person: PIIItem) -> "GroundTruthPIIItem":
        values = {
            field.name: tuple(GroundTruthValue(canonical=value) for value in getattr(person, field.name))
            for field in fields(person)
        }
        return cls(**values)

    def serialize(self) -> Dict[str, object]:
        return {
            field.name: [value.serialize() for value in getattr(self, field.name)] for field in fields(self)
        }


def _validate_mapping(serialized: object, *, context: str) -> None:
    if not isinstance(serialized, Mapping):
        raise TypeError(f"{context} must be an object: {serialized!r}")


def _validate_known_fields(
    serialized: Mapping[str, object], *, expected: frozenset[str], context: str
) -> None:
    unknown = set(serialized) - expected
    if unknown:
        raise ValueError(f"unknown {context} fields: {sorted(unknown, key=repr)}")


def _deserialize_ground_truth_values(serialized: object, *, field_name: str) -> GroundTruthValues:
    if not isinstance(serialized, list):
        raise TypeError(f"ground-truth person field {field_name!r} must be a list: {serialized!r}")
    return tuple(GroundTruthValue.from_serialized(value) for value in serialized)
