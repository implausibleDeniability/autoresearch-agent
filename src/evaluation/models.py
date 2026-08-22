from dataclasses import dataclass, fields
from typing import Dict, Mapping, Sequence, Tuple

PIIValues = Tuple[str, ...]
SerializedGroundTruthValue = str | Mapping[str, object]
GROUND_TRUTH_VALUE_FIELDS = frozenset({"canonical", "variants", "optional"})
OPTIONAL_GROUND_TRUTH_FIELDS = frozenset({"first_name", "middle_name", "last_name"})


class GroundTruthPersonValidationError(ValueError):
    pass


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
    optional: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.optional, bool):
            raise TypeError(f"ground-truth optional marker must be a boolean: {self.optional!r}")
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
    def from_serialized(
        cls,
        serialized: SerializedGroundTruthValue,
        *,
        context: str = "ground-truth value",
    ) -> "GroundTruthValue":
        if isinstance(serialized, str):
            return cls(canonical=serialized)
        _validate_mapping(serialized, context=context)
        _validate_known_fields(
            serialized,
            expected=GROUND_TRUTH_VALUE_FIELDS,
            context=context,
        )
        canonical = serialized.get("canonical")
        variants = serialized.get("variants", [])
        optional = serialized.get("optional", False)
        if not isinstance(canonical, str) or not isinstance(variants, list):
            raise TypeError(f"{context} must contain a string canonical and list variants")
        if any(not isinstance(variant, str) for variant in variants):
            raise TypeError(f"{context} variants must be strings")
        if not isinstance(optional, bool):
            raise TypeError(f"{context}.optional must be a boolean: {optional!r}")
        return cls(canonical=canonical, variants=tuple(variants), optional=optional)

    def serialize(self) -> str | Dict[str, object]:
        if not self.variants and not self.optional:
            return self.canonical
        serialized: Dict[str, object] = {"canonical": self.canonical}
        if self.variants:
            serialized["variants"] = list(self.variants)
        if self.optional:
            serialized["optional"] = True
        return serialized


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

    def __post_init__(self) -> None:
        optional_fields = self._optional_fields()
        unsupported = tuple(
            field_name for field_name in optional_fields if field_name not in OPTIONAL_GROUND_TRUTH_FIELDS
        )
        if unsupported:
            raise GroundTruthPersonValidationError(
                f"optional ground-truth values are not allowed in fields: {unsupported}"
            )
        if optional_fields and not self.email:
            raise GroundTruthPersonValidationError(
                "optional ground-truth names require an email on the same person"
            )

    def _optional_fields(self) -> Tuple[str, ...]:
        return tuple(
            field.name for field in fields(self) if any(value.optional for value in getattr(self, field.name))
        )

    @classmethod
    def from_serialized(
        cls,
        serialized: Mapping[str, Sequence[SerializedGroundTruthValue]],
        *,
        context: str = "ground-truth person",
    ) -> "GroundTruthPIIItem":
        _validate_mapping(serialized, context=context)
        expected_fields = frozenset(field.name for field in fields(cls))
        _validate_known_fields(serialized, expected=expected_fields, context=context)
        values = {
            field.name: _deserialize_ground_truth_values(
                serialized.get(field.name, []),
                context=f"{context}.{field.name}",
            )
            for field in fields(cls)
        }
        try:
            return cls(**values)
        except GroundTruthPersonValidationError as error:
            raise GroundTruthPersonValidationError(f"{context}: {error}") from error

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


def _deserialize_ground_truth_values(serialized: object, *, context: str) -> GroundTruthValues:
    if not isinstance(serialized, list):
        raise TypeError(f"{context} must be a list: {serialized!r}")
    return tuple(
        GroundTruthValue.from_serialized(value, context=f"{context}[{index}]")
        for index, value in enumerate(serialized)
    )
