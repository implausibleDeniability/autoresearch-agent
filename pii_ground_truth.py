import csv
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from pii_item import PIIItem


DEFAULT_GROUND_TRUTH_PATH = Path("data/dev/ground_truth_pii.csv")
FIELD_NAME_MAPPING = {
    "first_name": "first_name",
    "middle_name": "middle_name",
    "last_name": "last_name",
    "age": "age",
    "birthdate": "birthdate",
    "phone": "phone",
    "email": "email",
    "personal_email": "email",
    "work_email": "email",
    "social_network_identifier": "social_network_identifier",
    "telegram_alias": "social_network_identifier",
    "messenger_id": "social_network_identifier",
    "address": "location",
    "location": "location",
    "ssn": "ssn",
    "social_security_number": "ssn",
}

DocumentPII = Dict[str, List[PIIItem]]


def load_ground_truth(path: Path) -> DocumentPII:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        _validate_columns(reader.fieldnames, path=path)
        return _parse_rows(list(reader), path=path)


def _validate_columns(columns: Sequence[str], *, path: Path) -> None:
    required = {"document_id", "answer"}
    if columns is None or not required.issubset(columns):
        raise ValueError(f"Ground truth at {path} must contain columns {sorted(required)}; got {columns}")


def _parse_rows(rows: Sequence[Mapping[str, str]], *, path: Path) -> DocumentPII:
    result: DocumentPII = {}
    for row in rows:
        document_id = row["document_id"]
        if not document_id or document_id in result:
            raise ValueError(f"Invalid or duplicate document_id {document_id!r} in {path}")
        result[document_id] = _parse_answer(row["answer"])
    return result


def _parse_answer(answer: str) -> List[PIIItem]:
    segments = [segment.strip() for segment in answer.split(";") if segment.strip()]
    return [_parse_person(segment) for segment in segments]


def _parse_person(segment: str) -> PIIItem:
    values = {field.name: [] for field in fields(PIIItem)}
    for line in segment.splitlines():
        _append_value(values, line=line.strip())
    if not any(values.values()):
        raise ValueError(f"Ground-truth person has no supported PII values: {segment!r}")
    return PIIItem(**{field_name: tuple(field_values) for field_name, field_values in values.items()})


def _append_value(values: Dict[str, List[str]], *, line: str) -> None:
    if not line or line == "Undefined":
        return
    source_field, separator, value = line.partition(": ")
    if not separator or not source_field or not value:
        raise ValueError(f"Malformed ground-truth line: {line!r}")
    if source_field not in FIELD_NAME_MAPPING:
        raise ValueError(f"Unsupported ground-truth field: {source_field!r}")
    values[FIELD_NAME_MAPPING[source_field]].append(value)
