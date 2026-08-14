import json
from pathlib import Path

import pytest

JUSTIN_EMAILS = {"schwabjustin@epa.gov", "schwab.justin@epa.gov"}


@pytest.mark.parametrize(
    ("dataset", "expected_people", "expected_values"),
    (("dev-19k", 73, 196), ("dev-87k", 92, 228), ("dev-205k", 534, 1334)),
)
def test_visible_dataset_consolidates_justin_schwab(dataset: str, expected_people: int, expected_values: int):
    # setup
    ground_truth = json.loads((Path("data") / dataset / "ground_truth.json").read_text())
    people = [person for document in ground_truth.values() for person in document]

    # operate
    justin_schwab = [
        person
        for person in ground_truth["xzbn0226"]
        if person["first_name"] == ["Justin"] and person["last_name"] == ["Schwab"]
    ]
    value_count = sum(len(values) for person in people for values in person.values())

    # check
    assert (len(people), value_count) == (expected_people, expected_values)
    assert len(justin_schwab) == 1
    assert set(justin_schwab[0]["email"]) == JUSTIN_EMAILS
