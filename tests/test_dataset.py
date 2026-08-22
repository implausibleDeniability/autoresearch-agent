import json
from pathlib import Path

import pytest

from src.evaluation.models import GroundTruthPIIItem

JUSTIN_EMAILS = {"schwabjustin@epa.gov", "schwab.justin@epa.gov"}
NAME_FIELDS = ("first_name", "middle_name", "last_name")


@pytest.mark.parametrize(
    ("dataset", "expected_people", "expected_values", "expected_negative"),
    (
        ("dev-19k", 74, 203, 9),
        ("dev-87k", 93, 238, 11),
        ("dev-205k", 535, 1329, 45),
    ),
)
def test_visible_dataset_counts_and_consolidates_justin_schwab(
    dataset: str, expected_people: int, expected_values: int, expected_negative: int
):
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
    negative_count = sum(not people for people in ground_truth.values())

    # check
    assert (len(people), value_count, negative_count) == (
        expected_people,
        expected_values,
        expected_negative,
    )
    assert len(justin_schwab) == 1
    assert set(justin_schwab[0]["email"]) == JUSTIN_EMAILS


@pytest.mark.parametrize("dataset", ["dev-19k", "dev-87k", "dev-205k"])
def test_visible_dataset_contains_corrected_shared_labels(dataset: str):
    ground_truth = json.loads((Path("data") / dataset / "ground_truth.json").read_text())

    michael = _person(ground_truth, document="nrbn0226", first_name="Michael")
    jennifer = _person(ground_truth, document="nrbn0226", first_name="Jennifer")
    elliot = _person(ground_truth, document="yrcg0257", first_name="Elliot")
    laura = _person(ground_truth, document="rpmw0257", first_name="Laura")
    shannon = _person(ground_truth, document="tybn0226", first_name="Shannon")
    byron = _person(ground_truth, document="tybn0226", first_name="Byron")
    kevin = _person(ground_truth, document="xzbn0226", first_name="Kevin")

    assert michael["phone"] == ["513-558-7949", "419-892-2502"]
    assert michael["location"] == ["160 Panzeca Way Cincinnati OH 45267-0056"]
    assert jennifer["phone"] == ["(202) 249-6732", "(202) 330-5646"]
    assert jennifer["location"][0]["variants"] == ["700 2nd9reet, NE | Washington, DC| 20002"]
    assert elliot["location"] == ["California, United States"]
    assert laura["phone"] == ["1 973 549 1808", "1 973 549 6808"]
    assert shannon["email"][0]["variants"] == ["kenny.shannon@8pa.gov"]
    assert byron["email"][0]["variants"] == ["fbrown.byron@epa.gov"]
    assert kevin["phone"] == ["202-564-8040", "202-564-5551"]
    assert ground_truth["pzvv0257"][0]["first_name"] == ["Craig"]


@pytest.mark.parametrize("dataset", ["dev-87k", "dev-205k"])
def test_extended_dataset_contains_corrected_michael_dourson_labels(dataset: str):
    ground_truth = json.loads((Path("data") / dataset / "ground_truth.json").read_text())

    michael = _person(ground_truth, document="rtbn0226", first_name="Michael")

    assert michael["phone"] == ["513-558-7949", "419-892-2502"]
    assert michael["email"][1]["variants"] == ["michael.dourson@ucedu"]
    assert michael["location"] == ["160 Panzeca Way Cincinnati OH 45267-0056"]


@pytest.mark.parametrize(
    ("dataset", "expected_optional_values"),
    (
        ("debug", 0),
        ("dev-19k", 0),
        ("dev-87k", 0),
        ("dev-205k", 63),
    ),
)
def test_visible_dataset_optional_name_counts_and_invariants(dataset: str, expected_optional_values: int):
    # setup
    ground_truth = json.loads((Path("data") / dataset / "ground_truth.json").read_text())
    people = [person for document in ground_truth.values() for person in document]

    # operate
    parsed_people = [GroundTruthPIIItem.from_serialized(person) for person in people]
    optional_values = [
        value
        for person in parsed_people
        for field in NAME_FIELDS
        for value in getattr(person, field)
        if value.optional
    ]

    # check
    assert len(optional_values) == expected_optional_values
    for person in parsed_people:
        email_local_parts = tuple(
            _normalized_alnum(email.partition("@")[0])
            for value in person.email
            for email in value.accepted_values
        )
        for field in NAME_FIELDS:
            for value in getattr(person, field):
                if value.optional:
                    assert person.email
                    assert any(
                        _normalized_alnum(accepted) in local_part
                        for accepted in value.accepted_values
                        for local_part in email_local_parts
                    )


def test_nested_visible_datasets_keep_shared_ground_truth_identical():
    ground_truth = {
        dataset: json.loads((Path("data") / dataset / "ground_truth.json").read_text())
        for dataset in ("dev-19k", "dev-87k", "dev-205k")
    }

    for document_id, people in ground_truth["dev-19k"].items():
        assert ground_truth["dev-87k"][document_id] == people
        assert ground_truth["dev-205k"][document_id] == people
    for document_id, people in ground_truth["dev-87k"].items():
        assert ground_truth["dev-205k"][document_id] == people


def _person(ground_truth: dict, *, document: str, first_name: str) -> dict:
    matches = [person for person in ground_truth[document] if person["first_name"] == [first_name]]
    assert len(matches) == 1
    return matches[0]


def _normalized_alnum(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())
