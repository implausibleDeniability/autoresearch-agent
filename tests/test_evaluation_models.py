import pytest

from src.evaluation.models import GroundTruthPIIItem, GroundTruthValue


def test_ground_truth_person_accepts_legacy_strings_and_value_variants():
    # setup
    serialized = {
        "first_name": ["Shannon"],
        "email": [
            {
                "canonical": "kenny.shannon@epa.gov",
                "variants": ["kenny.shannon@8pa.gov"],
            }
        ],
    }

    # operate
    person = GroundTruthPIIItem.from_serialized(serialized)

    # check
    assert person.first_name[0].accepted_values == ("Shannon",)
    assert person.email[0].accepted_values == (
        "kenny.shannon@epa.gov",
        "kenny.shannon@8pa.gov",
    )
    assert person.last_name == ()


def test_optional_ground_truth_value_round_trips_with_and_without_variants():
    # setup
    first_name = {"canonical": "John", "optional": True}
    last_name = {
        "canonical": "Doe",
        "variants": ["D0e"],
        "optional": True,
    }

    # operate
    person = GroundTruthPIIItem.from_serialized(
        {
            "first_name": [first_name],
            "last_name": [last_name],
            "email": ["john.doe@example.com"],
        }
    )

    # check
    assert person.first_name[0].optional
    assert person.last_name[0].accepted_values == ("Doe", "D0e")
    assert person.serialize()["first_name"] == [first_name]
    assert person.serialize()["last_name"] == [last_name]


def test_explicit_false_optional_marker_serializes_as_legacy_value():
    value = GroundTruthValue.from_serialized({"canonical": "John", "optional": False})

    assert not value.optional
    assert value.serialize() == "John"


@pytest.mark.parametrize("optional", [None, 0, 1, "true"])
def test_ground_truth_value_requires_boolean_optional_marker(optional: object):
    with pytest.raises(TypeError, match=r"first_name\[0\]\.optional must be a boolean"):
        GroundTruthPIIItem.from_serialized(
            {
                "first_name": [{"canonical": "John", "optional": optional}],
                "email": ["john@example.com"],
            }
        )


def test_optional_ground_truth_value_is_limited_to_name_fields():
    with pytest.raises(ValueError, match="not allowed in fields:.*phone"):
        GroundTruthPIIItem.from_serialized(
            {
                "phone": [{"canonical": "555-0100", "optional": True}],
                "email": ["john@example.com"],
            }
        )


def test_optional_ground_truth_name_requires_email_anchor():
    with pytest.raises(ValueError, match="optional ground-truth names require an email"):
        GroundTruthPIIItem.from_serialized({"first_name": [{"canonical": "John", "optional": True}]})


def test_ground_truth_person_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown ground-truth person fields:.*nickname"):
        GroundTruthPIIItem.from_serialized({"first_name": ["Shannon"], "nickname": ["Kenny"]})


def test_ground_truth_value_rejects_unknown_fields():
    serialized = {
        "email": [
            {
                "canonical": "kenny.shannon@epa.gov",
                "variants": ["kenny.shannon@8pa.gov"],
                "source": "OCR",
            }
        ]
    }

    with pytest.raises(ValueError, match=r"unknown ground-truth person\.email\[0\] fields:.*source"):
        GroundTruthPIIItem.from_serialized(serialized)


@pytest.mark.parametrize("serialized", ["Shannon", {"canonical": "Shannon"}, None])
def test_ground_truth_person_rejects_non_list_field_values(serialized: object):
    with pytest.raises(TypeError, match="ground-truth person.first_name must be a list"):
        GroundTruthPIIItem.from_serialized({"first_name": serialized})
