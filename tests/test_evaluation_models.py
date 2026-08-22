import pytest

from src.evaluation.models import GroundTruthPIIItem


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

    with pytest.raises(ValueError, match="unknown ground-truth value fields:.*source"):
        GroundTruthPIIItem.from_serialized(serialized)


@pytest.mark.parametrize("serialized", ["Shannon", {"canonical": "Shannon"}, None])
def test_ground_truth_person_rejects_non_list_field_values(serialized: object):
    with pytest.raises(TypeError, match="field 'first_name' must be a list"):
        GroundTruthPIIItem.from_serialized({"first_name": serialized})
