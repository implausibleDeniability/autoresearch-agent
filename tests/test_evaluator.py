import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from solution import _has_candidate_content
from src.evaluation.cli import Dataset, _parse_arguments
from src.evaluation.matching import (
    MATCH,
    _people_match_approximately,
    _people_match_exactly,
    compare_values,
    match_people,
)
from src.evaluation.metrics import evaluate, evaluate_trace
from src.evaluation.models import GroundTruthPIIItem, GroundTruthValue, PIIItem


def test_cli_requires_dataset():
    with pytest.raises(SystemExit):
        _parse_arguments(())


@pytest.mark.parametrize("dataset", [Dataset.DEV_19K, Dataset.DEV_87K, Dataset.DEV_202K])
def test_cli_accepts_dev_dataset(dataset: str):
    assert _parse_arguments(("--dataset", dataset)).dataset == dataset


def test_cli_accepts_debug_dataset():
    assert _parse_arguments(("--dataset", Dataset.DEBUG)).dataset == Dataset.DEBUG


@pytest.mark.parametrize("dataset", ["dev-5k", "dev-10k", "dev-50k"])
def test_cli_rejects_legacy_dev_dataset(dataset: str):
    with pytest.raises(SystemExit):
        _parse_arguments(("--dataset", dataset))


def test_candidate_detection_skips_punctuation_only_chunks():
    assert not _has_candidate_content(".......")
    assert _has_candidate_content("John Doe")


def test_perfect_predictions_score_canonical_labels_and_negative_documents(tmp_path: Path):
    # setup
    positive = PIIItem(
        first_name=("John",),
        last_name=("Doe",),
        email=("john@example.com", "john@company.com"),
        social_network_identifier=("@john",),
        location=("Sofia, Bulgaria",),
    )
    ground_truth_path = _write_ground_truth(tmp_path, rows={"positive": [positive], "negative": []})

    # operate
    result = evaluate({"positive": [positive], "negative": []}, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == 1.0


@pytest.mark.parametrize("email", ["kenny.shannon@epa.gov", "kenny.shannon@8pa.gov"])
def test_ground_truth_variants_score_as_one_value(tmp_path: Path, email: str):
    # setup
    person = asdict(PIIItem(first_name=("Shannon",), last_name=("Kenny",)))
    person["email"] = [
        {
            "canonical": "kenny.shannon@epa.gov",
            "variants": ["kenny.shannon@8pa.gov"],
        }
    ]
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(json.dumps({"document": [person]}))
    prediction = PIIItem(first_name=("Shannon",), last_name=("Kenny",), email=(email,))

    # operate
    result = evaluate({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    assert result.true_positive == 3
    assert result.f_score == 1.0


def test_ground_truth_variants_cannot_match_twice(tmp_path: Path):
    # setup
    person = asdict(PIIItem(first_name=("Shannon",), last_name=("Kenny",)))
    person["email"] = [
        {
            "canonical": "kenny.shannon@epa.gov",
            "variants": ["kenny.shannon@8pa.gov"],
        }
    ]
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(json.dumps({"document": [person]}))
    prediction = PIIItem(
        first_name=("Shannon",),
        last_name=("Kenny",),
        email=("kenny.shannon@epa.gov", "kenny.shannon@8pa.gov"),
    )

    # operate
    result = evaluate({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    assert (result.true_positive, result.false_positive, result.false_negative) == (3, 1, 0)


def test_missing_optional_name_is_not_a_false_negative(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(tmp_path)
    prediction = PIIItem(email=("john.doe@example.com",))

    # operate
    result = evaluate({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    assert (result.true_positive, result.false_positive, result.false_negative) == (1, 0, 0)


def test_exact_optional_name_is_ignored_instead_of_scored(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(tmp_path)
    prediction = PIIItem(first_name=("John",), email=("john.doe@example.com",))

    # operate
    trace = evaluate_trace({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    first_name = next(field for field in trace.documents[0].fields if field.field == "first_name")
    assert (trace.metrics.true_positive, trace.metrics.false_positive, trace.metrics.false_negative) == (
        1,
        0,
        0,
    )
    assert len(first_name.ignored_optional_matches) == 1
    assert first_name.unmatched_optional_values == ()


def test_optional_variant_is_ignored(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(
        tmp_path,
        first_name={"canonical": "John", "variants": ["J0hn"], "optional": True},
    )
    prediction = PIIItem(first_name=("J0hn",), email=("john.doe@example.com",))

    # operate
    result = evaluate({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    assert (result.true_positive, result.false_positive, result.false_negative) == (1, 0, 0)


def test_fuzzy_optional_name_remains_a_false_positive(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(
        tmp_path,
        first_name={"canonical": "Joe", "optional": True},
    )
    prediction = PIIItem(first_name=("Jon",), email=("john.doe@example.com",))

    # operate
    trace = evaluate_trace({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    first_name = next(field for field in trace.documents[0].fields if field.field == "first_name")
    assert (trace.metrics.true_positive, trace.metrics.false_positive, trace.metrics.false_negative) == (
        1,
        1,
        0,
    )
    assert [value.value for value in first_name.false_positives] == ["Jon"]
    assert [value.value for value in first_name.unmatched_optional_values] == ["Joe"]


def test_required_name_matches_before_colliding_optional_name(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(
        tmp_path,
        first_name=["John", {"canonical": "John", "optional": True}],
    )
    prediction = PIIItem(first_name=("John",), email=("john.doe@example.com",))

    # operate
    trace = evaluate_trace({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    first_name = next(field for field in trace.documents[0].fields if field.field == "first_name")
    assert len(first_name.matches) == 1
    assert first_name.ignored_optional_matches == ()
    assert len(first_name.unmatched_optional_values) == 1


def test_ambiguous_required_names_cannot_be_absorbed_by_optional_name(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(
        tmp_path,
        first_name=["John", "Jonn", {"canonical": "Jon", "optional": True}],
    )
    prediction = PIIItem(first_name=("Jon",), email=("john.doe@example.com",))

    # operate
    trace = evaluate_trace({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    first_name = next(field for field in trace.documents[0].fields if field.field == "first_name")
    assert first_name.matches == ()
    assert first_name.ignored_optional_matches == ()
    assert [value.value for value in first_name.false_positives] == ["Jon"]
    assert [value.value for value in first_name.false_negatives] == ["John", "Jonn"]


@pytest.mark.parametrize("optional_name", [(), ("John",), ("Wrong",)])
def test_optional_name_does_not_change_email_anchored_person_pairing(
    tmp_path: Path, optional_name: tuple[str, ...]
):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(tmp_path)
    prediction = PIIItem(first_name=optional_name, email=("john.doe@example.com",))

    # operate
    trace = evaluate_trace({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    assert trace.documents[0].person_matches == ((0, 0),)


def test_optional_name_on_separate_person_remains_false_positive(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(tmp_path)
    predictions = [
        PIIItem(email=("john.doe@example.com",)),
        PIIItem(first_name=("John",)),
    ]

    # operate
    trace = evaluate_trace({"document": predictions}, ground_truth_path=ground_truth_path)

    # check
    assert trace.documents[0].person_matches == ((0, 0),)
    assert (trace.metrics.true_positive, trace.metrics.false_positive, trace.metrics.false_negative) == (
        1,
        1,
        0,
    )


def test_optional_only_person_matching_distinguishes_absent_and_optional_fields():
    email = (GroundTruthValue(canonical="john.doe@example.com"),)
    optional_name = (GroundTruthValue(canonical="John", optional=True),)
    prediction = PIIItem(first_name=("Predicted",), email=("john.doe@example.com",))

    assert _people_match_exactly(
        prediction,
        GroundTruthPIIItem(first_name=optional_name, email=email),
    )
    assert _people_match_exactly(
        PIIItem(email=("john.doe@example.com",)),
        GroundTruthPIIItem(first_name=optional_name, email=email),
    )
    assert not _people_match_exactly(prediction, GroundTruthPIIItem(email=email))


def test_optional_only_person_matching_keeps_required_values_as_exact_anchors():
    email = (GroundTruthValue(canonical="john.doe@example.com"),)
    names = (
        GroundTruthValue(canonical="John"),
        GroundTruthValue(canonical="Jack"),
        GroundTruthValue(canonical="Johnny", optional=True),
    )
    ground_truth = GroundTruthPIIItem(first_name=names, email=email)

    assert _people_match_exactly(
        PIIItem(first_name=("John",), email=("john.doe@example.com",)),
        ground_truth,
    )
    assert _people_match_exactly(
        PIIItem(first_name=("Jack",), email=("john.doe@example.com",)),
        ground_truth,
    )
    assert not _people_match_exactly(
        PIIItem(first_name=("Johnny",), email=("john.doe@example.com",)),
        ground_truth,
    )
    assert not _people_match_exactly(
        PIIItem(first_name=("Jane",), email=("john.doe@example.com",)),
        ground_truth,
    )


def test_optional_only_person_matching_leaves_approximate_fallback_unchanged():
    email = (GroundTruthValue(canonical="john.doe@example.com"),)
    optional_name = (GroundTruthValue(canonical="John", optional=True),)
    fuzzy_prediction = PIIItem(first_name=("Predicted",), email=("john.doe@example.co",))

    assert _people_match_approximately(
        fuzzy_prediction,
        GroundTruthPIIItem(first_name=optional_name, email=email),
    )
    assert _people_match_approximately(fuzzy_prediction, GroundTruthPIIItem(email=email))
    assert not _people_match_approximately(
        PIIItem(first_name=("Jane",), email=("john.doe@example.co",)),
        GroundTruthPIIItem(
            first_name=(GroundTruthValue(canonical="John"),),
            email=email,
        ),
    )


@pytest.mark.parametrize(
    ("reverse_predictions", "reverse_ground_truth"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_optional_only_person_matching_reserves_exact_email_before_fuzzy_candidates(
    reverse_predictions: bool,
    reverse_ground_truth: bool,
):
    emails = ("alex.smith@example.com", "alex.smyth@example.com")
    predictions = [
        PIIItem(first_name=(("Alex", "Alec")[index],), email=(email,)) for index, email in enumerate(emails)
    ]
    ground_truth = [
        GroundTruthPIIItem(
            first_name=(GroundTruthValue(canonical=("Alex", "Alec")[index], optional=True),),
            email=(GroundTruthValue(canonical=email),),
        )
        for index, email in enumerate(emails)
    ]
    if reverse_predictions:
        predictions.reverse()
    if reverse_ground_truth:
        ground_truth.reverse()

    comparison = compare_values(emails[0], ground_truth=emails[1])
    matches = match_people(predictions, ground_truth=ground_truth)

    assert comparison.result == MATCH
    assert not comparison.normalized_exact
    assert {
        predictions[prediction_index].email[0]: ground_truth[ground_index].email[0].canonical
        for prediction_index, ground_index in matches.items()
    } == {email: email for email in emails}


def test_optional_only_person_matching_keeps_duplicate_exact_anchors_ambiguous():
    predictions = [PIIItem(first_name=(name,), email=("shared@example.com",)) for name in ("Alice", "Bob")]
    ground_truth = [
        GroundTruthPIIItem(
            first_name=(GroundTruthValue(canonical=name, optional=True),),
            email=(GroundTruthValue(canonical="shared@example.com"),),
        )
        for name in ("Alice", "Bob")
    ]

    assert match_people(predictions, ground_truth=ground_truth) == {}


@pytest.mark.parametrize(
    ("document_id", "ground_indexes", "collision_indexes"),
    [
        ("jfkf0256", (2, 4, 5, 6, 7), (2, 4, 5, 7)),
        ("zldc0256", (0, 2), (0, 2)),
    ],
)
def test_dev_202k_optional_only_people_pair_by_exact_email(
    document_id: str,
    ground_indexes: tuple[int, ...],
    collision_indexes: tuple[int, ...],
):
    path = Path(__file__).parents[1] / "data" / "dev-202k" / "ground_truth.json"
    serialized = json.loads(path.read_text())[document_id]
    ground_truth = tuple(
        GroundTruthPIIItem.from_serialized(person, context=document_id) for person in serialized
    )
    predictions = [
        PIIItem(
            first_name=tuple(value.canonical for value in ground_truth[index].first_name),
            last_name=tuple(value.canonical for value in ground_truth[index].last_name),
            email=(ground_truth[index].email[0].canonical,),
        )
        for index in ground_indexes
    ]

    for index in ground_indexes:
        person = ground_truth[index]
        assert person.first_name and person.last_name
        assert all(value.optional for value in person.first_name + person.last_name)

    for index in collision_indexes:
        person = ground_truth[index]
        competitors = (
            value.canonical
            for competitor_index, competitor in enumerate(ground_truth)
            if competitor_index != index
            for value in competitor.email
        )
        assert any(
            comparison.result == MATCH and not comparison.normalized_exact
            for comparison in (
                compare_values(person.email[0].canonical, ground_truth=competitor)
                for competitor in competitors
            )
        )

    assert match_people(predictions, ground_truth=ground_truth) == {
        prediction_index: ground_index for prediction_index, ground_index in enumerate(ground_indexes)
    }


def test_unmatched_person_emits_false_negatives_for_required_values_only(tmp_path: Path):
    # setup
    ground_truth_path = _write_optional_name_ground_truth(tmp_path)

    # operate
    trace = evaluate_trace({}, ground_truth_path=ground_truth_path)

    # check
    first_name = next(field for field in trace.documents[0].fields if field.field == "first_name")
    assert (trace.metrics.true_positive, trace.metrics.false_positive, trace.metrics.false_negative) == (
        0,
        0,
        1,
    )
    assert first_name.false_negatives == ()
    assert [value.value for value in first_name.unmatched_optional_values] == ["John"]


def test_field_ledgers_partition_every_prediction_and_ground_truth_value(tmp_path: Path):
    # setup
    ground_truth = GroundTruthPIIItem(
        first_name=(GroundTruthValue(canonical="John"),),
        last_name=(GroundTruthValue(canonical="Doe", optional=True),),
        email=(GroundTruthValue(canonical="john.doe@example.com"),),
    )
    predictions = [
        PIIItem(
            first_name=("John",),
            last_name=("Doe",),
            phone=("555-0100",),
            email=("john.doe@example.com",),
        ),
        PIIItem(first_name=("Hallucinated",)),
    ]
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps({"document": [ground_truth.serialize()]}))

    # operate
    trace = evaluate_trace(
        {"document": predictions},
        ground_truth_path=path,
    )

    # check
    document = trace.documents[0]
    for field in document.fields:
        prediction_count = sum(len(getattr(person, field.field)) for person in document.predictions)
        ground_truth_count = sum(len(getattr(person, field.field)) for person in document.ground_truth)
        assert prediction_count == (
            len(field.matches) + len(field.ignored_optional_matches) + len(field.false_positives)
        )
        assert ground_truth_count == (
            len(field.matches)
            + len(field.ignored_optional_matches)
            + len(field.false_negatives)
            + len(field.unmatched_optional_values)
        )


def test_relaxed_person_matching_preserves_partial_pii_score(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={"document": [PIIItem(first_name=("Christine",), last_name=("Zelfman",))]},
    )
    predictions = {"document": [PIIItem(first_name=("Chris",))]}

    # operate
    result = evaluate(predictions, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == pytest.approx(6 / 11)


def test_conflicting_core_fields_prevent_person_match(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={"document": [PIIItem(first_name=("Christine",), last_name=("Zelfman",))]},
    )
    predictions = {"document": [PIIItem(first_name=("Christine",), last_name=("Black",))]}

    # operate
    result = evaluate(predictions, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == 0.0


def test_exact_matches_are_reserved_before_relaxed_matching(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={
            "document": [
                PIIItem(first_name=("Christine",), last_name=("Zelfman",)),
                PIIItem(first_name=("Chris",)),
            ]
        },
    )
    predictions = {
        "document": [
            PIIItem(first_name=("Chris",)),
            PIIItem(first_name=("Christine",), last_name=("Zelfman",)),
        ]
    }

    # operate
    result = evaluate(predictions, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == 1.0


def test_duplicate_prediction_reduces_f_score(tmp_path: Path):
    # setup
    person = PIIItem(first_name=("John",), last_name=("Doe",))
    ground_truth_path = _write_ground_truth(tmp_path, rows={"document": [person]})

    # operate
    result = evaluate({"document": [person, person]}, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == pytest.approx(6 / 7)


def test_f_score_weights_recall_five_times_more_than_precision(tmp_path: Path):
    # setup
    low_precision_ground = _write_ground_truth(
        tmp_path,
        rows={"document": [PIIItem(first_name=("John",))]},
    )
    low_precision = evaluate(
        {"document": [PIIItem(first_name=("John", "Wrong"))]},
        ground_truth_path=low_precision_ground,
    )
    low_recall_ground = _write_ground_truth(
        tmp_path,
        rows={"document": [PIIItem(first_name=("John",), last_name=("Doe",))]},
    )

    # operate
    low_recall = evaluate(
        {"document": [PIIItem(first_name=("John",))]},
        ground_truth_path=low_recall_ground,
    )

    # check
    assert low_precision.precision == pytest.approx(0.5)
    assert low_precision.recall == 1.0
    assert low_precision.f_score == pytest.approx(6 / 7)
    assert low_recall.precision == 1.0
    assert low_recall.recall == pytest.approx(0.5)
    assert low_recall.f_score == pytest.approx(6 / 11)


def test_entity_values_match_fuzzily_and_one_to_one(tmp_path: Path):
    # setup
    person = PIIItem(
        first_name=("Michael",),
        last_name=("Dourson",),
        phone=("513-558-7949",),
    )
    ground_truth_path = _write_ground_truth(tmp_path, rows={"document": [person]})
    prediction = PIIItem(
        first_name=("Michael",),
        last_name=("Dourson",),
        phone=("513 558 7949", "000-000-0000"),
    )

    # operate
    result = evaluate({"document": [prediction]}, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == pytest.approx(18 / 19)


def test_missing_prediction_document_counts_as_false_negative(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={"document": [PIIItem(first_name=("John",))]},
    )

    # operate
    result = evaluate({}, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == 0.0


def test_unknown_prediction_document_fails_fast(tmp_path: Path):
    ground_truth_path = _write_ground_truth(tmp_path, rows={"known": []})

    with pytest.raises(ValueError, match="unknown document IDs: \\['unknown'\\]"):
        evaluate({"unknown": []}, ground_truth_path=ground_truth_path)


def test_ground_truth_load_errors_include_json_path_context(tmp_path: Path):
    # setup
    path = tmp_path / "ground_truth.json"
    path.write_text(
        json.dumps(
            {
                "document": [
                    {
                        "first_name": [{"canonical": "John", "optional": "true"}],
                        "email": ["john@example.com"],
                    }
                ]
            }
        )
    )

    # operate/check
    with pytest.raises(
        TypeError,
        match=rf"{path}: document='document' person\[0\]\.first_name\[0\]\.optional",
    ):
        evaluate({}, ground_truth_path=path)


def test_prediction_on_negative_document_scores_zero(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(tmp_path, rows={"negative": []})

    # operate
    result = evaluate(
        {"negative": [PIIItem(first_name=("Hallucinated",))]},
        ground_truth_path=ground_truth_path,
    )

    # check
    assert result.f_score == 0.0


def test_empty_dataset_metrics_do_not_divide_by_zero(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(tmp_path, rows={})

    # operate
    result = evaluate({}, ground_truth_path=ground_truth_path)

    # check
    assert result.f_score == 0.0


def _write_ground_truth(
    tmp_path: Path,
    *,
    rows: Mapping[str, Sequence[PIIItem]],
) -> Path:
    path = tmp_path / "ground_truth.json"
    serialized = {document_id: [asdict(person) for person in people] for document_id, people in rows.items()}
    path.write_text(json.dumps(serialized))
    return path


def _write_optional_name_ground_truth(
    tmp_path: Path,
    *,
    first_name: object | None = None,
) -> Path:
    path = tmp_path / "ground_truth.json"
    if first_name is None:
        first_name = {"canonical": "John", "optional": True}
    values = first_name if isinstance(first_name, list) else [first_name]
    person = asdict(PIIItem())
    person["first_name"] = values
    person["email"] = ["john.doe@example.com"]
    path.write_text(json.dumps({"document": [person]}))
    return path
