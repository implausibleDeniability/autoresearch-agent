import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from evaluation import EvaluationResult, Metrics, evaluate
from evaluator import Dataset, _parse_args
from pii_item import PIIItem
from solution import _has_candidate_content


def test_cli_requires_dataset():
    with pytest.raises(SystemExit):
        _parse_args(())


@pytest.mark.parametrize("dataset", [Dataset.DEV_5K, Dataset.DEV_50K])
def test_cli_accepts_dev_dataset(dataset: str):
    assert _parse_args(("--dataset", dataset)).dataset == dataset


def test_cli_accepts_debug_dataset():
    assert _parse_args(("--dataset", Dataset.DEBUG)).dataset == Dataset.DEBUG


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
    assert result.people == Metrics(true_positive=1)
    assert result.entities == Metrics(true_positive=6)
    assert result.document_accuracy == pytest.approx(1.0)


def test_relaxed_person_matching_preserves_partial_entity_recall(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={"document": [PIIItem(first_name=("Christine",), last_name=("Zelfman",))]},
    )
    predictions = {"document": [PIIItem(first_name=("Chris",))]}

    # operate
    result = evaluate(predictions, ground_truth_path=ground_truth_path)

    # check
    assert result.people == Metrics(true_positive=1)
    assert result.entities == Metrics(true_positive=1, false_negative=1)
    assert result.entities.recall == pytest.approx(0.5)


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
    assert result.people == Metrics(false_positive=1, false_negative=1)
    assert result.entities == Metrics(false_positive=2, false_negative=2)


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
    assert result.people == Metrics(true_positive=2)
    assert result.entities == Metrics(true_positive=3)


def test_duplicate_prediction_reduces_precision(tmp_path: Path):
    # setup
    person = PIIItem(first_name=("John",), last_name=("Doe",))
    ground_truth_path = _write_ground_truth(tmp_path, rows={"document": [person]})

    # operate
    result = evaluate({"document": [person, person]}, ground_truth_path=ground_truth_path)

    # check
    assert result.people == Metrics(true_positive=1, false_positive=1)
    assert result.people.precision == pytest.approx(0.5)
    assert result.entities == Metrics(true_positive=2, false_positive=2)


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
    assert result.entities == Metrics(true_positive=3, false_positive=1)


def test_missing_prediction_document_counts_as_false_negative(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={"document": [PIIItem(first_name=("John",))]},
    )

    # operate
    result = evaluate({}, ground_truth_path=ground_truth_path)

    # check
    assert result.people == Metrics(false_negative=1)
    assert result.people.f1 == 0.0
    assert result.document_accuracy == 0.0


def test_unknown_prediction_document_fails_fast(tmp_path: Path):
    ground_truth_path = _write_ground_truth(tmp_path, rows={"known": []})

    with pytest.raises(ValueError, match="unknown document IDs: \\['unknown'\\]"):
        evaluate({"unknown": []}, ground_truth_path=ground_truth_path)


def test_prediction_on_negative_document_reduces_accuracy(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(tmp_path, rows={"negative": []})

    # operate
    result = evaluate(
        {"negative": [PIIItem(first_name=("Hallucinated",))]},
        ground_truth_path=ground_truth_path,
    )

    # check
    assert result.people == Metrics(false_positive=1)
    assert result.entities == Metrics(false_positive=1)
    assert result.document_accuracy == 0.0


def test_empty_dataset_metrics_do_not_divide_by_zero(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(tmp_path, rows={})

    # operate
    result = evaluate({}, ground_truth_path=ground_truth_path)

    # check
    assert result == EvaluationResult(people=Metrics(), entities=Metrics(), document_accuracy=0.0)
    assert result.people.precision == 0.0
    assert result.people.recall == 0.0
    assert result.people.f1 == 0.0


def _write_ground_truth(
    tmp_path: Path,
    *,
    rows: Mapping[str, Sequence[PIIItem]],
) -> Path:
    path = tmp_path / "ground_truth.json"
    serialized = {document_id: [asdict(person) for person in people] for document_id, people in rows.items()}
    path.write_text(json.dumps(serialized))
    return path
