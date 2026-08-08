import csv
from pathlib import Path
from typing import Mapping

import pytest

from evaluator import EvaluationResult, Metrics, evaluate
from pii_item import PIIItem


CSV_FIELDS = ("document_id", "question", "answer", "dataset_class", "output_type")


def test_perfect_predictions_parse_document_labels_and_score_negative_documents(
    tmp_path: Path,
):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={
            "positive": """first_name: John
last_name: Doe
personal_email: john@example.com
work_email: john@company.com
telegram_alias: @john
address: Sofia, Bulgaria""",
            "negative": "",
        },
    )
    predictions = {
        "positive": [
            PIIItem(
                first_name=("John",),
                last_name=("Doe",),
                email=("john@example.com", "john@company.com"),
                social_network_identifier=("@john",),
                location=("Sofia, Bulgaria",),
            )
        ],
        "negative": [],
    }

    # operate
    result = evaluate(predictions, ground_truth_path=ground_truth_path)

    # check
    assert result.people == Metrics(true_positive=1)
    assert result.entities == Metrics(true_positive=6)
    assert result.documents == Metrics(true_positive=1, true_negative=1)
    assert result.documents.accuracy == pytest.approx(1.0)


def test_relaxed_person_matching_preserves_partial_entity_recall(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={
            "document": """first_name: Christine
last_name: Zelfman"""
        },
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
        rows={
            "document": """first_name: Christine
last_name: Zelfman"""
        },
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
            "document": """first_name: Christine
last_name: Zelfman
;first_name: Chris"""
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
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={
            "document": """first_name: John
last_name: Doe"""
        },
    )
    person = PIIItem(first_name=("John",), last_name=("Doe",))

    # operate
    result = evaluate({"document": [person, person]}, ground_truth_path=ground_truth_path)

    # check
    assert result.people == Metrics(true_positive=1, false_positive=1)
    assert result.people.precision == pytest.approx(0.5)
    assert result.entities == Metrics(true_positive=2, false_positive=2)


def test_entity_values_match_fuzzily_and_one_to_one(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(
        tmp_path,
        rows={
            "document": """first_name: Michael
last_name: Dourson
phone: 513-558-7949"""
        },
    )
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
        rows={"document": "first_name: John"},
    )

    # operate
    result = evaluate({}, ground_truth_path=ground_truth_path)

    # check
    assert result.people == Metrics(false_negative=1)
    assert result.documents == Metrics(false_negative=1)
    assert result.people.f1 == 0.0


def test_unknown_prediction_document_fails_fast(tmp_path: Path):
    ground_truth_path = _write_ground_truth(tmp_path, rows={"known": ""})

    with pytest.raises(ValueError, match="unknown document IDs: \\['unknown'\\]"):
        evaluate({"unknown": []}, ground_truth_path=ground_truth_path)


def test_prediction_on_negative_document_counts_as_false_positive(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(tmp_path, rows={"negative": ""})

    # operate
    result = evaluate(
        {"negative": [PIIItem(first_name=("Hallucinated",))]},
        ground_truth_path=ground_truth_path,
    )

    # check
    assert result.people == Metrics(false_positive=1)
    assert result.entities == Metrics(false_positive=1)
    assert result.documents == Metrics(false_positive=1)
    assert result.documents.accuracy == 0.0


@pytest.mark.parametrize("answer", ["nickname: Johnny", "first_name=John"])
def test_invalid_ground_truth_fails_fast(tmp_path: Path, answer: str):
    ground_truth_path = _write_ground_truth(tmp_path, rows={"document": answer})

    with pytest.raises(ValueError):
        evaluate({}, ground_truth_path=ground_truth_path)


def test_empty_dataset_metrics_do_not_divide_by_zero(tmp_path: Path):
    # setup
    ground_truth_path = _write_ground_truth(tmp_path, rows={})

    # operate
    result = evaluate({}, ground_truth_path=ground_truth_path)

    # check
    assert result == EvaluationResult(people=Metrics(), entities=Metrics(), documents=Metrics())
    assert result.people.precision == 0.0
    assert result.people.recall == 0.0
    assert result.people.f1 == 0.0
    assert result.documents.accuracy == 0.0


def _write_ground_truth(tmp_path: Path, *, rows: Mapping[str, str]) -> Path:
    path = tmp_path / "ground_truth.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for document_id, answer in rows.items():
            writer.writerow(
                {
                    "document_id": document_id,
                    "question": "Extract all PII entities",
                    "answer": answer,
                    "dataset_class": "test",
                    "output_type": "ListType",
                }
            )
    return path
