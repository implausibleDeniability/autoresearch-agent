import json
import stat

import pytest

from src.evaluation.diagnostics import preflight_diagnostics_path, write_diagnostics
from src.evaluation.models import PIIItem
from src.evaluation.trace import build_evaluation_trace


def test_diagnostics_preserve_raw_values_matches_errors_and_occurrences(tmp_path):
    # setup
    predictions = {
        "doc": [
            PIIItem(first_name=("Christine",), phone=("123", "wrong"), location=("Δelta",)),
            PIIItem(last_name=("Black",), location=("",)),
        ]
    }
    ground_truth = {
        "doc": [
            PIIItem(first_name=("Chris",), phone=("123", "missing"), location=("Δelta",)),
            PIIItem(last_name=("White",)),
        ]
    }
    text = "Christine called 123. Christine met Δelta."
    trace = build_evaluation_trace(predictions, ground_truth=ground_truth)
    path = tmp_path / "diagnostics.json"

    # operate
    write_diagnostics(path, trace=trace, texts={"doc": text}, dataset="debug")
    first_serialization = path.read_bytes()
    write_diagnostics(path, trace=trace, texts={"doc": text}, dataset="debug")

    # check
    result = json.loads(path.read_text())
    assert result["schema_version"] == 1
    assert result["metrics"]["true_positive"] == 3
    assert result["metrics"]["false_positive"] == 3
    assert result["metrics"]["false_negative"] == 2
    document = result["documents"][0]
    assert document["predictions"][0]["first_name"] == ["Christine"]
    assert document["ground_truth"][0]["first_name"] == ["Chris"]
    assert document["person_matches"] == [{"prediction_index": 0, "ground_truth_index": 0}]
    assert document["unmatched_prediction_indexes"] == [1]
    assert document["unmatched_ground_truth_indexes"] == [1]
    first_name = next(field for field in document["field_results"] if field["field"] == "first_name")
    assert first_name["matches"][0]["prediction"]["occurrences"] == [
        {"start": 0, "end": 9, "context_start": 0, "context_end": 42, "context": text},
        {"start": 22, "end": 31, "context_start": 0, "context_end": 42, "context": text},
    ]
    location = next(field for field in document["field_results"] if field["field"] == "location")
    assert location["matches"][0]["prediction"]["occurrences"][0]["start"] == 36
    assert location["false_positives"][0]["occurrences"] == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes() == first_serialization


def test_trace_field_ledger_totals_equal_document_and_aggregate_metrics():
    predictions = {"doc": [PIIItem(first_name=("John",), email=("john@example.com", "extra"))]}
    ground_truth = {"doc": [PIIItem(first_name=("John",), email=("john@example.com",))]}

    trace = build_evaluation_trace(predictions, ground_truth=ground_truth)

    field_total = sum((field.metrics for field in trace.documents[0].fields), start=trace.metrics.__class__())
    assert field_total == trace.documents[0].metrics == trace.metrics


def test_diagnostics_preflight_rejects_missing_parent(tmp_path):
    path = tmp_path / "missing" / "diagnostics.json"

    with pytest.raises(ValueError, match=f"parent directory does not exist: {path.parent}"):
        preflight_diagnostics_path(path)


def test_diagnostics_cap_common_value_occurrences_without_losing_count(tmp_path):
    predictions = {"doc": [PIIItem(location=("a",))]}
    trace = build_evaluation_trace(predictions, ground_truth={"doc": []})
    path = tmp_path / "diagnostics.json"

    write_diagnostics(path, trace=trace, texts={"doc": "a" * 25}, dataset="debug")

    document = json.loads(path.read_text())["documents"][0]
    location = next(field for field in document["field_results"] if field["field"] == "location")
    value = location["false_positives"][0]
    assert value["occurrence_count"] == 25
    assert value["occurrences_truncated"] is True
    assert len(value["occurrences"]) == 20
