import json
import stat

import pytest

from src.evaluation.diagnostics import preflight_diagnostics_path, write_diagnostics
from src.evaluation.models import GroundTruthPIIItem, GroundTruthValue, PIIItem
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
    assert result["schema_version"] == 8
    assert list(result) == [
        "schema_version",
        "source_matching_policy",
        "dataset",
        "dataset_document_count",
        "document_count",
        "metrics",
        "field_metrics",
        "documents",
    ]
    assert result["source_matching_policy"]["similarity_threshold"] == 0.65
    assert result["source_matching_policy"]["fuzzy_work_budget"] == 50_000_000
    assert result["source_matching_policy"]["candidate_enumeration_budget"] == 200_000
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
    assert first_name["matches"][0]["prediction"]["source_evidence"] == [
        {
            "start": 0,
            "end": 9,
            "match_kind": "raw_exact",
            "similarity": 1.0,
            "context_start": 0,
            "context_end": 42,
            "context": text,
        },
        {
            "start": 22,
            "end": 31,
            "match_kind": "raw_exact",
            "similarity": 1.0,
            "context_start": 0,
            "context_end": 42,
            "context": text,
        },
    ]
    location = next(field for field in document["field_results"] if field["field"] == "location")
    assert location["matches"][0]["prediction"]["source_evidence"][0]["start"] == 36
    assert location["false_positives"][0]["source_evidence"] == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes() == first_serialization


def test_diagnostics_serialize_optional_values_and_neutral_ledger(tmp_path):
    # setup
    prediction = PIIItem(first_name=("John",), email=("john.doe@example.com",))
    ground_truth = GroundTruthPIIItem(
        first_name=(GroundTruthValue(canonical="John", optional=True),),
        last_name=(GroundTruthValue(canonical="Doe", optional=True),),
        email=(GroundTruthValue(canonical="john.doe@example.com"),),
    )
    trace = build_evaluation_trace({"doc": [prediction]}, ground_truth={"doc": [ground_truth]})
    path = tmp_path / "diagnostics.json"

    # operate
    write_diagnostics(
        path,
        trace=trace,
        texts={"doc": "john.doe@example.com"},
        dataset="debug",
    )

    # check
    document = json.loads(path.read_text())["documents"][0]
    first_name = next(field for field in document["field_results"] if field["field"] == "first_name")
    last_name = next(field for field in document["field_results"] if field["field"] == "last_name")
    assert document["ground_truth"][0]["first_name"][0]["optional"] is True
    assert first_name["ignored_optional_matches"][0]["ground_truth"]["optional"] is True
    assert first_name["matches"] == []
    assert last_name["unmatched_optional_values"][0]["optional"] is True
    assert last_name["false_negatives"] == []


def test_diagnostics_preserve_ground_truth_variants_and_source_form(tmp_path):
    # setup
    prediction = PIIItem(first_name=("Shannon",), email=("kenny.shannon@8pa.gov",))
    ground_truth = GroundTruthPIIItem(
        first_name=(GroundTruthValue("Shannon"),),
        email=(
            GroundTruthValue(
                canonical="kenny.shannon@epa.gov",
                variants=("kenny.shannon@8pa.gov",),
            ),
        ),
    )
    trace = build_evaluation_trace({"doc": [prediction]}, ground_truth={"doc": [ground_truth]})
    path = tmp_path / "diagnostics.json"

    # operate
    write_diagnostics(
        path,
        trace=trace,
        texts={"doc": "Kenny.Shannon@8pa.gov"},
        dataset="debug",
    )

    # check
    document = json.loads(path.read_text())["documents"][0]
    email = next(field for field in document["field_results"] if field["field"] == "email")
    assert document["ground_truth"][0]["email"][0]["variants"] == ["kenny.shannon@8pa.gov"]
    assert email["matches"][0]["ground_truth"]["source_value"] == "kenny.shannon@8pa.gov"


def test_diagnostics_use_none_when_no_accepted_source_form_matches(tmp_path):
    # setup
    ground_truth = GroundTruthPIIItem(
        first_name=(GroundTruthValue(canonical="Absent", variants=("Missing",)),),
    )
    trace = build_evaluation_trace({"doc": []}, ground_truth={"doc": [ground_truth]})
    path = tmp_path / "diagnostics.json"

    # operate
    write_diagnostics(path, trace=trace, texts={"doc": "unrelated"}, dataset="debug")

    # check
    first_name = json.loads(path.read_text())["documents"][0]["field_results"][0]
    assert first_name["false_negatives"][0]["source_value"] is None


def test_diagnostics_aggregate_fuzzy_search_completeness_across_variants(tmp_path):
    # setup
    ground_truth = GroundTruthPIIItem(
        first_name=(GroundTruthValue(canonical="yyy", variants=("y" * 50,)),),
    )
    trace = build_evaluation_trace({"doc": []}, ground_truth={"doc": [ground_truth]})
    path = tmp_path / "diagnostics.json"

    # operate
    write_diagnostics(path, trace=trace, texts={"doc": "x " * 1_000}, dataset="debug")

    # check
    first_name = json.loads(path.read_text())["documents"][0]["field_results"][0]
    assert first_name["false_negatives"][0]["fuzzy_search_complete"] is False


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
    assert value["source_evidence_count"] == 25
    assert value["raw_occurrence_count"] == 25
    assert value["source_evidence_truncated"] is True
    assert len(value["source_evidence"]) == 20


def test_diagnostics_report_case_normalized_and_fuzzy_source_evidence(tmp_path):
    predictions = {"doc": [PIIItem(email=("rvinas@gmaonline.org",), phone=("513-558-7949",))]}
    trace = build_evaluation_trace(predictions, ground_truth={"doc": []})
    path = tmp_path / "diagnostics.json"

    write_diagnostics(
        path,
        trace=trace,
        texts={"doc": "RVINAS@GMAONLINE.ORG called 513 558 7949"},
        dataset="debug",
    )

    fields = json.loads(path.read_text())["documents"][0]["field_results"]
    email = next(field for field in fields if field["field"] == "email")["false_positives"][0]
    phone = next(field for field in fields if field["field"] == "phone")["false_positives"][0]
    assert email["normalized_occurrence_count"] == 1
    assert email["fuzzy_search_complete"] is True
    assert email["source_evidence"][0]["match_kind"] == "normalized_exact"
    assert phone["fuzzy_occurrence_count"] == 1
    assert phone["source_evidence"][0]["match_kind"] == "fuzzy"


def test_diagnostics_preserve_prediction_and_ground_truth_comparison_direction(tmp_path):
    predictions = {"prediction": [PIIItem(first_name=("aba",))]}
    ground_truth = {
        "prediction": [],
        "ground_truth": [PIIItem(first_name=("aba",))],
    }
    trace = build_evaluation_trace(predictions, ground_truth=ground_truth)
    path = tmp_path / "diagnostics.json"

    write_diagnostics(
        path,
        trace=trace,
        texts={"prediction": "bca", "ground_truth": "bca"},
        dataset="debug",
    )

    documents = {item["document_id"]: item for item in json.loads(path.read_text())["documents"]}
    prediction_field = documents["prediction"]["field_results"][0]
    ground_truth_field = documents["ground_truth"]["field_results"][0]
    assert prediction_field["false_positives"][0]["source_evidence_count"] == 0
    assert ground_truth_field["false_negatives"][0]["fuzzy_occurrence_count"] == 1
