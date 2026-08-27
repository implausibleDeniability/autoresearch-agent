import json
import os
import stat
import subprocess

import pytest

from src.evaluation.evidence import (
    canonical_fingerprint,
    EvidenceValidationError,
    discover_evidence_bank,
    load_evidence_files,
)
from src.evaluation.provenance import capture_evidence_context, runtime_fingerprint
from tests.comparison_fixtures import make_evidence, rewrite_fingerprint, write_bank


def test_complete_synthetic_bank_loads_and_reconciles(tmp_path):
    bank = write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2))

    records = load_evidence_files(discover_evidence_bank(bank, arm="incumbent"), arm="incumbent")

    assert [record.seed for record in records] == [0, 1, 2]
    assert len(records[0].documents) == 121
    assert len(records[0].receipts) == 121
    assert records[0].aggregate.true_positive == 1210


def test_evidence_fingerprint_rejects_tampering_without_echoing_payload(tmp_path):
    payload = make_evidence(arm="incumbent", seed=0)
    payload["run"]["dataset"] = "private-secret"
    path = tmp_path / "tampered.evidence.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(EvidenceValidationError) as caught:
        load_evidence_files((path,), arm="incumbent")

    serialized = json.dumps([issue.serialize() for issue in caught.value.issues])
    assert "evidence_fingerprint" in serialized
    assert "private-secret" not in serialized


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (lambda payload: payload["run"]["coverage"].update(completed=120), "$.run.coverage"),
        (
            lambda payload: payload["metrics"]["documents"][0]["counts"].update(true_positive=9),
            "$.metrics.documents[0].fields",
        ),
        (
            lambda payload: payload["metrics"]["aggregate"].update(false_positive=True),
            "$.metrics.aggregate.false_positive",
        ),
    ],
)
def test_evidence_rejects_partial_inconsistent_and_boolean_counts(tmp_path, mutate, expected_path):
    payload = make_evidence(arm="incumbent", seed=0)
    mutate(payload)
    rewrite_fingerprint(payload)
    path = tmp_path / "invalid.evidence.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(EvidenceValidationError) as caught:
        load_evidence_files((path,), arm="incumbent")

    assert caught.value.issues[0].json_path == expected_path


def test_evidence_rejects_non_finite_duplicate_json_and_symlinks(tmp_path):
    malformed_paths = []
    for name, serialized in (
        ("constant", '{"value":NaN}'),
        ("overflow", '{"value":1e999}'),
        ("duplicate", '{"value":1,"value":2}'),
    ):
        path = tmp_path / f"{name}.evidence.json"
        path.write_text(serialized)
        malformed_paths.append(path)
    target = tmp_path / "target.evidence.json"
    target.write_text("{}")
    symlink = tmp_path / "link.evidence.json"
    symlink.symlink_to(target)

    for path in (*malformed_paths, symlink):
        with pytest.raises(EvidenceValidationError) as caught:
            load_evidence_files((path,), arm="candidate")
        assert caught.value.issues[0].code == "unreadable_evidence"


def test_evidence_rejects_fifo_without_blocking(tmp_path):
    path = tmp_path / "stream.evidence.json"
    os.mkfifo(path)

    with pytest.raises(EvidenceValidationError) as caught:
        load_evidence_files((path,), arm="candidate")

    assert caught.value.issues[0].code == "unreadable_evidence"


def test_evidence_rejects_noncontiguous_request_ordinals(tmp_path):
    payload = make_evidence(arm="incumbent", seed=0)
    payload["requests"]["receipts"][0]["request_ordinal"] = 1
    receipts = payload["requests"]["receipts"]
    plan = [
        {
            "document_ordinal": receipt["document_ordinal"],
            "request_ordinal": receipt["request_ordinal"],
            "request_key": receipt["request_key"],
        }
        for receipt in receipts
    ]
    payload["requests"]["request_plan_fingerprint"] = canonical_fingerprint(plan)
    payload["requests"]["response_bank_fingerprint"] = canonical_fingerprint(receipts)
    rewrite_fingerprint(payload)
    path = tmp_path / "invalid-ordinals.evidence.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(EvidenceValidationError) as caught:
        load_evidence_files((path,), arm="incumbent")

    assert caught.value.issues[0].json_path == "$.requests.receipts"


def test_evidence_rejects_document_counts_that_disagree_with_document_fields(tmp_path):
    payload = make_evidence(arm="incumbent", seed=0)
    payload["metrics"]["documents"][0]["counts"]["true_positive"] += 1
    payload["metrics"]["documents"][1]["counts"]["true_positive"] -= 1
    rewrite_fingerprint(payload)
    path = tmp_path / "invalid-document-fields.evidence.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(EvidenceValidationError) as caught:
        load_evidence_files((path,), arm="incumbent")

    assert caught.value.issues[0].json_path == "$.metrics.documents[0].fields"


def test_bank_discovery_rejects_stale_or_ambiguous_entries(tmp_path):
    bank = write_bank(tmp_path, arm="incumbent", seeds=(0,))
    (bank / "notes.txt").write_text("stale")

    with pytest.raises(EvidenceValidationError) as caught:
        discover_evidence_bank(bank, arm="incumbent")

    assert caught.value.issues[0].code == "unexpected_bank_entry"


def test_evidence_enforces_bank_count_and_file_size_limits(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    for index in range(17):
        (bank / f"{index}.evidence.json").write_text("{}")
    oversized = tmp_path / "oversized.evidence.json"
    with oversized.open("wb") as handle:
        handle.truncate(5_000_001)

    with pytest.raises(EvidenceValidationError) as bank_error:
        discover_evidence_bank(bank, arm="candidate")
    with pytest.raises(EvidenceValidationError) as file_error:
        load_evidence_files((oversized,), arm="candidate")

    assert bank_error.value.issues[0].code == "too_many_evidence_files"
    assert file_error.value.issues[0].code == "unreadable_evidence"


def test_arm_rejects_duplicate_seed_and_mixed_commits(tmp_path):
    first = make_evidence(arm="incumbent", seed=0)
    second = make_evidence(arm="incumbent", seed=0)
    second["run"]["run_id"] = "f" * 64
    second["provenance"]["repository_commit"] = "3" * 40
    rewrite_fingerprint(second)
    paths = []
    for index, payload in enumerate((first, second)):
        path = tmp_path / f"{index}.evidence.json"
        path.write_text(json.dumps(payload))
        paths.append(path)

    with pytest.raises(EvidenceValidationError) as caught:
        load_evidence_files(paths, arm="incumbent")

    assert caught.value.issues[0].code == "duplicate_seed"


def test_clean_committed_solution_snapshot_is_promotion_capable(tmp_path, monkeypatch):
    (tmp_path / "data" / "debug" / "texts").mkdir(parents=True)
    (tmp_path / "data" / "debug" / "texts" / "safe0000.txt").write_text("synthetic")
    (tmp_path / "data" / "debug" / "ground_truth.json").write_text('{"safe0000":[]}')
    (tmp_path / "solution.py").write_text("def extract_pii(_text):\n    return []\n")
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)

    with capture_evidence_context(run_id="a" * 32, dataset="debug") as context:
        snapshot_path = context.snapshot_directory / f"{context.solution_module}.py"
        assert snapshot_path.read_text() == (tmp_path / "solution.py").read_text()
        assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o400
        provenance = context.finalize()

    assert provenance.repository_available is True
    assert provenance.promotion_capable is True
    assert provenance.invalidation_reasons == ()


def test_dirty_repository_is_diagnostic_only(tmp_path, monkeypatch):
    (tmp_path / "data" / "debug" / "texts").mkdir(parents=True)
    (tmp_path / "data" / "debug" / "texts" / "safe0000.txt").write_text("synthetic")
    (tmp_path / "data" / "debug" / "ground_truth.json").write_text('{"safe0000":[]}')
    (tmp_path / "solution.py").write_text("def extract_pii(_text):\n    return []\n")
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    (tmp_path / "untracked.txt").write_text("dirty")
    monkeypatch.chdir(tmp_path)

    with capture_evidence_context(run_id="a" * 32, dataset="debug") as context:
        provenance = context.finalize()

    assert provenance.repository_clean_start is False
    assert provenance.promotion_capable is False
    assert "repository_dirty_at_start" in provenance.invalidation_reasons


def test_runtime_fingerprint_binds_normalized_upstream_endpoint():
    first = runtime_fingerprint(upstream_base_url="HTTPS://API.EXAMPLE.TEST/v1/")
    equivalent = runtime_fingerprint(upstream_base_url="https://api.example.test:443/v1")
    different = runtime_fingerprint(upstream_base_url="https://other.example.test/v1")

    assert first == equivalent
    assert first != different
