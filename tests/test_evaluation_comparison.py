import math
import json
from fractions import Fraction
from statistics import stdev

import pytest

from src.evaluation.evidence import (
    Counts,
    EvidenceValidationError,
    discover_evidence_bank,
    load_evidence_files,
)
from src.evaluation.experiment_comparison import (
    ChangeType,
    DIRECT_THRESHOLD,
    ONE_SIDED_ALPHA,
    PRACTICAL_THRESHOLD,
    _decision,
    aggregate_score,
    compare_experiments,
    score_gradient,
)
from tests.comparison_fixtures import make_evidence, replace_receipts, rewrite_fingerprint, write_bank


def test_aggregate_score_reconstructs_recall_weighted_formula_exactly():
    assert aggregate_score(Counts(10, 2, 3)) == Fraction(60, 77)

    with pytest.raises(ValueError, match="denominator"):
        aggregate_score(Counts(0, 0, 0))


def test_score_gradient_matches_finite_differences():
    counts = (10.0, 2.0, 3.0)
    analytic = score_gradient(counts)
    step = 1e-5

    def score(values):
        true_positive, false_positive, false_negative = values
        return 6 * true_positive / (6 * true_positive + false_positive + 5 * false_negative)

    numerical = []
    for index in range(3):
        upper = list(counts)
        lower = list(counts)
        upper[index] += step
        lower[index] -= step
        numerical.append((score(upper) - score(lower)) / (2 * step))

    assert analytic == pytest.approx(numerical, rel=1e-8)


def test_point_estimate_is_mean_of_seed_scores_not_pooled_counts(tmp_path):
    incumbent_bank = tmp_path / "incumbent"
    candidate_bank = tmp_path / "candidate"
    incumbent_bank.mkdir()
    candidate_bank.mkdir()
    incumbent_payloads = (
        make_evidence(arm="incumbent", seed=0, per_document=lambda _ordinal: (1, 0, 1)),
        make_evidence(arm="incumbent", seed=1, per_document=lambda _ordinal: (100, 0, 0)),
        make_evidence(arm="incumbent", seed=2),
    )
    candidate_payloads = (
        make_evidence(arm="candidate", seed=0, per_document=lambda _ordinal: (1, 0, 0)),
        make_evidence(arm="candidate", seed=1, per_document=lambda _ordinal: (100, 0, 1)),
    )
    for seed, payload in enumerate(incumbent_payloads):
        (incumbent_bank / f"{seed}.evidence.json").write_text(json.dumps(payload))
    for seed, payload in enumerate(candidate_payloads):
        (candidate_bank / f"{seed}.evidence.json").write_text(json.dumps(payload))
    incumbent = _load(incumbent_bank, "incumbent")
    candidate = _load(candidate_bank, "candidate")

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert result.payload["scores"]["delta"]["fraction"] == "27/121"


def test_response_changing_one_run_can_stop_for_futility(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2)), "incumbent")
    candidate = _load(
        write_bank(tmp_path, arm="candidate", seeds=(0,), false_negatives=4),
        "candidate",
    )

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert result.decision == "stop_for_futility"
    assert result.payload["formal_look"] is False
    assert result.payload["next_action"]["action"] == "restore_incumbent"
    assert any(warning["code"] == "early_futility_unseen_seed_risk" for warning in result.payload["warnings"])


def test_response_changing_viable_prefix_requests_the_next_fixed_seed(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2)), "incumbent")
    candidate = _load(
        write_bank(tmp_path, arm="candidate", seeds=(0, 1), false_negatives=1),
        "candidate",
    )

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert result.decision == "run_again"
    assert result.payload["next_action"] == {"action": "run_candidate_seed", "next_seed": 2}


def test_three_run_large_uniform_improvement_is_promotion_eligible(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2)), "incumbent")
    candidate = _load(
        write_bank(tmp_path, arm="candidate", seeds=(0, 1, 2), false_negatives=1),
        "candidate",
    )

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert result.decision == "promotion_eligible"
    assert result.payload["formal_look"] is True
    assert result.payload["seed_panel"]["required_incumbent_seeds"] == [0, 1, 2]
    assert len(result.payload["influences"]) == 121
    assert result.payload["uncertainty_metrics"]["z_statistic"] is None
    assert result.payload["uncertainty_metrics"]["one_sided_p_value"] == 0


def test_three_run_moderate_improvement_requires_qualitative_review(tmp_path):
    incumbent = _load(
        write_bank(
            tmp_path,
            arm="incumbent",
            seeds=(0, 1, 2),
            per_document=lambda _ordinal: (100, 2, 2),
        ),
        "incumbent",
    )
    candidate = _load(
        write_bank(
            tmp_path,
            arm="candidate",
            seeds=(0, 1, 2),
            per_document=lambda _ordinal: (100, 2, 1),
        ),
        "candidate",
    )

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert result.decision == "promotion_eligible_with_review"
    assert result.payload["next_action"]["action"] == "complete_qualitative_review"


def test_concentrated_positive_gain_can_remain_inconclusive(tmp_path):
    incumbent = _load(
        write_bank(
            tmp_path,
            arm="incumbent",
            seeds=(0, 1, 2),
            per_document=lambda ordinal: (100, 2, 100 if ordinal == 0 else 2),
        ),
        "incumbent",
    )
    candidate = _load(
        write_bank(
            tmp_path,
            arm="candidate",
            seeds=(0, 1, 2),
            per_document=lambda ordinal: (100, 2, 0 if ordinal == 0 else 2),
        ),
        "candidate",
    )

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert result.decision == "inconclusive"
    assert result.payload["next_action"]["action"] == "postpone_candidate"
    assert result.payload["influence_concentration"]["top_ten_share_of_squared_centered_influence"] > 0.5
    influences = [row["influence_score_units"] for row in result.payload["influences"]]
    expected_standard_error = stdev(influences) / math.sqrt(121)
    assert sum(influences) == pytest.approx(0, abs=1e-12)
    assert result.payload["uncertainty_metrics"]["standard_error_score_units"] == pytest.approx(
        expected_standard_error
    )


def test_three_run_regression_is_formally_rejected(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2)), "incumbent")
    candidate = _load(
        write_bank(tmp_path, arm="candidate", seeds=(0, 1, 2), false_negatives=3),
        "candidate",
    )

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert result.decision == "reject"
    assert result.payload["eligibility_gates"]["upper_80_below_0_3pp"] is True


def test_formal_decision_thresholds_use_strict_p_and_upper_bound_boundaries():
    assert (
        _decision(
            delta=DIRECT_THRESHOLD,
            p_value=ONE_SIDED_ALPHA,
            upper_80=float(DIRECT_THRESHOLD),
            formal_look=True,
        )
        == "inconclusive"
    )
    assert (
        _decision(
            delta=PRACTICAL_THRESHOLD,
            p_value=0.049,
            upper_80=float(PRACTICAL_THRESHOLD),
            formal_look=True,
        )
        == "promotion_eligible_with_review"
    )
    assert (
        _decision(
            delta=Fraction(),
            p_value=0.5,
            upper_80=float(PRACTICAL_THRESHOLD),
            formal_look=True,
        )
        == "inconclusive"
    )
    assert (
        _decision(
            delta=Fraction(),
            p_value=0.5,
            upper_80=float(PRACTICAL_THRESHOLD) - 1e-12,
            formal_look=True,
        )
        == "reject"
    )


def test_comparison_rejects_mixed_runtime_and_nonprefix_candidate_seed(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2)), "incumbent")
    candidate_bank = write_bank(tmp_path, arm="candidate", seeds=(1,))
    candidate_path = next(candidate_bank.glob("*.evidence.json"))
    payload = json.loads(candidate_path.read_text())
    payload["provenance"]["runtime_fingerprint"] = "f" * 64
    rewrite_fingerprint(payload)
    candidate_path.write_text(json.dumps(payload))
    candidate = _load(candidate_bank, "candidate")

    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    codes = {issue.code for issue in caught.value.issues}
    assert {"candidate_seed_prefix", "mixed_runtime"} <= codes


def test_comparison_rejects_identical_solution_snapshots_across_commits(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2)), "incumbent")
    candidate_bank = write_bank(tmp_path, arm="candidate", seeds=(0,))
    candidate_path = next(candidate_bank.glob("*.evidence.json"))
    payload = json.loads(candidate_path.read_text())
    payload["provenance"]["solution_snapshot_fingerprint"] = "d" * 64
    rewrite_fingerprint(payload)
    candidate_path.write_text(json.dumps(payload))
    candidate = _load(candidate_bank, "candidate")

    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert any(issue.code == "same_artifact" for issue in caught.value.issues)


def test_fixed_replay_can_reach_a_formal_decision_from_one_identical_bank(tmp_path):
    incumbent = _load(
        write_bank(
            tmp_path,
            arm="incumbent",
            seeds=(0,),
            mode="cache",
            replayed=True,
            receipt_scope="fixed",
        ),
        "incumbent",
    )
    candidate = _load(
        write_bank(
            tmp_path,
            arm="candidate",
            seeds=(0,),
            false_negatives=1,
            mode="cache",
            replayed=True,
            receipt_scope="fixed",
        ),
        "candidate",
    )

    result = compare_experiments(incumbent, candidate, change_type=ChangeType.FIXED_REPLAY)

    assert result.decision == "promotion_eligible"
    assert result.payload["formal_look"] is True
    assert result.payload["seed_panel"]["required_incumbent_seeds"] == [0]
    assert result.payload["next_action"]["candidate_runs_become_initial_response_bank"] is True
    assert result.payload["next_action"]["control_eligible"] is False
    assert result.payload["next_action"]["additional_control_runs_required"] == 2


def test_fixed_replay_can_average_three_distinct_identical_bank_pairs(tmp_path):
    # setup
    incumbent = _load(
        write_bank(
            tmp_path,
            arm="incumbent",
            seeds=(0, 1, 2),
            mode="cache",
            replayed=True,
            receipt_scope="fixed",
        ),
        "incumbent",
    )
    candidate = _load(
        write_bank(
            tmp_path,
            arm="candidate",
            seeds=(0, 1, 2),
            false_negatives=1,
            mode="cache",
            replayed=True,
            receipt_scope="fixed",
        ),
        "candidate",
    )

    # operate
    result = compare_experiments(incumbent, candidate, change_type=ChangeType.FIXED_REPLAY)

    # check
    assert result.decision == "promotion_eligible"
    assert result.payload["seed_panel"]["matched_seeds"] == [0, 1, 2]
    assert result.payload["seed_panel"]["required_incumbent_seeds"] == [0, 1, 2]
    assert result.payload["next_action"]["control_eligible"] is True
    assert result.payload["next_action"]["additional_control_runs_required"] == 0


def test_fixed_replay_fails_closed_when_receipts_differ(tmp_path):
    incumbent = _load(
        write_bank(tmp_path, arm="incumbent", seeds=(0,), mode="cache", replayed=True),
        "incumbent",
    )
    candidate = _load(
        write_bank(tmp_path, arm="candidate", seeds=(0,), mode="cache", replayed=True),
        "candidate",
    )

    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(incumbent, candidate, change_type=ChangeType.FIXED_REPLAY)

    assert any(issue.code == "fixed_replay_mismatch" for issue in caught.value.issues)


def test_fixed_replay_checks_receipt_identity_for_every_seed(tmp_path):
    # setup
    incumbent_bank = write_bank(
        tmp_path,
        arm="incumbent",
        seeds=(0, 1),
        mode="cache",
        replayed=True,
        receipt_scope="fixed",
    )
    candidate_bank = write_bank(
        tmp_path,
        arm="candidate",
        seeds=(0, 1),
        mode="cache",
        replayed=True,
        receipt_scope="fixed",
    )
    candidate_path = candidate_bank / "seed-1.evidence.json"
    payload = json.loads(candidate_path.read_text())
    receipts = payload["requests"]["receipts"]
    receipts[0]["response_content_sha256"] = "f" * 64
    replace_receipts(payload, receipts=receipts)
    candidate_path.write_text(json.dumps(payload))

    # operate
    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(
            _load(incumbent_bank, "incumbent"),
            _load(candidate_bank, "candidate"),
            change_type=ChangeType.FIXED_REPLAY,
        )

    # check
    assert any(issue.code == "fixed_replay_mismatch" for issue in caught.value.issues)


@pytest.mark.parametrize(
    ("incumbent_seeds", "candidate_seeds"),
    [((0, 2), (0, 2)), ((0, 1, 2), (0, 1))],
)
def test_fixed_replay_rejects_invalid_seed_panels(tmp_path, incumbent_seeds, candidate_seeds):
    # setup
    incumbent = _load(
        write_bank(
            tmp_path,
            arm="incumbent",
            seeds=incumbent_seeds,
            mode="cache",
            replayed=True,
            receipt_scope="fixed",
        ),
        "incumbent",
    )
    candidate = _load(
        write_bank(
            tmp_path,
            arm="candidate",
            seeds=candidate_seeds,
            mode="cache",
            replayed=True,
            receipt_scope="fixed",
        ),
        "candidate",
    )

    # operate
    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(incumbent, candidate, change_type=ChangeType.FIXED_REPLAY)

    # check
    assert any(issue.code == "fixed_replay_seed_panel" for issue in caught.value.issues)


def test_fixed_replay_rejects_duplicate_response_banks(tmp_path):
    # setup
    incumbent_bank = write_bank(
        tmp_path,
        arm="incumbent",
        seeds=(0, 1),
        mode="cache",
        replayed=True,
        receipt_scope="fixed",
    )
    candidate_bank = write_bank(
        tmp_path,
        arm="candidate",
        seeds=(0, 1),
        mode="cache",
        replayed=True,
        receipt_scope="fixed",
    )
    for bank in (incumbent_bank, candidate_bank):
        seed_zero = json.loads((bank / "seed-0.evidence.json").read_text())
        seed_one_path = bank / "seed-1.evidence.json"
        seed_one = json.loads(seed_one_path.read_text())
        replace_receipts(seed_one, receipts=seed_zero["requests"]["receipts"])
        seed_one_path.write_text(json.dumps(seed_one))

    # operate
    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(
            _load(incumbent_bank, "incumbent"),
            _load(candidate_bank, "candidate"),
            change_type=ChangeType.FIXED_REPLAY,
        )

    # check
    assert any(issue.code == "duplicate_fixed_replay_response_bank" for issue in caught.value.issues)


def test_fixed_replay_fails_closed_without_strict_all_replayed_evidence(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0,)), "incumbent")
    candidate = _load(write_bank(tmp_path, arm="candidate", seeds=(0,)), "candidate")

    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(incumbent, candidate, change_type=ChangeType.FIXED_REPLAY)

    codes = {issue.code for issue in caught.value.issues}
    assert {"fixed_replay_mode", "fixed_replay_receipts"} <= codes


def test_comparison_rejects_a_reordered_document_panel(tmp_path):
    incumbent = _load(write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2)), "incumbent")
    candidate_bank = write_bank(tmp_path, arm="candidate", seeds=(0,))
    candidate_path = next(candidate_bank.glob("*.evidence.json"))
    payload = json.loads(candidate_path.read_text())
    documents = list(reversed(payload["metrics"]["documents"]))
    for ordinal, document in enumerate(documents):
        document["ordinal"] = ordinal
    payload["metrics"]["documents"] = documents
    payload["metrics"]["document_ids"] = [document["document_id"] for document in documents]
    rewrite_fingerprint(payload)
    candidate_path.write_text(json.dumps(payload))
    candidate = _load(candidate_bank, "candidate")

    with pytest.raises(EvidenceValidationError) as caught:
        compare_experiments(incumbent, candidate, change_type=ChangeType.RESPONSE_CHANGING)

    assert any(issue.code == "document_panel_mismatch" for issue in caught.value.issues)


def _load(bank, arm):
    return load_evidence_files(discover_evidence_bank(bank, arm=arm), arm=arm)
