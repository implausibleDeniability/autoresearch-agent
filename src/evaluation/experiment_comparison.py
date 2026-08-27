import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from statistics import stdev
from typing import Dict, Mapping, Sequence, Tuple

from src.evaluation.evidence import (
    Counts,
    EvidenceDocument,
    EvidenceIssue,
    EvidenceRunRecord,
    EvidenceValidationError,
)

OUTPUT_SCHEMA_VERSION = 1
CALCULATION_VERSION = "paired-linearized-v1"
REQUIRED_INCUMBENT_SEEDS = (0, 1, 2)
FIXED_REPLAY_SEED_PREFIXES = ((0,), (0, 1), (0, 1, 2))
PRACTICAL_THRESHOLD = Fraction(3, 1000)
DIRECT_THRESHOLD = Fraction(1, 100)
ONE_SIDED_ALPHA = 0.05
Z_95_ONE_SIDED = 1.6448536269514722
Z_80_ONE_SIDED = 0.8416212335729143
CONCENTRATION_WARNING_THRESHOLD = 0.5


class ChangeType(StrEnum):
    RESPONSE_CHANGING = "response-changing"
    FIXED_REPLAY = "fixed-replay"


@dataclass(frozen=True)
class SeedComparison:
    seed: int
    incumbent_score: Fraction
    candidate_score: Fraction

    @property
    def delta(self) -> Fraction:
        return self.candidate_score - self.incumbent_score


@dataclass(frozen=True)
class ComparisonResult:
    payload: Mapping[str, object]

    @property
    def decision(self) -> str:
        return str(self.payload["decision"])

    def serialize(self) -> Dict[str, object]:
        return dict(self.payload)


def aggregate_score(counts: Counts) -> Fraction:
    denominator = 6 * counts.true_positive + counts.false_positive + 5 * counts.false_negative
    if denominator == 0:
        raise ValueError("score denominator is zero")
    return Fraction(6 * counts.true_positive, denominator)


def score_gradient(counts: Tuple[float, float, float]) -> Tuple[float, float, float]:
    true_positive, false_positive, false_negative = counts
    denominator = 6 * true_positive + false_positive + 5 * false_negative
    if denominator <= 0:
        raise ValueError("score-gradient denominator is zero")
    squared = denominator * denominator
    return (
        6 * (false_positive + 5 * false_negative) / squared,
        -6 * true_positive / squared,
        -30 * true_positive / squared,
    )


def compare_experiments(
    incumbent: Sequence[EvidenceRunRecord],
    candidate: Sequence[EvidenceRunRecord],
    *,
    change_type: ChangeType,
) -> ComparisonResult:
    _validate_comparison(incumbent, candidate, change_type=change_type)
    incumbent_by_seed = {run.seed: run for run in incumbent}
    candidate_by_seed = {run.seed: run for run in candidate}
    matched_seeds = tuple(sorted(candidate_by_seed))
    seed_comparisons = tuple(
        SeedComparison(
            seed=seed,
            incumbent_score=aggregate_score(incumbent_by_seed[seed].aggregate),
            candidate_score=aggregate_score(candidate_by_seed[seed].aggregate),
        )
        for seed in matched_seeds
    )
    mean_incumbent = _fraction_mean([item.incumbent_score for item in seed_comparisons])
    mean_candidate = _fraction_mean([item.candidate_score for item in seed_comparisons])
    delta = _fraction_mean([item.delta for item in seed_comparisons])
    influences = _paired_influences(
        [incumbent_by_seed[seed] for seed in matched_seeds],
        [candidate_by_seed[seed] for seed in matched_seeds],
    )
    standard_error = stdev(influences) / math.sqrt(len(influences)) if len(influences) > 1 else 0.0
    delta_float = float(delta)
    z_statistic, p_value = _one_sided_test(delta_float, standard_error=standard_error)
    upper_80 = delta_float + Z_80_ONE_SIDED * standard_error
    formal_look = change_type is ChangeType.FIXED_REPLAY or len(candidate) == 3
    decision = _decision(
        delta=delta,
        p_value=p_value,
        upper_80=upper_80,
        formal_look=formal_look,
    )
    field_deltas, field_warnings = _field_deltas(
        [incumbent_by_seed[seed] for seed in matched_seeds],
        [candidate_by_seed[seed] for seed in matched_seeds],
    )
    influence_rows, concentration = _influence_rows(
        influences,
        documents=incumbent_by_seed[matched_seeds[0]].documents,
    )
    warnings = list(field_warnings)
    if decision == "stop_for_futility":
        warnings.append(
            {
                "code": "early_futility_unseen_seed_risk",
                "message": "Unseen candidate-seed responses could reverse this heuristic stop.",
            }
        )
    if concentration > CONCENTRATION_WARNING_THRESHOLD:
        warnings.append(
            {
                "code": "influence_concentration",
                "message": "The ten most influential documents exceed 50% of squared influence.",
            }
        )
    next_action = _next_action(decision, candidate_runs=len(candidate), warnings=warnings)
    payload = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "calculation_version": CALCULATION_VERSION,
        "validation_status": "valid",
        "decision": decision,
        "decision_basis": _decision_basis(decision),
        "formal_look": formal_look,
        "change_type": change_type.value,
        "estimand": "arithmetic mean of matched-seed aggregate score deltas",
        "uncertainty": {
            "included": ["document-distribution variation conditional on matched response banks"],
            "excluded": [
                "unseen model-response variation",
                "backend drift",
                "collection hierarchy",
                "adaptive hypothesis selection",
            ],
        },
        "arms": {
            "incumbent": _arm_summary(incumbent),
            "candidate": _arm_summary(candidate),
        },
        "seed_panel": {
            "required_incumbent_seeds": list(
                REQUIRED_INCUMBENT_SEEDS if change_type is ChangeType.RESPONSE_CHANGING else matched_seeds
            ),
            "incumbent_seeds": [run.seed for run in incumbent],
            "candidate_seeds": [run.seed for run in candidate],
            "matched_seeds": list(matched_seeds),
        },
        "scores": {
            "incumbent_mean": _fraction_value(mean_incumbent),
            "candidate_mean": _fraction_value(mean_candidate),
            "delta": {
                **_fraction_value(delta),
                "percentage_points": 100 * delta_float,
            },
        },
        "uncertainty_metrics": {
            "standard_error_score_units": standard_error,
            "standard_error_percentage_points": 100 * standard_error,
            "z_statistic": z_statistic,
            "one_sided_p_value": p_value,
            "upper_80_score_units": upper_80,
            "upper_80_percentage_points": 100 * upper_80,
            "approximate": True,
            "degenerate_standard_error": standard_error == 0,
        },
        "thresholds": {
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "one_sided_alpha_z_critical": Z_95_ONE_SIDED,
            "practical_effect_fraction": "3/1000",
            "practical_effect_score_units": float(PRACTICAL_THRESHOLD),
            "direct_promotion_fraction": "1/100",
            "direct_promotion_score_units": float(DIRECT_THRESHOLD),
            "futility_upper_bound_quantile": 0.8,
            "futility_upper_bound_z_critical": Z_80_ONE_SIDED,
        },
        "per_seed": [
            {
                "seed": item.seed,
                "incumbent_score": _fraction_value(item.incumbent_score),
                "candidate_score": _fraction_value(item.candidate_score),
                "delta": _fraction_value(item.delta),
                "delta_percentage_points": 100 * float(item.delta),
            }
            for item in seed_comparisons
        ],
        "per_seed_delta_range": {
            "minimum_percentage_points": 100 * float(min(item.delta for item in seed_comparisons)),
            "maximum_percentage_points": 100 * float(max(item.delta for item in seed_comparisons)),
        },
        "influences": influence_rows,
        "influence_concentration": {
            "top_ten_share_of_squared_centered_influence": concentration,
            "warning_threshold": CONCENTRATION_WARNING_THRESHOLD,
        },
        "field_deltas": field_deltas,
        "eligibility_gates": _eligibility_gates(
            delta=delta,
            p_value=p_value,
            formal_look=formal_look,
            upper_80=upper_80,
        ),
        "reason_codes": _reason_codes(decision),
        "warnings": warnings,
        "errors": [],
        "next_action": next_action,
    }
    return ComparisonResult(payload)


def _validate_comparison(
    incumbent: Sequence[EvidenceRunRecord],
    candidate: Sequence[EvidenceRunRecord],
    *,
    change_type: ChangeType,
) -> None:
    issues = []
    incumbent_seeds = tuple(run.seed for run in incumbent)
    candidate_seeds = tuple(run.seed for run in candidate)
    if change_type is ChangeType.RESPONSE_CHANGING:
        if incumbent_seeds != REQUIRED_INCUMBENT_SEEDS:
            issues.append(
                _comparison_issue("incumbent_seed_panel", "Incumbent must contain seeds 0, 1, and 2.")
            )
        if candidate_seeds not in {(0,), (0, 1), (0, 1, 2)}:
            issues.append(
                _comparison_issue("candidate_seed_prefix", "Candidate seeds must be a prefix of 0, 1, and 2.")
            )
        if len({run.request_plan_fingerprint for run in candidate}) != len(candidate):
            issues.append(
                _comparison_issue(
                    "duplicate_candidate_request_plan",
                    "Candidate response-changing runs repeat a request plan.",
                )
            )
        if len({run.response_bank_fingerprint for run in candidate}) != len(candidate):
            issues.append(
                _comparison_issue(
                    "duplicate_candidate_response_bank",
                    "Candidate response-changing runs repeat a response bank.",
                )
            )
    else:
        if incumbent_seeds not in FIXED_REPLAY_SEED_PREFIXES or candidate_seeds != incumbent_seeds:
            issues.append(
                _comparison_issue(
                    "fixed_replay_seed_panel",
                    "Fixed replay requires the same seed prefix from 0 through 2 in both arms.",
                )
            )
        all_runs = (*incumbent, *candidate)
        if any(run.evaluation_mode != "cache" for run in all_runs):
            issues.append(
                _comparison_issue(
                    "fixed_replay_mode", "Fixed replay requires strict-cache evidence in both arms."
                )
            )
        if any(not run.receipts or not all(receipt.replayed for receipt in run.receipts) for run in all_runs):
            issues.append(
                _comparison_issue(
                    "fixed_replay_receipts", "Fixed replay requires non-empty all-replayed receipts."
                )
            )
        if len(incumbent) == len(candidate) and any(
            incumbent_run.receipts != candidate_run.receipts
            for incumbent_run, candidate_run in zip(incumbent, candidate)
        ):
            issues.append(
                _comparison_issue(
                    "fixed_replay_mismatch",
                    "Fixed replay request and response receipts differ within a matched seed.",
                )
            )
        for arm, runs in (("incumbent", incumbent), ("candidate", candidate)):
            if len({run.response_bank_fingerprint for run in runs}) != len(runs):
                issues.append(
                    _comparison_issue(
                        "duplicate_fixed_replay_response_bank",
                        f"Fixed replay repeats a response bank within the {arm} arm.",
                    )
                )
    all_runs = (*incumbent, *candidate)
    if len({run.run_id for run in all_runs}) != len(all_runs):
        issues.append(_comparison_issue("duplicate_run_across_arms", "A run ID appears in both arms."))
    if len({run.evidence_fingerprint for run in all_runs}) != len(all_runs):
        issues.append(
            _comparison_issue("duplicate_evidence_across_arms", "An evidence artifact appears in both arms.")
        )
    for attribute, code, problem in (
        (
            "scoring_contract_fingerprint",
            "mixed_scoring_contract",
            "Scoring contracts differ across evidence.",
        ),
        ("dataset_fingerprint", "mixed_dataset", "Dataset fingerprints differ across evidence."),
        ("runtime_fingerprint", "mixed_runtime", "Runtime fingerprints differ across evidence."),
    ):
        if len({getattr(run, attribute) for run in all_runs}) != 1:
            issues.append(_comparison_issue(code, problem))
    if incumbent and candidate:
        incumbent_ids = tuple(document.document_id for document in incumbent[0].documents)
        if any(
            tuple(document.document_id for document in run.documents) != incumbent_ids for run in all_runs
        ):
            issues.append(
                _comparison_issue(
                    "document_panel_mismatch", "Ordered document panels differ across evidence."
                )
            )
        if incumbent[0].solution_snapshot_fingerprint == candidate[0].solution_snapshot_fingerprint:
            issues.append(
                _comparison_issue(
                    "same_artifact", "Incumbent and candidate identify the same solution artifact."
                )
            )
    if issues:
        raise EvidenceValidationError(issues)


def _paired_influences(
    incumbent: Sequence[EvidenceRunRecord],
    candidate: Sequence[EvidenceRunRecord],
) -> Tuple[float, ...]:
    document_count = len(incumbent[0].documents)
    per_seed = []
    for incumbent_run, candidate_run in zip(incumbent, candidate):
        incumbent_mean = _mean_counts(incumbent_run.aggregate, document_count=document_count)
        candidate_mean = _mean_counts(candidate_run.aggregate, document_count=document_count)
        incumbent_gradient = score_gradient(incumbent_mean)
        candidate_gradient = score_gradient(candidate_mean)
        per_seed.append(
            tuple(
                _dot(candidate_gradient, _center(document.counts, candidate_mean))
                - _dot(incumbent_gradient, _center(incumbent_run.documents[index].counts, incumbent_mean))
                for index, document in enumerate(candidate_run.documents)
            )
        )
    return tuple(sum(seed[index] for seed in per_seed) / len(per_seed) for index in range(document_count))


def _field_deltas(
    incumbent: Sequence[EvidenceRunRecord],
    candidate: Sequence[EvidenceRunRecord],
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    names = sorted(set().union(*(run.fields for run in (*incumbent, *candidate))))
    rows = []
    warnings = []
    zero = Counts(0, 0, 0)
    for name in names:
        incumbent_precision = _fraction_mean([_precision(run.fields.get(name, zero)) for run in incumbent])
        candidate_precision = _fraction_mean([_precision(run.fields.get(name, zero)) for run in candidate])
        incumbent_recall = _fraction_mean([_recall(run.fields.get(name, zero)) for run in incumbent])
        candidate_recall = _fraction_mean([_recall(run.fields.get(name, zero)) for run in candidate])
        precision_delta = candidate_precision - incumbent_precision
        recall_delta = candidate_recall - incumbent_recall
        rows.append(
            {
                "field": name,
                "precision_delta_percentage_points": 100 * float(precision_delta),
                "recall_delta_percentage_points": 100 * float(recall_delta),
            }
        )
        for metric, delta in (("precision", precision_delta), ("recall", recall_delta)):
            if delta <= Fraction(-1, 100):
                warnings.append(
                    {
                        "code": "field_regression",
                        "field": name,
                        "metric": metric,
                        "delta_percentage_points": 100 * float(delta),
                    }
                )
    return rows, warnings


def _influence_rows(
    influences: Sequence[float],
    *,
    documents: Sequence[EvidenceDocument],
) -> tuple[list[Dict[str, object]], float]:
    centered = [value - sum(influences) / len(influences) for value in influences]
    total_squared = sum(value * value for value in centered)
    ranked = sorted(range(len(centered)), key=lambda index: (-abs(centered[index]), index))
    ranks = {index: rank + 1 for rank, index in enumerate(ranked)}
    top_ten_squared = sum(centered[index] ** 2 for index in ranked[:10])
    concentration = top_ten_squared / total_squared if total_squared else 0.0
    rows = [
        {
            "document_ordinal": index,
            "document_id": documents[index].document_id,
            "influence_score_units": centered[index],
            "influence_percentage_points": 100 * centered[index],
            "absolute_rank": ranks[index],
        }
        for index in range(len(centered))
    ]
    return rows, concentration


def _decision(
    *,
    delta: Fraction,
    p_value: float,
    upper_80: float,
    formal_look: bool,
) -> str:
    if not formal_look:
        return "stop_for_futility" if upper_80 < float(PRACTICAL_THRESHOLD) else "run_again"
    if p_value < ONE_SIDED_ALPHA and delta >= DIRECT_THRESHOLD:
        return "promotion_eligible"
    if p_value < ONE_SIDED_ALPHA and delta >= PRACTICAL_THRESHOLD:
        return "promotion_eligible_with_review"
    if upper_80 < float(PRACTICAL_THRESHOLD):
        return "reject"
    return "inconclusive"


def _one_sided_test(delta: float, *, standard_error: float) -> tuple[float | None, float]:
    if standard_error == 0:
        return None, 0.0 if delta > 0 else 1.0 if delta < 0 else 0.5
    z_statistic = delta / standard_error
    return z_statistic, 0.5 * math.erfc(z_statistic / math.sqrt(2))


def _arm_summary(runs: Sequence[EvidenceRunRecord]) -> Dict[str, object]:
    return {
        "repository_commit": runs[0].repository_commit,
        "solution_snapshot_fingerprint": runs[0].solution_snapshot_fingerprint,
        "run_count": len(runs),
        "run_ids": [run.run_id for run in runs],
        "evidence_fingerprints": [run.evidence_fingerprint for run in runs],
        "request_plan_fingerprints": [run.request_plan_fingerprint for run in runs],
        "response_bank_fingerprints": [run.response_bank_fingerprint for run in runs],
        "mean_observed_api_cost_usd": str(
            sum((run.observed_api_cost_usd for run in runs), start=0) / len(runs)
        ),
        "mean_duration_seconds": sum(run.duration_seconds for run in runs) / len(runs),
    }


def _fraction_value(value: Fraction) -> Dict[str, object]:
    return {"fraction": f"{value.numerator}/{value.denominator}", "score_units": float(value)}


def _fraction_mean(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction()) / len(values)


def _mean_counts(counts: Counts, *, document_count: int) -> Tuple[float, float, float]:
    return (
        counts.true_positive / document_count,
        counts.false_positive / document_count,
        counts.false_negative / document_count,
    )


def _center(counts: Counts, mean: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        counts.true_positive - mean[0],
        counts.false_positive - mean[1],
        counts.false_negative - mean[2],
    )


def _dot(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> float:
    return sum(first * second for first, second in zip(left, right))


def _precision(counts: Counts) -> Fraction:
    denominator = counts.true_positive + counts.false_positive
    return Fraction(counts.true_positive, denominator) if denominator else Fraction()


def _recall(counts: Counts) -> Fraction:
    denominator = counts.true_positive + counts.false_negative
    return Fraction(counts.true_positive, denominator) if denominator else Fraction()


def _eligibility_gates(
    *,
    delta: Fraction,
    p_value: float,
    formal_look: bool,
    upper_80: float,
) -> Dict[str, bool]:
    return {
        "formal_evidence_floor_met": formal_look,
        "one_sided_p_below_0_05": p_value < ONE_SIDED_ALPHA,
        "delta_at_least_0_3pp": delta >= PRACTICAL_THRESHOLD,
        "delta_at_least_1pp": delta >= DIRECT_THRESHOLD,
        "upper_80_below_0_3pp": upper_80 < float(PRACTICAL_THRESHOLD),
    }


def _reason_codes(decision: str) -> list[str]:
    return {
        "stop_for_futility": ["early_upper_bound_below_practical_threshold", "not_a_formal_reject"],
        "run_again": ["candidate_seed_floor_not_met", "early_futility_not_met"],
        "promotion_eligible": ["formal_significance_met", "delta_at_least_one_point"],
        "promotion_eligible_with_review": [
            "formal_significance_met",
            "delta_between_practical_and_direct_thresholds",
        ],
        "reject": ["formal_upper_bound_below_practical_threshold"],
        "inconclusive": ["formal_evidence_neither_promotes_nor_rejects"],
    }[decision]


def _decision_basis(decision: str) -> str:
    return {
        "stop_for_futility": (
            "Partial-bank 80% upper bound is below +0.3 percentage points; this is an early "
            "heuristic stop."
        ),
        "run_again": (
            "Candidate remains viable but has not reached the three-run response-changing " "evidence floor."
        ),
        "promotion_eligible": (
            "Formal one-sided significance and the +1 percentage-point threshold are both met."
        ),
        "promotion_eligible_with_review": (
            "Formal significance and the +0.3-point threshold are met; qualitative review remains "
            "required."
        ),
        "reject": ("At the formal look, the 80% upper confidence bound is below +0.3 percentage points."),
        "inconclusive": (
            "Formal evidence is neither promotion-eligible nor rejectable under the locked thresholds."
        ),
    }[decision]


def _next_action(decision: str, *, candidate_runs: int, warnings: Sequence[object]) -> Dict[str, object]:
    if decision == "run_again":
        return {"action": "run_candidate_seed", "next_seed": candidate_runs}
    if decision in {"stop_for_futility", "reject"}:
        return {"action": "restore_incumbent", "preserve_evidence": True}
    if decision == "promotion_eligible":
        additional_runs = max(3 - candidate_runs, 0)
        return {
            "action": "promote_exact_candidate_commit",
            "candidate_runs_become_initial_response_bank": True,
            "control_eligible": additional_runs == 0,
            "additional_control_runs_required": additional_runs,
        }
    if decision == "promotion_eligible_with_review":
        return {
            "action": "complete_qualitative_review",
            "blocking_checks": [
                "record the exact mechanism and artifact",
                "inspect field precision and recall deltas",
                "inspect the ten most influential document pairs",
                "confirm cost and runtime constraints",
                "resolve important unexplained warnings",
            ],
            "warning_count": len(warnings),
        }
    return {"action": "postpone_candidate", "preserve_evidence": True, "targeted_repeat_allowed": False}


def _comparison_issue(code: str, problem: str) -> EvidenceIssue:
    return EvidenceIssue(
        code=code,
        problem=problem,
        cause="The supplied banks do not satisfy the locked comparison design.",
        fix="Use exact complete banks under one evaluator and the required fixed seed panel.",
        docs_ref="research-runbook.md#paired-comparison-decisions",
    )
