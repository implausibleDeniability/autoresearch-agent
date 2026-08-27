import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Sequence, Tuple

from src.evaluation.evidence import (
    EvidenceIssue,
    EvidenceValidationError,
    discover_evidence_bank,
    load_evidence_files,
)
from src.evaluation.experiment_comparison import (
    CALCULATION_VERSION,
    OUTPUT_SCHEMA_VERSION,
    ChangeType,
    compare_experiments,
)

COMPARATOR_VERSION = "1.0.0"


class UsageError(ValueError):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def main(arguments: Sequence[str] = ()) -> int:
    debug = "--debug" in arguments
    try:
        parsed = _parse_arguments(arguments or sys.argv[1:])
        debug = parsed.debug
        incumbent_paths = _arm_paths(parsed.incumbent, parsed.incumbent_file, arm="incumbent")
        candidate_paths = _arm_paths(parsed.candidate, parsed.candidate_file, arm="candidate")
        incumbent = load_evidence_files(incumbent_paths, arm="incumbent")
        candidate = load_evidence_files(candidate_paths, arm="candidate")
        result = compare_experiments(
            incumbent,
            candidate,
            change_type=ChangeType(parsed.change_type),
        ).serialize()
        _render(result, output_format=parsed.format)
        return 0
    except UsageError:
        _print_json(_invalid_payload([_usage_issue()]))
        return 2
    except EvidenceValidationError as error:
        _print_json(_invalid_payload(error.issues))
        return 2
    except Exception:
        if debug:
            traceback.print_exc(file=sys.stderr)
        _print_json(_internal_error_payload())
        return 1


def entrypoint() -> int:
    return main(sys.argv[1:])


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = JsonArgumentParser(
        description="Compare complete dev-202k experiment evidence with paired linearization.",
        epilog=(
            "Examples:\n"
            "  pii-compare --incumbent diagnostics/incumbent-bank --candidate diagnostics/candidate-bank\n"
            "  pii-compare --incumbent-file control-0.evidence.json --candidate-file change-0.evidence.json "
            "--change-type fixed-replay --format text"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--incumbent", type=Path, help="non-recursive incumbent evidence-bank directory")
    parser.add_argument("--candidate", type=Path, help="non-recursive candidate evidence-bank directory")
    parser.add_argument(
        "--incumbent-file",
        type=Path,
        action="append",
        default=[],
        help="exact incumbent .evidence.json file; repeat for a panel",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        action="append",
        default=[],
        help="exact candidate .evidence.json file; repeat for a panel",
    )
    parser.add_argument(
        "--change-type",
        choices=tuple(change_type.value for change_type in ChangeType),
        default=ChangeType.RESPONSE_CHANGING.value,
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--debug", action="store_true", help="write a local traceback to stderr")
    parser.add_argument("--version", action="version", version=f"pii-compare {COMPARATOR_VERSION}")
    parsed = parser.parse_args(arguments)
    _validate_input_forms(parsed)
    return parsed


def _validate_input_forms(parsed: argparse.Namespace) -> None:
    for arm in ("incumbent", "candidate"):
        directory = getattr(parsed, arm)
        files = getattr(parsed, f"{arm}_file")
        if bool(directory) == bool(files):
            raise UsageError(f"{arm} requires exactly one of directory or repeatable file inputs")


def _arm_paths(directory: Path | None, files: Sequence[Path], *, arm: str) -> Tuple[Path, ...]:
    return discover_evidence_bank(directory, arm=arm) if directory is not None else tuple(files)


def _render(payload: Dict[str, object], *, output_format: str) -> None:
    if output_format == "json":
        _print_json(payload)
        return
    scores = payload["scores"]
    uncertainty = payload["uncertainty_metrics"]
    next_action = payload["next_action"]
    print(
        f"Decision: {payload['decision']} | delta={scores['delta']['percentage_points']:.3f}pp | "
        f"SE={uncertainty['standard_error_percentage_points']:.3f}pp | "
        f"p={uncertainty['one_sided_p_value']:.6g} | "
        f"U80={uncertainty['upper_80_percentage_points']:.3f}pp"
    )
    print(f"Next action: {next_action['action']}")
    print(f"Basis: {payload['decision_basis']}")
    print("\nExplanation:")
    print(f"  Change type: {payload['change_type']}; formal look: {payload['formal_look']}")
    for arm in ("incumbent", "candidate"):
        summary = payload["arms"][arm]
        print(
            f"  {arm.title()}: commit={summary['repository_commit']} "
            f"runs={summary['run_count']} snapshot={summary['solution_snapshot_fingerprint']}"
        )
    print(
        f"  Mean scores: incumbent={scores['incumbent_mean']['score_units']:.6f}; "
        f"candidate={scores['candidate_mean']['score_units']:.6f}"
    )
    print(f"  Matched seeds: {payload['seed_panel']['matched_seeds']}")
    print(
        "  Per-seed deltas: "
        + ", ".join(
            f"seed {row['seed']}={row['delta_percentage_points']:+.3f}pp" for row in payload["per_seed"]
        )
    )
    seed_range = payload["per_seed_delta_range"]
    print(
        f"  Per-seed delta range: {seed_range['minimum_percentage_points']:+.3f}pp to "
        f"{seed_range['maximum_percentage_points']:+.3f}pp"
    )
    print(
        "  Top-ten squared-influence share: "
        f"{100 * payload['influence_concentration']['top_ten_share_of_squared_centered_influence']:.2f}%"
    )
    gates = payload["eligibility_gates"]
    print(
        "  Eligibility gates: "
        + ", ".join(f"{name}={'pass' if passed else 'fail'}" for name, passed in gates.items())
    )
    print("Most influential documents:")
    ranked = sorted(payload["influences"], key=lambda row: row["absolute_rank"])
    for row in ranked[:10]:
        print(
            f"  {row['absolute_rank']:>2}. document={row['document_id']} "
            f"influence={row['influence_percentage_points']:+.4f}pp"
        )
    warnings = payload["warnings"]
    if warnings:
        print(f"Warnings: {len(warnings)}; inspect JSON output for structured details.")


def _invalid_payload(issues: Sequence[EvidenceIssue]) -> Dict[str, object]:
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "calculation_version": CALCULATION_VERSION,
        "validation_status": "invalid",
        "decision": None,
        "decision_basis": "Comparison evidence is invalid; no scientific decision was calculated.",
        "formal_look": None,
        "change_type": None,
        "errors": [issue.serialize() for issue in issues[:20]],
        "warnings": [],
        "next_action": {"action": "fix_evidence"},
    }


def _usage_issue() -> EvidenceIssue:
    return EvidenceIssue(
        code="invalid_usage",
        problem="Command-line arguments do not identify exactly one input form for each arm.",
        cause="A required argument is missing, duplicated, or unsupported.",
        fix="Run pii-compare --help and provide one bank directory or repeatable exact files per arm.",
        docs_ref="research-runbook.md#paired-comparison-decisions",
    )


def _internal_error_payload() -> Dict[str, object]:
    issue = EvidenceIssue(
        code="internal_error",
        problem="The comparison could not complete because of an internal failure.",
        cause="A local calculation or serialization step failed unexpectedly.",
        fix="Rerun with --debug, preserve the evidence, and report the local traceback.",
        docs_ref="research-runbook.md#paired-comparison-decisions",
    )
    payload = _invalid_payload([issue])
    payload["next_action"] = {"action": "report_internal_error"}
    return payload


def _print_json(payload: Dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
