import json
import subprocess
from pathlib import Path

from src.evaluation.compare_cli import main
from tests.comparison_fixtures import write_bank


def test_cli_emits_one_json_object_for_a_valid_scientific_outcome(tmp_path, capsys):
    incumbent = write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2))
    candidate = write_bank(tmp_path, arm="candidate", seeds=(0,), false_negatives=1)

    exit_code = main(("--incumbent", str(incumbent), "--candidate", str(candidate)))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert payload["validation_status"] == "valid"
    assert payload["decision"] == "run_again"
    assert payload["output_schema_version"] == 1


def test_cli_emits_structured_redacted_invalid_evidence(tmp_path, capsys):
    incumbent = write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2))
    candidate = write_bank(tmp_path, arm="candidate", seeds=(0,), false_negatives=1)
    path = next(candidate.glob("*.json"))
    path.write_text(path.read_text().replace('"dataset": "dev-202k"', '"dataset": "secret value"'))

    exit_code = main(("--incumbent", str(incumbent), "--candidate", str(candidate)))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["validation_status"] == "invalid"
    assert payload["decision"] is None
    assert payload["next_action"]["action"] == "fix_evidence"
    assert "secret value" not in captured.out


def test_cli_usage_error_is_json_and_exit_two(capsys):
    exit_code = main(("--change-type", "response-changing"))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["errors"][0]["code"] == "invalid_usage"


def test_cli_accepts_repeatable_exact_files_in_any_argument_order(tmp_path, capsys):
    incumbent = write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2))
    candidate = write_bank(tmp_path, arm="candidate", seeds=(0,), false_negatives=1)
    incumbent_paths = tuple(reversed(sorted(incumbent.glob("*.evidence.json"))))

    exit_code = main(
        (
            "--candidate-file",
            str(next(candidate.glob("*.evidence.json"))),
            "--incumbent-file",
            str(incumbent_paths[0]),
            "--incumbent-file",
            str(incumbent_paths[1]),
            "--incumbent-file",
            str(incumbent_paths[2]),
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["seed_panel"]["incumbent_seeds"] == [0, 1, 2]


def test_cli_rejects_mixed_directory_and_file_forms(tmp_path, capsys):
    incumbent = write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2))
    candidate = write_bank(tmp_path, arm="candidate", seeds=(0,), false_negatives=1)

    exit_code = main(
        (
            "--incumbent",
            str(incumbent),
            "--incumbent-file",
            str(next(incumbent.glob("*.evidence.json"))),
            "--candidate",
            str(candidate),
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["errors"][0]["code"] == "invalid_usage"


def test_cli_text_leads_with_decision_delta_and_next_action(tmp_path, capsys):
    incumbent = write_bank(tmp_path, arm="incumbent", seeds=(0, 1, 2))
    candidate = write_bank(tmp_path, arm="candidate", seeds=(0,), false_negatives=1)

    exit_code = main(
        (
            "--incumbent",
            str(incumbent),
            "--candidate",
            str(candidate),
            "--format",
            "text",
        )
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.startswith("Decision: run_again | delta=")
    assert "Next action: run_candidate_seed" in output
    assert "Incumbent: commit=" in output
    assert "Mean scores: incumbent=" in output
    assert "Per-seed deltas: seed 0=" in output
    assert "Per-seed delta range:" in output
    assert "Top-ten squared-influence share:" in output
    assert "Eligibility gates:" in output
    assert "Most influential documents:" in output


def test_cli_compares_a_three_seed_fixed_replay_panel(tmp_path, capsys):
    # setup
    incumbent = write_bank(
        tmp_path,
        arm="incumbent",
        seeds=(0, 1, 2),
        mode="cache",
        replayed=True,
        receipt_scope="fixed",
    )
    candidate = write_bank(
        tmp_path,
        arm="candidate",
        seeds=(0, 1, 2),
        false_negatives=1,
        mode="cache",
        replayed=True,
        receipt_scope="fixed",
    )

    # operate
    exit_code = main(
        (
            "--incumbent",
            str(incumbent),
            "--candidate",
            str(candidate),
            "--change-type",
            "fixed-replay",
        )
    )

    # check
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["decision"] == "promotion_eligible"
    assert payload["seed_panel"]["matched_seeds"] == [0, 1, 2]


def test_installed_console_entry_runs_the_documented_pii_free_example():
    repository = Path(__file__).parents[1]

    completed = subprocess.run(
        [
            "pii-compare",
            "--incumbent",
            "examples/comparison-evidence/incumbent",
            "--candidate",
            "examples/comparison-evidence/candidate",
            "--format",
            "text",
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("Decision: run_again | delta=")


def test_installed_console_entry_documents_examples_and_version():
    help_result = subprocess.run(
        ("pii-compare", "--help"),
        text=True,
        capture_output=True,
        check=False,
    )
    version_result = subprocess.run(
        ("pii-compare", "--version"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert help_result.returncode == 0
    assert "Examples:" in help_result.stdout
    assert "--change-type" in help_result.stdout
    assert version_result.stdout.strip() == "pii-compare 1.0.0"
