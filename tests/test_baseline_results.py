import csv
import math
import re
from pathlib import Path

import pytest

from src.evaluation.cli import _count_source_tokens, _load_texts
from src.evaluation.results import EntityMetrics

BASELINE_PATH = Path(__file__).parents[1] / "baseline-results.tsv"
FIELDS = [
    "run",
    "commit",
    "dataset",
    "score",
    "precision",
    "recall",
    "true_positive",
    "false_positive",
    "false_negative",
    "cost",
    "budget_cost_usd",
    "duration_seconds",
]
DATASETS = ("dev-19k", "dev-87k", "dev-202k")


def test_saved_baselines_are_complete_and_internally_consistent():
    with BASELINE_PATH.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows = list(reader)

    assert reader.fieldnames == FIELDS, "baseline header changed"
    assert len(rows) == 15, "baseline must contain five runs for each development dataset"
    assert [int(row["run"]) for row in rows] == list(range(1, 16)), "baseline run numbers must be 1-15"
    assert [row["dataset"] for row in rows] == [dataset for dataset in DATASETS for _ in range(5)]

    commits = {row["commit"] for row in rows}
    assert len(commits) == 1, "baseline rows must use one solution commit"
    assert re.fullmatch(r"[0-9a-f]{7}", commits.pop()), "baseline solution commit must be seven hex digits"

    source_tokens = {dataset: _count_source_tokens(_load_texts(dataset)) for dataset in DATASETS}
    for row in rows:
        label = f"baseline row {row['run']} ({row['dataset']})"
        metrics = EntityMetrics(
            true_positive=int(row["true_positive"]),
            false_positive=int(row["false_positive"]),
            false_negative=int(row["false_negative"]),
        )
        reported = {
            "score": float(row["score"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "cost": float(row["cost"]),
            "budget_cost_usd": float(row["budget_cost_usd"]),
            "duration_seconds": float(row["duration_seconds"]),
        }

        assert all(math.isfinite(value) for value in reported.values()), f"{label} has a non-finite value"
        for field in ("score", "precision", "recall"):
            assert 0.0 <= reported[field] <= 1.0, f"{label} has invalid {field}"
        for field in ("cost", "budget_cost_usd", "duration_seconds"):
            assert reported[field] > 0.0, f"{label} has non-positive {field}"

        assert reported["precision"] == pytest.approx(
            metrics.precision, abs=1e-6
        ), f"{label} precision does not match its counts"
        assert reported["recall"] == pytest.approx(
            metrics.recall, abs=1e-6
        ), f"{label} recall does not match its counts"
        assert reported["score"] == pytest.approx(
            metrics.f_score, abs=1e-6
        ), f"{label} score does not match its counts"
        normalized_cost = reported["budget_cost_usd"] * 1_000_000 / source_tokens[row["dataset"]]
        assert reported["cost"] == pytest.approx(
            normalized_cost, abs=1e-6
        ), f"{label} normalized cost does not match its spend and source tokens"
