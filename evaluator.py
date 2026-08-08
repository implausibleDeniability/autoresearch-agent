import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Literal, Sequence, Tuple, cast

from evaluation import EvaluationResult, Metrics, evaluate
from pii_item import PIIItem
from solution import extract_pii

MAX_DOCUMENT_WORKERS = 4


class Dataset:
    DEV = "dev"

    @classmethod
    def all(cls) -> Tuple[str]:
        return (cls.DEV,)


DatasetName = Literal["dev"]


@dataclass(frozen=True)
class EvaluationRun:
    dataset: DatasetName
    document_count: int
    elapsed_seconds: float
    result: EvaluationResult


def main() -> None:
    arguments = _parse_args(sys.argv[1:])
    run = run_evaluation(cast(DatasetName, arguments.dataset))
    _print_run(run)


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate solution.py on one dataset split.")
    parser.add_argument("--dataset", choices=Dataset.all(), required=True)
    return parser.parse_args(arguments)


def run_evaluation(dataset: DatasetName) -> EvaluationRun:
    document_paths = _document_paths(dataset)
    started_at = perf_counter()
    predictions = _extract_documents(document_paths)
    result = evaluate(predictions, ground_truth_path=_ground_truth_path(dataset))
    return EvaluationRun(
        dataset=dataset,
        document_count=len(document_paths),
        elapsed_seconds=perf_counter() - started_at,
        result=result,
    )


def _document_paths(dataset: DatasetName) -> List[Path]:
    return sorted((Path("data") / dataset / "texts").glob("*.txt"))


def _extract_documents(document_paths: Sequence[Path]) -> Dict[str, List[PIIItem]]:
    if not document_paths:
        return {}
    with ThreadPoolExecutor(max_workers=min(MAX_DOCUMENT_WORKERS, len(document_paths))) as executor:
        extracted = executor.map(_extract_document, document_paths)
        return {path.stem: people for path, people in zip(document_paths, extracted)}


def _extract_document(path: Path) -> List[PIIItem]:
    return extract_pii(path.read_text())


def _ground_truth_path(dataset: DatasetName) -> Path:
    return Path("data") / dataset / "ground_truth.json"


def _print_run(run: EvaluationRun) -> None:
    print(f"dataset={run.dataset}")
    print(f"documents={run.document_count}")
    print(f"elapsed_seconds={run.elapsed_seconds:.6f}")
    print("cost_usd_per_million_source_tokens=not_measured")
    _print_metrics("people", run.result.people)
    _print_metrics("entity", run.result.entities)
    print(f"document_accuracy={run.result.document_accuracy:.6f}")


def _print_metrics(name: str, metrics: Metrics) -> None:
    print(f"{name}_precision={metrics.precision:.6f}")
    print(f"{name}_recall={metrics.recall:.6f}")
    print(f"{name}_f1={metrics.f1:.6f}")


if __name__ == "__main__":
    main()
