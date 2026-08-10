import json
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from solution import extract_pii
from src.evaluation.metrics import EntityMetrics, evaluate
from src.evaluation.models import PIIItem

DEV_TEXTS = Path("data/dev-50k/texts")
DEV_GROUND_TRUTH = Path("data/dev-50k/ground_truth.json")
SMALL_DOCUMENTS = ("fjwg0257", "pzvv0257", "nrbn0226")
LARGE_DOCUMENT = "rtbn0226"


@pytest.mark.live
def test_live_synthetic_positive_and_negative_extraction():
    # setup
    positive = "John A. Smith can be reached at john.smith@example.com."
    negative = "Quarterly revenue increased while operating expenses declined."

    # operate
    positive_result = extract_pii(positive)
    negative_result = extract_pii(negative)

    # check
    assert _contains_person(positive_result, first_name="John", last_name="Smith")
    assert negative_result == []


@pytest.mark.live
def test_live_development_evaluation(tmp_path: Path):
    # setup
    small_ground_truth = _filter_ground_truth(tmp_path, document_ids=SMALL_DOCUMENTS)
    full_document_ids = (*SMALL_DOCUMENTS, LARGE_DOCUMENT)
    full_ground_truth = _filter_ground_truth(tmp_path, document_ids=full_document_ids)

    # operate: evaluate the three smaller documents first
    predictions = _extract_documents(SMALL_DOCUMENTS)
    small_result = evaluate(predictions, ground_truth_path=small_ground_truth)

    # check: stop before the expensive document if the baseline is structurally broken
    _assert_positive_baseline(small_result)

    # operate: add the large document and evaluate the complete visible split
    predictions[LARGE_DOCUMENT] = extract_pii(_read_document(LARGE_DOCUMENT))
    full_result = evaluate(predictions, ground_truth_path=full_ground_truth)

    # check: report and retain a minimal quality floor
    print(f"small_dev={small_result}")
    print(f"full_dev={full_result}")
    _assert_positive_baseline(full_result)


def _extract_documents(document_ids: Sequence[str]) -> Dict[str, List[PIIItem]]:
    return {document_id: extract_pii(_read_document(document_id)) for document_id in document_ids}


def _read_document(document_id: str) -> str:
    return (DEV_TEXTS / f"{document_id}.txt").read_text()


def _filter_ground_truth(tmp_path: Path, *, document_ids: Sequence[str]) -> Path:
    target = tmp_path / "ground_truth.json"
    ground_truth = json.loads(DEV_GROUND_TRUTH.read_text())
    selected = {document_id: ground_truth[document_id] for document_id in document_ids}
    target.write_text(json.dumps(selected))
    return target


def _contains_person(people: Sequence[PIIItem], *, first_name: str, last_name: str) -> bool:
    return any(first_name in person.first_name and last_name in person.last_name for person in people)


def _assert_positive_baseline(metrics: EntityMetrics) -> None:
    assert metrics.f_score > 0
