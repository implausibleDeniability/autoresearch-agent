import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

from src.evaluation.evidence import canonical_fingerprint

DOCUMENT_COUNT = 121
SHARED_SCORING_FINGERPRINT = "a" * 64
SHARED_DATASET_FINGERPRINT = "b" * 64
SHARED_RUNTIME_FINGERPRINT = "c" * 64


def write_bank(
    root: Path,
    *,
    arm: str,
    seeds: Sequence[int],
    false_negatives: int = 2,
    false_positives: int = 2,
    mode: str = "fresh",
    replayed: bool = False,
    receipt_scope: str = "",
    per_document: Callable[[int], tuple[int, int, int]] | None = None,
) -> Path:
    bank = root / arm
    bank.mkdir(exist_ok=True)
    for seed in seeds:
        payload = make_evidence(
            arm=arm,
            seed=seed,
            false_negatives=false_negatives,
            false_positives=false_positives,
            mode=mode,
            replayed=replayed,
            receipt_scope=receipt_scope,
            per_document=per_document,
        )
        path = bank / f"seed-{seed}.evidence.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return bank


def make_evidence(
    *,
    arm: str,
    seed: int,
    false_negatives: int = 2,
    false_positives: int = 2,
    mode: str = "fresh",
    replayed: bool = False,
    receipt_scope: str = "",
    per_document: Callable[[int], tuple[int, int, int]] | None = None,
) -> dict[str, object]:
    document_counts = [
        per_document(ordinal) if per_document is not None else (10, false_positives, false_negatives)
        for ordinal in range(DOCUMENT_COUNT)
    ]
    documents = [
        {
            "ordinal": ordinal,
            "document_id": f"synthetic-{ordinal:03d}",
            "status": "completed",
            "counts": _counts(*counts),
            "fields": {"synthetic": _counts(*counts)},
        }
        for ordinal, counts in enumerate(document_counts)
    ]
    aggregate = tuple(sum(values) for values in zip(*document_counts))
    scope = receipt_scope or arm
    receipts = [
        {
            "document_ordinal": ordinal,
            "request_ordinal": 0,
            "request_key": _digest(f"request:{scope}:{seed}:{ordinal}"),
            "response_content_sha256": _digest(f"response:{scope}:{seed}:{ordinal}"),
            "replayed": replayed,
        }
        for ordinal in range(DOCUMENT_COUNT)
    ]
    plan = [
        {
            "document_ordinal": receipt["document_ordinal"],
            "request_ordinal": receipt["request_ordinal"],
            "request_key": receipt["request_key"],
        }
        for receipt in receipts
    ]
    payload = {
        "artifact_kind": "pii_comparison_evidence",
        "schema_version": 1,
        "fingerprint_algorithm": "sha256",
        "run": {
            "run_id": _digest(f"run:{arm}:{seed}"),
            "dataset": "dev-202k",
            "lifecycle_status": "terminal",
            "result_status": "complete",
            "score_is_final": True,
            "termination_category": "none",
            "evaluation_seed": seed,
            "evaluation_mode": mode,
            "duration_seconds": 1.0,
            "coverage": {
                "total": DOCUMENT_COUNT,
                "completed": DOCUMENT_COUNT,
                "failed": 0,
                "not_attempted": 0,
            },
            "metering_status": "complete",
            "metering_error_count": 0,
            "observed_api_cost_usd": "0.01",
        },
        "provenance": {
            "repository_available": True,
            "repository_commit": ("1" if arm == "incumbent" else "2") * 40,
            "repository_clean_start": True,
            "repository_clean_end": True,
            "solution_matches_head_start": True,
            "solution_matches_snapshot_end": True,
            "repository_commit_unchanged": True,
            "solution_snapshot_fingerprint": ("d" if arm == "incumbent" else "e") * 64,
            "scoring_contract_fingerprint": SHARED_SCORING_FINGERPRINT,
            "dataset_fingerprint": SHARED_DATASET_FINGERPRINT,
            "runtime_fingerprint": SHARED_RUNTIME_FINGERPRINT,
            "promotion_capable": True,
            "invalidation_reasons": [],
        },
        "metrics": {
            "aggregate": _counts(*aggregate),
            "fields": {"synthetic": _counts(*aggregate)},
            "document_ids": [document["document_id"] for document in documents],
            "documents": documents,
        },
        "requests": {
            "status": "complete",
            "receipt_count": len(receipts),
            "all_replayed": replayed,
            "receipts": receipts,
            "request_plan_fingerprint": canonical_fingerprint(plan),
            "response_bank_fingerprint": canonical_fingerprint(receipts),
        },
    }
    payload["evidence_fingerprint"] = canonical_fingerprint(payload)
    return payload


def rewrite_fingerprint(payload: dict[str, object]) -> None:
    payload.pop("evidence_fingerprint", None)
    payload["evidence_fingerprint"] = canonical_fingerprint(payload)


def replace_receipts(
    payload: dict[str, object],
    *,
    receipts: Sequence[dict[str, object]],
) -> None:
    copied = [dict(receipt) for receipt in receipts]
    plan = [
        {
            "document_ordinal": receipt["document_ordinal"],
            "request_ordinal": receipt["request_ordinal"],
            "request_key": receipt["request_key"],
        }
        for receipt in copied
    ]
    payload["requests"]["receipts"] = copied
    payload["requests"]["receipt_count"] = len(copied)
    payload["requests"]["all_replayed"] = bool(copied) and all(receipt["replayed"] for receipt in copied)
    payload["requests"]["request_plan_fingerprint"] = canonical_fingerprint(plan)
    payload["requests"]["response_bank_fingerprint"] = canonical_fingerprint(copied)
    rewrite_fingerprint(payload)


def _counts(true_positive: int, false_positive: int, false_negative: int) -> dict[str, int]:
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
