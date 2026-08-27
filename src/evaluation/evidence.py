import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, cast

from src.evaluation.provenance import EvidenceContext, FinalProvenance
from src.evaluation.results import DocumentEvaluation, EntityMetrics, EvaluationTrace
from src.evaluation.run_results import EvaluationRun, ResultStatus

EVIDENCE_ARTIFACT_KIND = "pii_comparison_evidence"
EVIDENCE_SCHEMA_VERSION = 1
FINGERPRINT_ALGORITHM = "sha256"
EXPECTED_DATASET = "dev-202k"
EXPECTED_DOCUMENT_COUNT = 121
MAX_EVIDENCE_BYTES = 5_000_000
MAX_EVIDENCE_FILES = 16
MAX_JSON_DEPTH = 32
MAX_INTEGER = 1_000_000_000_000


@dataclass(frozen=True)
class Counts:
    true_positive: int
    false_positive: int
    false_negative: int

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.true_positive + other.true_positive,
            self.false_positive + other.false_positive,
            self.false_negative + other.false_negative,
        )


@dataclass(frozen=True)
class EvidenceReceipt:
    document_ordinal: int
    request_ordinal: int
    request_key: str
    response_content_sha256: str
    replayed: bool


@dataclass(frozen=True)
class EvidenceDocument:
    ordinal: int
    document_id: str
    counts: Counts
    fields: Mapping[str, Counts]


@dataclass(frozen=True)
class EvidenceRunRecord:
    run_id: str
    seed: int
    evaluation_mode: str
    repository_commit: str
    solution_snapshot_fingerprint: str
    scoring_contract_fingerprint: str
    dataset_fingerprint: str
    runtime_fingerprint: str
    aggregate: Counts
    fields: Mapping[str, Counts]
    documents: Tuple[EvidenceDocument, ...]
    receipts: Tuple[EvidenceReceipt, ...]
    request_plan_fingerprint: str
    response_bank_fingerprint: str
    evidence_fingerprint: str
    observed_api_cost_usd: Decimal
    duration_seconds: float


@dataclass(frozen=True)
class EvidenceIssue:
    code: str
    problem: str
    cause: str
    fix: str
    docs_ref: str
    arm: str = ""
    input_index: Optional[int] = None
    json_path: str = ""
    expected: str = ""

    def serialize(self) -> Dict[str, object]:
        payload = {
            "code": self.code,
            "problem": self.problem,
            "cause": self.cause,
            "fix": self.fix,
            "docs_ref": self.docs_ref,
        }
        for name in ("arm", "input_index", "json_path", "expected"):
            value = getattr(self, name)
            if value not in ("", None):
                payload[name] = value
        return payload


class EvidenceValidationError(ValueError):
    def __init__(self, issues: Sequence[EvidenceIssue]) -> None:
        super().__init__(issues[0].problem if issues else "evidence validation failed")
        self.issues = tuple(issues[:20])


def evidence_path_for(diagnostics_path: Path) -> Path:
    if diagnostics_path.suffix != ".json":
        raise ValueError("diagnostics path must end in .json to derive its evidence sidecar")
    return diagnostics_path.with_name(f"{diagnostics_path.stem}.evidence.json")


def write_evidence(
    path: Path,
    *,
    run: EvaluationRun,
    context: EvidenceContext,
    provenance: Optional[FinalProvenance] = None,
) -> None:
    resolved_provenance = provenance or context.running_provenance()
    payload = serialize_evidence(run, context=context, provenance=resolved_provenance)
    payload["evidence_fingerprint"] = canonical_fingerprint(payload)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def serialize_evidence(
    run: EvaluationRun,
    *,
    context: EvidenceContext,
    provenance: FinalProvenance,
) -> Dict[str, object]:
    final_response_bank = (
        run.lifecycle_status == "terminal"
        and run.result_status == ResultStatus.COMPLETE
        and run.cost.status == "complete"
    )
    receipts = [asdict(receipt) for receipt in run.cost.request_receipts]
    requests = {
        "status": "complete" if final_response_bank else "incomplete",
        "receipt_count": len(receipts),
        "all_replayed": bool(receipts) and all(receipt["replayed"] for receipt in receipts),
        "receipts": receipts,
        "request_plan_fingerprint": (
            canonical_fingerprint(
                [
                    {
                        "document_ordinal": receipt["document_ordinal"],
                        "request_ordinal": receipt["request_ordinal"],
                        "request_key": receipt["request_key"],
                    }
                    for receipt in receipts
                ]
            )
            if final_response_bank
            else None
        ),
        "response_bank_fingerprint": canonical_fingerprint(receipts) if final_response_bank else None,
    }
    statuses = [document.status for document in run.documents]
    return {
        "artifact_kind": EVIDENCE_ARTIFACT_KIND,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "run": {
            "run_id": run.run_id,
            "dataset": run.dataset,
            "lifecycle_status": run.lifecycle_status,
            "result_status": run.result_status,
            "score_is_final": run.result_status == ResultStatus.COMPLETE,
            "termination_category": run.termination_category,
            "evaluation_seed": run.evaluation_seed,
            "evaluation_mode": run.cost.evaluation_mode.value,
            "duration_seconds": context.duration_seconds,
            "coverage": {
                "total": len(run.documents),
                "completed": statuses.count("completed"),
                "failed": statuses.count("failed"),
                "not_attempted": statuses.count("not_attempted"),
            },
            "metering_status": run.cost.status,
            "metering_error_count": len(run.cost.errors),
            "observed_api_cost_usd": str(run.cost.report.total_usd),
        },
        "provenance": asdict(provenance),
        "metrics": _serialize_metrics(run.trace, run=run),
        "requests": requests,
    }


def canonical_fingerprint(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _serialize_metrics(trace: EvaluationTrace, *, run: EvaluationRun) -> Dict[str, object]:
    evaluated_by_id = {document.document_id: document for document in trace.documents}
    documents = [
        _serialize_document(
            ordinal=document.ordinal,
            document_id=document.document_id,
            status=document.status,
            evaluation=evaluated_by_id.get(document.document_id),
        )
        for document in run.documents
    ]
    return {
        "aggregate": _counts(trace.metrics),
        "fields": {field: _counts(metrics) for field, metrics in sorted(trace.field_metrics.items())},
        "document_ids": [document.document_id for document in run.documents],
        "documents": documents,
    }


def _serialize_document(
    *,
    ordinal: int,
    document_id: str,
    status: str,
    evaluation: Optional[DocumentEvaluation],
) -> Dict[str, object]:
    metrics = evaluation.metrics if evaluation is not None else EntityMetrics()
    fields = (
        {
            field.field: _counts(field.metrics)
            for field in sorted(evaluation.fields, key=lambda item: item.field)
        }
        if evaluation is not None
        else {}
    )
    return {
        "ordinal": ordinal,
        "document_id": document_id,
        "status": status,
        "counts": _counts(metrics),
        "fields": fields,
    }


def _counts(metrics: EntityMetrics) -> Dict[str, int]:
    return {
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
    }


def discover_evidence_bank(directory: Path, *, arm: str) -> Tuple[Path, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise EvidenceValidationError(
            [_issue("bank_not_directory", "Experiment bank is not a directory.", arm=arm)]
        )
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    invalid = [path for path in entries if path.is_symlink() or not path.is_file()]
    invalid.extend(path for path in entries if path.is_file() and not path.name.endswith(".evidence.json"))
    if invalid:
        raise EvidenceValidationError(
            [
                _issue(
                    "unexpected_bank_entry",
                    "Experiment bank contains an unsupported entry.",
                    arm=arm,
                    expected="only regular .evidence.json files",
                )
            ]
        )
    if not entries:
        raise EvidenceValidationError([_issue("empty_bank", "Experiment bank is empty.", arm=arm)])
    if len(entries) > MAX_EVIDENCE_FILES:
        raise EvidenceValidationError(
            [
                _issue(
                    "too_many_evidence_files",
                    "Experiment bank contains too many evidence files.",
                    arm=arm,
                    expected=f"at most {MAX_EVIDENCE_FILES}",
                )
            ]
        )
    return tuple(entries)


def load_evidence_files(paths: Sequence[Path], *, arm: str) -> Tuple[EvidenceRunRecord, ...]:
    if not paths or len(paths) > MAX_EVIDENCE_FILES:
        raise EvidenceValidationError(
            [
                _issue(
                    "invalid_input_count",
                    "Evidence file count is outside the supported range.",
                    arm=arm,
                    expected=f"1 to {MAX_EVIDENCE_FILES}",
                )
            ]
        )
    records = []
    issues = []
    for index, path in enumerate(paths):
        try:
            records.append(_load_evidence_file(path, arm=arm, input_index=index))
        except EvidenceValidationError as error:
            issues.extend(error.issues)
    if issues:
        raise EvidenceValidationError(issues)
    _validate_unique_records(records, arm=arm)
    return tuple(sorted(records, key=lambda record: record.seed))


def _load_evidence_file(path: Path, *, arm: str, input_index: int) -> EvidenceRunRecord:
    payload = _read_bounded_json(path, arm=arm, input_index=input_index)
    try:
        root = _object(payload, "$")
        _expect(root.get("artifact_kind") == EVIDENCE_ARTIFACT_KIND, "artifact_kind", "$.artifact_kind")
        _expect(root.get("schema_version") == EVIDENCE_SCHEMA_VERSION, "schema_version", "$.schema_version")
        _expect(
            root.get("fingerprint_algorithm") == FINGERPRINT_ALGORITHM,
            "sha256 fingerprint algorithm",
            "$.fingerprint_algorithm",
        )
        evidence_fingerprint = _digest(root.get("evidence_fingerprint"), "$.evidence_fingerprint")
        fingerprint_payload = dict(root)
        fingerprint_payload.pop("evidence_fingerprint", None)
        _expect(
            canonical_fingerprint(fingerprint_payload) == evidence_fingerprint,
            "evidence_fingerprint",
            "$.evidence_fingerprint",
        )
        run = _object(root.get("run"), "$.run")
        _validate_complete_run(run)
        provenance = _object(root.get("provenance"), "$.provenance")
        _expect(provenance.get("promotion_capable") is True, "promotion_capable", "$.provenance")
        _expect(provenance.get("repository_available") is True, "repository_available", "$.provenance")
        for gate in (
            "repository_clean_start",
            "repository_clean_end",
            "solution_matches_head_start",
            "solution_matches_snapshot_end",
            "repository_commit_unchanged",
        ):
            _expect(provenance.get(gate) is True, "passed provenance gate", f"$.provenance.{gate}")
        _expect(
            provenance.get("invalidation_reasons") == [],
            "no provenance invalidation reasons",
            "$.provenance.invalidation_reasons",
        )
        metrics = _object(root.get("metrics"), "$.metrics")
        aggregate, fields, documents = _parse_metrics(metrics)
        requests = _object(root.get("requests"), "$.requests")
        receipts, plan_fingerprint, bank_fingerprint = _parse_requests(requests)
        return EvidenceRunRecord(
            run_id=_string(run.get("run_id"), "$.run.run_id"),
            seed=_integer(run.get("evaluation_seed"), "$.run.evaluation_seed"),
            evaluation_mode=_enum(
                run.get("evaluation_mode"),
                {"cache", "cache-fill", "fresh"},
                "$.run.evaluation_mode",
            ),
            repository_commit=_digest(
                provenance.get("repository_commit"), "$.provenance.repository_commit", length=40
            ),
            solution_snapshot_fingerprint=_digest(
                provenance.get("solution_snapshot_fingerprint"),
                "$.provenance.solution_snapshot_fingerprint",
            ),
            scoring_contract_fingerprint=_digest(
                provenance.get("scoring_contract_fingerprint"),
                "$.provenance.scoring_contract_fingerprint",
            ),
            dataset_fingerprint=_digest(
                provenance.get("dataset_fingerprint"),
                "$.provenance.dataset_fingerprint",
            ),
            runtime_fingerprint=_digest(
                provenance.get("runtime_fingerprint"),
                "$.provenance.runtime_fingerprint",
            ),
            aggregate=aggregate,
            fields=fields,
            documents=documents,
            receipts=receipts,
            request_plan_fingerprint=plan_fingerprint,
            response_bank_fingerprint=bank_fingerprint,
            evidence_fingerprint=evidence_fingerprint,
            observed_api_cost_usd=_decimal(
                run.get("observed_api_cost_usd"),
                "$.run.observed_api_cost_usd",
            ),
            duration_seconds=_finite_number(run.get("duration_seconds"), "$.run.duration_seconds"),
        )
    except _InvalidField as error:
        raise EvidenceValidationError(
            [
                _issue(
                    "invalid_evidence_field",
                    "Evidence contains an invalid or inconsistent field.",
                    arm=arm,
                    input_index=input_index,
                    json_path=error.json_path,
                    expected=error.expected,
                )
            ]
        ) from None


def _read_bounded_json(path: Path, *, arm: str, input_index: int) -> object:
    try:
        if path.is_symlink():
            raise OSError("symlink")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_EVIDENCE_BYTES:
                raise OSError("unsupported file")
            chunks = []
            remaining = MAX_EVIDENCE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65_536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            serialized = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if len(serialized) > MAX_EVIDENCE_BYTES or (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise OSError("file changed")
        payload = json.loads(
            serialized,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite")),
        )
        _validate_json_depth(payload, depth=0)
        return payload
    except (OSError, UnicodeError, ValueError, TypeError):
        raise EvidenceValidationError(
            [
                _issue(
                    "unreadable_evidence",
                    "Evidence file is unsafe, malformed, or too large.",
                    arm=arm,
                    input_index=input_index,
                    expected=f"regular JSON file no larger than {MAX_EVIDENCE_BYTES} bytes",
                )
            ]
        ) from None


def _validate_complete_run(run: Mapping[str, object]) -> None:
    _expect(run.get("dataset") == EXPECTED_DATASET, "dataset", "$.run.dataset")
    _expect(run.get("lifecycle_status") == "terminal", "terminal lifecycle", "$.run.lifecycle_status")
    _expect(run.get("result_status") == "complete", "complete result", "$.run.result_status")
    _expect(run.get("score_is_final") is True, "final score", "$.run.score_is_final")
    _expect(run.get("termination_category") == "none", "successful termination", "$.run.termination_category")
    _expect(run.get("metering_status") == "complete", "complete metering", "$.run.metering_status")
    _expect(run.get("metering_error_count") == 0, "zero metering errors", "$.run.metering_error_count")
    coverage = _object(run.get("coverage"), "$.run.coverage")
    expected = {
        "total": EXPECTED_DOCUMENT_COUNT,
        "completed": EXPECTED_DOCUMENT_COUNT,
        "failed": 0,
        "not_attempted": 0,
    }
    _expect(coverage == expected, "full 121-document coverage", "$.run.coverage")


def _parse_metrics(
    metrics: Mapping[str, object],
) -> tuple[Counts, Mapping[str, Counts], Tuple[EvidenceDocument, ...]]:
    aggregate = _parse_counts(metrics.get("aggregate"), "$.metrics.aggregate")
    _expect(
        6 * aggregate.true_positive + aggregate.false_positive + 5 * aggregate.false_negative > 0,
        "non-zero score denominator",
        "$.metrics.aggregate",
    )
    fields = _parse_count_map(metrics.get("fields"), "$.metrics.fields")
    document_ids = _list(metrics.get("document_ids"), "$.metrics.document_ids")
    serialized_documents = _list(metrics.get("documents"), "$.metrics.documents")
    _expect(
        len(document_ids) == EXPECTED_DOCUMENT_COUNT, "121 ordered document IDs", "$.metrics.document_ids"
    )
    _expect(
        len(serialized_documents) == EXPECTED_DOCUMENT_COUNT, "121 document metrics", "$.metrics.documents"
    )
    documents = []
    for ordinal, serialized in enumerate(serialized_documents):
        item = _object(serialized, f"$.metrics.documents[{ordinal}]")
        document_id = _string(item.get("document_id"), f"$.metrics.documents[{ordinal}].document_id")
        _expect(item.get("ordinal") == ordinal, "ordered ordinal", f"$.metrics.documents[{ordinal}].ordinal")
        _expect(
            item.get("status") == "completed", "completed document", f"$.metrics.documents[{ordinal}].status"
        )
        _expect(
            document_ids[ordinal] == document_id, "matching document ID", f"$.metrics.document_ids[{ordinal}]"
        )
        documents.append(
            EvidenceDocument(
                ordinal=ordinal,
                document_id=document_id,
                counts=_parse_counts(item.get("counts"), f"$.metrics.documents[{ordinal}].counts"),
                fields=_parse_count_map(item.get("fields"), f"$.metrics.documents[{ordinal}].fields"),
            )
        )
        _expect(
            _sum_counts(documents[-1].fields.values()) == documents[-1].counts,
            "document fields sum to document counts",
            f"$.metrics.documents[{ordinal}].fields",
        )
    _expect(
        len(set(cast(Sequence[str], document_ids))) == EXPECTED_DOCUMENT_COUNT,
        "unique document IDs",
        "$.metrics.document_ids",
    )
    _expect(
        _sum_counts(document.counts for document in documents) == aggregate,
        "aggregate equals document counts",
        "$.metrics.aggregate",
    )
    for field, counts in fields.items():
        document_counts = _sum_counts(document.fields.get(field, Counts(0, 0, 0)) for document in documents)
        _expect(document_counts == counts, "field aggregate equals document counts", "$.metrics.fields")
    _expect(_sum_counts(fields.values()) == aggregate, "fields sum to aggregate", "$.metrics.fields")
    return aggregate, fields, tuple(documents)


def _parse_requests(
    requests: Mapping[str, object],
) -> tuple[Tuple[EvidenceReceipt, ...], str, str]:
    _expect(requests.get("status") == "complete", "complete response bank", "$.requests.status")
    serialized_receipts = _list(requests.get("receipts"), "$.requests.receipts")
    receipts = []
    for index, serialized in enumerate(serialized_receipts):
        item = _object(serialized, f"$.requests.receipts[{index}]")
        receipts.append(
            EvidenceReceipt(
                document_ordinal=_integer(
                    item.get("document_ordinal"),
                    f"$.requests.receipts[{index}].document_ordinal",
                    minimum=-1,
                ),
                request_ordinal=_integer(
                    item.get("request_ordinal"), f"$.requests.receipts[{index}].request_ordinal"
                ),
                request_key=_digest(item.get("request_key"), f"$.requests.receipts[{index}].request_key"),
                response_content_sha256=_digest(
                    item.get("response_content_sha256"),
                    f"$.requests.receipts[{index}].response_content_sha256",
                ),
                replayed=_boolean(item.get("replayed"), f"$.requests.receipts[{index}].replayed"),
            )
        )
    expected_order = sorted(receipts, key=lambda item: (item.document_ordinal, item.request_ordinal))
    _expect(receipts == expected_order, "canonical receipt order", "$.requests.receipts")
    identities = [(item.document_ordinal, item.request_ordinal) for item in receipts]
    _expect(len(identities) == len(set(identities)), "unique receipt ordinals", "$.requests.receipts")
    _expect(
        all(item.document_ordinal < EXPECTED_DOCUMENT_COUNT for item in receipts),
        "document ordinal below 121",
        "$.requests.receipts",
    )
    _expect(
        not any(item.document_ordinal == -1 for item in receipts)
        or all(item.document_ordinal == -1 for item in receipts),
        "uniform aggregate or document receipt attribution",
        "$.requests.receipts",
    )
    ordinals_by_document: Dict[int, list[int]] = {}
    for receipt in receipts:
        ordinals_by_document.setdefault(receipt.document_ordinal, []).append(receipt.request_ordinal)
    _expect(
        all(ordinals == list(range(len(ordinals))) for ordinals in ordinals_by_document.values()),
        "contiguous request ordinals starting at zero within each document",
        "$.requests.receipts",
    )
    _expect(
        requests.get("receipt_count") == len(receipts), "matching receipt count", "$.requests.receipt_count"
    )
    serialized = [asdict(receipt) for receipt in receipts]
    plan = [
        {
            "document_ordinal": receipt.document_ordinal,
            "request_ordinal": receipt.request_ordinal,
            "request_key": receipt.request_key,
        }
        for receipt in receipts
    ]
    plan_fingerprint = _digest(
        requests.get("request_plan_fingerprint"), "$.requests.request_plan_fingerprint"
    )
    bank_fingerprint = _digest(
        requests.get("response_bank_fingerprint"), "$.requests.response_bank_fingerprint"
    )
    _expect(
        canonical_fingerprint(plan) == plan_fingerprint,
        "matching request-plan fingerprint",
        "$.requests.request_plan_fingerprint",
    )
    _expect(
        canonical_fingerprint(serialized) == bank_fingerprint,
        "matching response-bank fingerprint",
        "$.requests.response_bank_fingerprint",
    )
    _expect(
        requests.get("all_replayed") is (bool(receipts) and all(item.replayed for item in receipts)),
        "matching replay summary",
        "$.requests.all_replayed",
    )
    return tuple(receipts), plan_fingerprint, bank_fingerprint


def _validate_unique_records(records: Sequence[EvidenceRunRecord], *, arm: str) -> None:
    if len({record.run_id for record in records}) != len(records):
        raise EvidenceValidationError([_issue("duplicate_run_id", "Evidence repeats a run ID.", arm=arm)])
    if len({record.evidence_fingerprint for record in records}) != len(records):
        raise EvidenceValidationError(
            [_issue("duplicate_evidence", "Evidence repeats the same artifact.", arm=arm)]
        )
    if len({record.seed for record in records}) != len(records):
        raise EvidenceValidationError(
            [_issue("duplicate_seed", "Evidence repeats a seed within one arm.", arm=arm)]
        )
    if len({record.repository_commit for record in records}) != 1:
        raise EvidenceValidationError(
            [_issue("mixed_commits", "One arm contains multiple commits.", arm=arm)]
        )
    if len({record.solution_snapshot_fingerprint for record in records}) != 1:
        raise EvidenceValidationError(
            [_issue("mixed_solution_snapshots", "One arm contains multiple solution snapshots.", arm=arm)]
        )


class _InvalidField(ValueError):
    def __init__(self, json_path: str, expected: str) -> None:
        super().__init__(json_path)
        self.json_path = json_path
        self.expected = expected


def _issue(
    code: str,
    problem: str,
    *,
    arm: str = "",
    input_index: Optional[int] = None,
    json_path: str = "",
    expected: str = "",
) -> EvidenceIssue:
    return EvidenceIssue(
        code=code,
        problem=problem,
        cause="The supplied comparison evidence did not satisfy the locked evidence contract.",
        fix="Regenerate or reorganize the exact evaluation sidecars, then compare again.",
        docs_ref="research-runbook.md#paired-comparison-decisions",
        arm=arm,
        input_index=input_index,
        json_path=json_path,
        expected=expected,
    )


def _validate_json_depth(value: object, *, depth: int) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError("JSON nesting is too deep")
    if isinstance(value, dict):
        for item in value.values():
            _validate_json_depth(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_depth(item, depth=depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON number is non-finite")


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> Dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object repeats a key")
        result[key] = value
    return result


def _object(value: object, json_path: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise _InvalidField(json_path, "object")
    return cast(Mapping[str, object], value)


def _list(value: object, json_path: str) -> list[object]:
    if not isinstance(value, list):
        raise _InvalidField(json_path, "array")
    return value


def _string(value: object, json_path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _InvalidField(json_path, "non-empty string")
    return value


def _enum(value: object, allowed: set[str], json_path: str) -> str:
    parsed = _string(value, json_path)
    if parsed not in allowed:
        raise _InvalidField(json_path, "supported enum value")
    return parsed


def _integer(value: object, json_path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= MAX_INTEGER:
        raise _InvalidField(json_path, f"integer from {minimum} through {MAX_INTEGER}")
    return value


def _boolean(value: object, json_path: str) -> bool:
    if not isinstance(value, bool):
        raise _InvalidField(json_path, "boolean")
    return value


def _finite_number(value: object, json_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidField(json_path, "finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise _InvalidField(json_path, "finite non-negative number")
    return parsed


def _decimal(value: object, json_path: str) -> Decimal:
    if not isinstance(value, str):
        raise _InvalidField(json_path, "finite non-negative decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise _InvalidField(json_path, "finite non-negative decimal string") from None
    if not parsed.is_finite() or parsed < 0:
        raise _InvalidField(json_path, "finite non-negative decimal string")
    return parsed


def _digest(value: object, json_path: str, *, length: int = 64) -> str:
    parsed = _string(value, json_path)
    if len(parsed) != length or any(character not in "0123456789abcdef" for character in parsed):
        raise _InvalidField(json_path, f"{length}-character lowercase hexadecimal digest")
    return parsed


def _parse_counts(value: object, json_path: str) -> Counts:
    serialized = _object(value, json_path)
    expected_keys = {"true_positive", "false_positive", "false_negative"}
    _expect(set(serialized) == expected_keys, "TP/FP/FN count object", json_path)
    return Counts(
        true_positive=_integer(serialized.get("true_positive"), f"{json_path}.true_positive"),
        false_positive=_integer(serialized.get("false_positive"), f"{json_path}.false_positive"),
        false_negative=_integer(serialized.get("false_negative"), f"{json_path}.false_negative"),
    )


def _parse_count_map(value: object, json_path: str) -> Mapping[str, Counts]:
    serialized = _object(value, json_path)
    result = {}
    for index, (name, counts) in enumerate(serialized.items()):
        parsed_name = _string(name, f"{json_path}.<field-{index}>")
        result[parsed_name] = _parse_counts(counts, f"{json_path}.<counts-{index}>")
    return result


def _sum_counts(values: Iterable[Counts]) -> Counts:
    return sum(values, Counts(0, 0, 0))


def _expect(condition: bool, expected: str, json_path: str) -> None:
    if not condition:
        raise _InvalidField(json_path, expected)
