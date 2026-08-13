import importlib
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, cast

from src.cost_metering.accounting import CostReport, CostStatus, MeteringOutcome
from src.cost_metering.proxy import MeteringProxy
from src.evaluation.models import PIIItem
from src.evaluation.run_results import DocumentExecution, DocumentStatus

RESULT_FD_ENVIRONMENT = "EVALUATION_RESULT_FD"
MAX_RESULT_BYTES = 10_000_000
PROCESS_GRACE_SECONDS = 0.5
CheckpointCallback = Callable[[Sequence[DocumentExecution], MeteringOutcome], None]
MAX_CONCURRENT_DOCUMENTS = 8


class WorkerProtocolError(RuntimeError):
    pass


def extract_documents(
    texts: Mapping[str, str],
    *,
    module_name: str,
) -> Dict[str, List[PIIItem]]:
    extract_pii: Callable[[str], List[PIIItem]] = importlib.import_module(module_name).extract_pii
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOCUMENTS) as executor:
        predictions = executor.map(extract_pii, texts.values())
        return dict(zip(texts, predictions))


def run_worker(module_name: str) -> int:
    result_fd = int(os.environ[RESULT_FD_ENVIRONMENT])
    extract_pii = importlib.import_module(module_name).extract_pii
    with os.fdopen(result_fd, "w") as result_file:
        for line in sys.stdin:
            request = json.loads(line)
            record = _execute_document(request, extract_pii=extract_pii)
            result_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            result_file.flush()
    return 0


def run_solution_documents(
    texts: Mapping[str, str],
    *,
    module: str,
    meter: MeteringProxy,
    deadline: float,
    environment: Mapping[str, str],
    source_tokens: Mapping[str, int],
    on_checkpoint: CheckpointCallback,
) -> Tuple[Tuple[DocumentExecution, ...], str]:
    process, result_fd, logs = _start_worker(module=module, environment=environment)
    documents: List[DocumentExecution] = []
    usage_count = 0
    buffer = b""
    termination_category = "none"
    try:
        for ordinal, (document_id, text) in enumerate(texts.items()):
            started_at = time.monotonic()
            _send_document(process, ordinal=ordinal, document_id=document_id, text=text)
            record, buffer = _read_record(result_fd, buffer=buffer, deadline=deadline)
            checkpoint = meter.checkpoint(timeout=_remaining_seconds(deadline))
            usage = CostReport(checkpoint.report.usages[usage_count:])
            usage_count = len(checkpoint.report.usages)
            document = _parse_document_record(
                record,
                ordinal=ordinal,
                expected_document_id=document_id,
                source_tokens=source_tokens[document_id],
                usage=usage,
                usage_complete=checkpoint.status == CostStatus.COMPLETE,
                latency_seconds=time.monotonic() - started_at,
            )
            documents.append(document)
            on_checkpoint(tuple(documents), checkpoint)
            if checkpoint.status == CostStatus.INCOMPLETE:
                termination_category = _meter_termination_category(checkpoint)
                break
    except TimeoutError:
        termination_category = "dataset_deadline"
        _append_current_failure(
            documents, texts=texts, source_tokens=source_tokens, category=termination_category
        )
    except (BrokenPipeError, EOFError, WorkerProtocolError):
        termination_category = "worker_protocol_error"
        _append_current_failure(
            documents, texts=texts, source_tokens=source_tokens, category=termination_category
        )
    finally:
        _stop_worker(process, deadline=deadline)
        os.close(result_fd)
        logs.close()
    return (
        _complete_document_ledger(documents, texts=texts, source_tokens=source_tokens),
        termination_category,
    )


def _execute_document(request: Mapping[str, object], *, extract_pii: Callable) -> Dict[str, object]:
    ordinal = int(request["ordinal"])
    document_id = str(request["document_id"])
    text = str(request["text"])
    try:
        predictions = extract_pii(text)
        return {
            "ordinal": ordinal,
            "document_id": document_id,
            "status": DocumentStatus.COMPLETED,
            "predictions": [asdict(person) for person in predictions],
        }
    except Exception as error:
        return {
            "ordinal": ordinal,
            "document_id": document_id,
            "status": DocumentStatus.FAILED,
            "failure_category": "solution_error",
            "error_message": f"{type(error).__name__}: extract_pii failed",
            "retryable": False,
        }


def _start_worker(*, module: str, environment: Mapping[str, str]):
    read_fd, write_fd = os.pipe()
    child_environment = dict(environment)
    child_environment[RESULT_FD_ENVIRONMENT] = str(write_fd)
    logs = tempfile.TemporaryFile(mode="w+")
    process = subprocess.Popen(
        [sys.executable, "-m", "src.evaluation.cli", "--worker", "--module", module],
        stdin=subprocess.PIPE,
        stdout=logs,
        stderr=logs,
        text=True,
        env=child_environment,
        pass_fds=(write_fd,),
        start_new_session=True,
    )
    os.close(write_fd)
    return process, read_fd, logs


def _send_document(process: subprocess.Popen, *, ordinal: int, document_id: str, text: str) -> None:
    if process.stdin is None:
        raise BrokenPipeError("worker stdin is unavailable")
    process.stdin.write(json.dumps({"ordinal": ordinal, "document_id": document_id, "text": text}) + "\n")
    process.stdin.flush()


def _read_record(result_fd: int, *, buffer: bytes, deadline: float) -> Tuple[Dict[str, object], bytes]:
    while b"\n" not in buffer:
        readable, _, _ = select.select([result_fd], [], [], _remaining_seconds(deadline))
        if not readable:
            raise TimeoutError("dataset deadline expired while waiting for worker result")
        chunk = os.read(result_fd, 65_536)
        if not chunk:
            raise EOFError("worker result channel closed before a complete record")
        buffer += chunk
        if len(buffer) > MAX_RESULT_BYTES:
            raise WorkerProtocolError(f"worker result exceeded {MAX_RESULT_BYTES} bytes")
    line, remainder = buffer.split(b"\n", 1)
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise WorkerProtocolError("worker emitted invalid JSON") from error
    if not isinstance(record, dict):
        raise WorkerProtocolError("worker record must be a JSON object")
    return cast(Dict[str, object], record), remainder


def _parse_document_record(
    record: Mapping[str, object],
    *,
    ordinal: int,
    expected_document_id: str,
    source_tokens: int,
    usage: CostReport,
    usage_complete: bool,
    latency_seconds: float,
) -> DocumentExecution:
    if record.get("ordinal") != ordinal:
        raise WorkerProtocolError(f"worker returned sequence {record.get('ordinal')!r}; expected {ordinal!r}")
    if record.get("document_id") != expected_document_id:
        raise WorkerProtocolError(
            f"worker returned document {record.get('document_id')!r}; expected {expected_document_id!r}"
        )
    status = record.get("status")
    if status == DocumentStatus.COMPLETED:
        predictions = _deserialize_predictions(record.get("predictions"))
        return DocumentExecution(
            ordinal=ordinal,
            document_id=expected_document_id,
            status=DocumentStatus.COMPLETED,
            source_tokens=source_tokens,
            usage=usage,
            usage_complete=usage_complete,
            latency_seconds=latency_seconds,
            predictions=predictions,
        )
    if status != DocumentStatus.FAILED:
        raise WorkerProtocolError(f"worker returned unsupported document status {status!r}")
    return DocumentExecution(
        ordinal=ordinal,
        document_id=expected_document_id,
        status=DocumentStatus.FAILED,
        source_tokens=source_tokens,
        usage=usage,
        usage_complete=usage_complete,
        latency_seconds=latency_seconds,
        failure_category=str(record.get("failure_category", "solution_error")),
        error_message=str(record.get("error_message", "solution failed"))[:500],
        retryable=record.get("retryable") is True,
    )


def _deserialize_predictions(serialized: object) -> Tuple[PIIItem, ...]:
    if not isinstance(serialized, list):
        raise WorkerProtocolError("worker predictions must be a list")
    try:
        return tuple(
            PIIItem(**{field: tuple(values) for field, values in person.items()}) for person in serialized
        )
    except (AttributeError, TypeError) as error:
        raise WorkerProtocolError("worker predictions do not match the PII schema") from error


def _append_current_failure(
    documents: List[DocumentExecution],
    *,
    texts: Mapping[str, str],
    source_tokens: Mapping[str, int],
    category: str,
) -> None:
    if len(documents) >= len(texts):
        return
    document_id = tuple(texts)[len(documents)]
    documents.append(
        DocumentExecution(
            ordinal=len(documents),
            document_id=document_id,
            status=DocumentStatus.FAILED,
            source_tokens=source_tokens[document_id],
            failure_category=category,
            error_message="document execution did not complete",
            retryable=True,
        )
    )


def _complete_document_ledger(
    documents: Sequence[DocumentExecution],
    *,
    texts: Mapping[str, str],
    source_tokens: Mapping[str, int],
) -> Tuple[DocumentExecution, ...]:
    completed = list(documents)
    for ordinal, document_id in enumerate(tuple(texts)[len(completed) :], start=len(completed)):
        completed.append(
            DocumentExecution(
                ordinal=ordinal,
                document_id=document_id,
                status=DocumentStatus.NOT_ATTEMPTED,
                source_tokens=source_tokens[document_id],
            )
        )
    return tuple(completed)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("dataset deadline expired")
    return remaining


def _meter_termination_category(outcome: MeteringOutcome) -> str:
    if any("exceeded run limit" in error for error in outcome.errors):
        return "spending_limit"
    return "metering_incomplete"


def _stop_worker(process: subprocess.Popen, *, deadline: float) -> None:
    if process.stdin is not None:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=min(PROCESS_GRACE_SECONDS, max(deadline - time.monotonic(), 0.01)))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=PROCESS_GRACE_SECONDS)
