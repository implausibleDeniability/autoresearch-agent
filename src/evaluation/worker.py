import importlib
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

from src.cost_metering.accounting import CostReport, CostStatus, MeteringOutcome
from src.cost_metering.proxy import MeteringProxy
from src.evaluation.execution import (
    AdmissionStrategy,
    AdmissionStrategyValue,
    DEFAULT_MAX_CONCURRENT_DOCUMENTS,
    ExecutionMode,
    ExecutionModeValue,
)
from src.evaluation.models import PIIItem
from src.evaluation.run_results import DocumentExecution, DocumentStatus, UsageAttributionStatus

RESULT_FD_ENVIRONMENT = "EVALUATION_RESULT_FD"
MAX_RESULT_BYTES = 10_000_000
PROCESS_GRACE_SECONDS = 0.5
CheckpointCallback = Callable[[Sequence[DocumentExecution], MeteringOutcome], None]


class WorkerProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class DocumentTask:
    ordinal: int
    document_id: str
    text: str
    source_tokens: int
    run_token: str


@dataclass(frozen=True)
class WorkerResult:
    task: DocumentTask
    latency_seconds: float
    record: Optional[Dict[str, object]] = None
    failure_category: str = ""


def extract_documents(
    texts: Mapping[str, str],
    *,
    module_name: str,
    max_concurrent_documents: int,
) -> Dict[str, List[PIIItem]]:
    extract_pii: Callable[[str], List[PIIItem]] = importlib.import_module(module_name).extract_pii
    with ThreadPoolExecutor(max_workers=max_concurrent_documents) as executor:
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
    max_concurrent_documents: int = DEFAULT_MAX_CONCURRENT_DOCUMENTS,
    execution_mode: ExecutionModeValue = ExecutionMode.ISOLATED,
    admission_strategy: AdmissionStrategyValue = AdmissionStrategy.RAMP,
    run_id: str = "isolated",
) -> Tuple[Tuple[DocumentExecution, ...], str]:
    if execution_mode == ExecutionMode.THREADED:
        from src.evaluation.threaded_executor import run_threaded_solution_documents

        return run_threaded_solution_documents(
            texts,
            module=module,
            meter=meter,
            deadline=deadline,
            environment=environment,
            source_tokens=source_tokens,
            on_checkpoint=on_checkpoint,
            max_concurrent_documents=max_concurrent_documents,
            admission_strategy=admission_strategy,
            run_id=run_id,
        )
    tasks = _document_tasks(texts, source_tokens=source_tokens, meter=meter)
    documents: Dict[int, DocumentExecution] = {}
    termination_category = "none"
    with ThreadPoolExecutor(max_workers=max_concurrent_documents) as executor:
        futures = _fill_worker_slots(
            {},
            tasks=tasks,
            executor=executor,
            module=module,
            environment=environment,
            deadline=deadline,
            max_concurrent_documents=max_concurrent_documents,
        )
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                futures.pop(future)
                document = _make_document_execution(result, meter=meter, deadline=deadline)
                documents[document.ordinal] = document
                progress = meter.progress()
                termination_category = _updated_termination_category(
                    termination_category, result=result, progress=progress
                )
            on_checkpoint(tuple(documents[index] for index in sorted(documents)), progress)
            if termination_category == "none":
                _fill_worker_slots(
                    futures,
                    tasks=tasks,
                    executor=executor,
                    module=module,
                    environment=environment,
                    deadline=deadline,
                    max_concurrent_documents=max_concurrent_documents,
                )
    return (
        _complete_document_ledger(tuple(documents.values()), texts=texts, source_tokens=source_tokens),
        termination_category,
    )


def _document_tasks(
    texts: Mapping[str, str], *, source_tokens: Mapping[str, int], meter: MeteringProxy
) -> Iterator[DocumentTask]:
    for ordinal, (document_id, text) in enumerate(texts.items()):
        yield DocumentTask(
            ordinal=ordinal,
            document_id=document_id,
            text=text,
            source_tokens=source_tokens[document_id],
            run_token=meter.issue_token(document_ordinal=ordinal),
        )


def _fill_worker_slots(
    futures: Dict[Future, DocumentTask],
    *,
    tasks: Iterator[DocumentTask],
    executor: ThreadPoolExecutor,
    module: str,
    environment: Mapping[str, str],
    deadline: float,
    max_concurrent_documents: int = DEFAULT_MAX_CONCURRENT_DOCUMENTS,
) -> Dict[Future, DocumentTask]:
    while len(futures) < max_concurrent_documents:
        task = next(tasks, None)
        if task is None:
            break
        future = executor.submit(
            _run_document_task,
            task,
            module=module,
            environment=environment,
            deadline=deadline,
        )
        futures[future] = task
    return futures


def _run_document_task(
    task: DocumentTask, *, module: str, environment: Mapping[str, str], deadline: float
) -> WorkerResult:
    started_at = time.monotonic()
    try:
        process, result_fd, logs = _start_worker(
            module=module, environment=_worker_environment(environment, run_token=task.run_token)
        )
    except OSError:
        return WorkerResult(
            task=task,
            failure_category="worker_protocol_error",
            latency_seconds=time.monotonic() - started_at,
        )
    try:
        _send_document(process, ordinal=task.ordinal, document_id=task.document_id, text=task.text)
        record, _ = _read_record(result_fd, buffer=b"", deadline=deadline)
        return WorkerResult(task=task, record=record, latency_seconds=time.monotonic() - started_at)
    except TimeoutError:
        return WorkerResult(
            task=task,
            failure_category="dataset_deadline",
            latency_seconds=time.monotonic() - started_at,
        )
    except (BrokenPipeError, EOFError, WorkerProtocolError):
        return WorkerResult(
            task=task,
            failure_category="worker_protocol_error",
            latency_seconds=time.monotonic() - started_at,
        )
    finally:
        _stop_worker(process, deadline=deadline)
        os.close(result_fd)
        logs.close()


def _worker_environment(environment: Mapping[str, str], *, run_token: str) -> Dict[str, str]:
    child_environment = dict(environment)
    child_environment["OPENAI_API_KEY"] = run_token
    return child_environment


def _make_document_execution(
    result: WorkerResult, *, meter: MeteringProxy, deadline: float
) -> DocumentExecution:
    task = result.task
    usage = meter.token_outcome(task.run_token, timeout=max(deadline - time.monotonic(), 0.0))
    if result.record is not None:
        return _parse_document_record(
            result.record,
            ordinal=task.ordinal,
            expected_document_id=task.document_id,
            source_tokens=task.source_tokens,
            usage=usage.report,
            usage_complete=usage.status == CostStatus.COMPLETE,
            latency_seconds=result.latency_seconds,
        )
    return DocumentExecution(
        ordinal=task.ordinal,
        document_id=task.document_id,
        status=DocumentStatus.FAILED,
        source_tokens=task.source_tokens,
        usage=usage.report,
        usage_complete=usage.status == CostStatus.COMPLETE,
        latency_seconds=result.latency_seconds,
        failure_category=result.failure_category,
        error_message="document execution did not complete",
        retryable=True,
    )


def _updated_termination_category(current: str, *, result: WorkerResult, progress: MeteringOutcome) -> str:
    if current != "none":
        return current
    if result.failure_category:
        return result.failure_category
    if progress.status == CostStatus.INCOMPLETE:
        return _meter_termination_category(progress)
    return current


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
    try:
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
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        logs.close()
        raise
    os.close(write_fd)
    return process, read_fd, logs


def _send_document(process: subprocess.Popen, *, ordinal: int, document_id: str, text: str) -> None:
    if process.stdin is None:
        raise BrokenPipeError("worker stdin is unavailable")
    process.stdin.write(json.dumps({"ordinal": ordinal, "document_id": document_id, "text": text}) + "\n")
    process.stdin.flush()


def _read_record(result_fd: int, *, buffer: bytes, deadline: float) -> Tuple[Dict[str, object], bytes]:
    if len(buffer) > MAX_RESULT_BYTES:
        raise WorkerProtocolError(f"worker result exceeded {MAX_RESULT_BYTES} bytes")
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
    usage_attribution_status: str = UsageAttributionStatus.EXACT,
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
            usage_attribution_status=usage_attribution_status,
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
        usage_attribution_status=usage_attribution_status,
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


def _complete_document_ledger(
    documents: Sequence[DocumentExecution],
    *,
    texts: Mapping[str, str],
    source_tokens: Mapping[str, int],
) -> Tuple[DocumentExecution, ...]:
    completed = {document.ordinal: document for document in documents}
    return tuple(
        completed.get(
            ordinal,
            DocumentExecution(
                ordinal=ordinal,
                document_id=document_id,
                status=DocumentStatus.NOT_ATTEMPTED,
                source_tokens=source_tokens[document_id],
            ),
        )
        for ordinal, document_id in enumerate(texts)
    )


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
