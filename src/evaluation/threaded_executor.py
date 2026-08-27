import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.cost_metering.accounting import CostReport, MeteringOutcome
from src.cost_metering.proxy import MeteringProxy
from src.evaluation.execution import AdmissionStrategyValue
from src.evaluation.run_results import DocumentExecution, DocumentStatus, UsageAttributionStatus
from src.evaluation.worker_frames import (
    MAX_AGGREGATE_RESULT_BYTES,
    PARENT_PID_ENVIRONMENT,
    PROTOCOL_VERSION,
    RESULT_FD_ENVIRONMENT,
    FrameType,
    ThreadedTask,
)

PROCESS_GRACE_SECONDS = 0.5
CheckpointCallback = Callable[[Sequence[DocumentExecution], MeteringOutcome], None]


def run_threaded_solution_documents(
    texts: Mapping[str, str],
    *,
    module: str,
    meter: MeteringProxy,
    deadline: float,
    environment: Mapping[str, str],
    source_tokens: Mapping[str, int],
    on_checkpoint: CheckpointCallback,
    max_concurrent_documents: int,
    admission_strategy: AdmissionStrategyValue,
    run_id: str,
) -> Tuple[Tuple[DocumentExecution, ...], str]:
    tasks = _make_tasks(texts, source_tokens=source_tokens)
    try:
        process, result_fd, logs = _start_worker(
            module=module,
            environment=environment,
            run_id=run_id,
            max_concurrent_documents=max_concurrent_documents,
            admission_strategy=admission_strategy,
        )
    except OSError:
        states = {task.ordinal: "queued" for task in tasks}
        return (
            _threaded_ledger(tasks, states=states, documents={}, termination="worker_protocol_error"),
            "worker_protocol_error",
        )
    states = {task.ordinal: "queued" for task in tasks}
    documents: Dict[int, DocumentExecution] = {}
    sender_errors: List[str] = []
    sender = _start_sender(process, tasks=tasks, run_id=run_id, errors=sender_errors)
    try:
        termination, worker_done = _drain_frames(
            result_fd,
            run_id=run_id,
            tasks=tasks,
            states=states,
            documents=documents,
            meter=meter,
            deadline=deadline,
            on_checkpoint=on_checkpoint,
        )
    except KeyboardInterrupt:
        termination, worker_done = "interrupted", False
    _stop_worker(process, deadline=deadline)
    sender.join(timeout=PROCESS_GRACE_SECONDS)
    _close_worker_resources(result_fd=result_fd, logs=logs)
    termination = _final_termination(
        termination,
        worker_done=worker_done,
        sender_errors=sender_errors,
    )
    ledger = _threaded_ledger(tasks, states=states, documents=documents, termination=termination)
    if termination == "none" and any(document.status != DocumentStatus.COMPLETED for document in ledger):
        termination = "document_failures"
    return ledger, termination


def _make_tasks(texts: Mapping[str, str], *, source_tokens: Mapping[str, int]) -> Tuple[ThreadedTask, ...]:
    return tuple(
        ThreadedTask(
            ordinal=ordinal,
            document_id=document_id,
            text=text,
            source_tokens=source_tokens[document_id],
        )
        for ordinal, (document_id, text) in enumerate(texts.items())
    )


def _start_worker(
    *,
    module: str,
    environment: Mapping[str, str],
    run_id: str,
    max_concurrent_documents: int,
    admission_strategy: AdmissionStrategyValue,
):
    read_fd, write_fd = os.pipe()
    child_environment = _child_environment(environment, result_fd=write_fd)
    logs = tempfile.TemporaryFile(mode="w+")
    try:
        process = subprocess.Popen(
            _worker_command(
                module=module,
                run_id=run_id,
                max_concurrent_documents=max_concurrent_documents,
                admission_strategy=admission_strategy,
            ),
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


def _child_environment(environment: Mapping[str, str], *, result_fd: int) -> Dict[str, str]:
    child_environment = dict(environment)
    child_environment[RESULT_FD_ENVIRONMENT] = str(result_fd)
    child_environment[PARENT_PID_ENVIRONMENT] = str(os.getpid())
    return child_environment


def _worker_command(
    *,
    module: str,
    run_id: str,
    max_concurrent_documents: int,
    admission_strategy: AdmissionStrategyValue,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "src.evaluation.cli",
        "--worker",
        "--threaded-worker",
        "--module",
        module,
        "--worker-run-id",
        run_id,
        "--max-concurrent-documents",
        str(max_concurrent_documents),
        "--admission-strategy",
        admission_strategy,
    ]


def _start_sender(
    process: subprocess.Popen,
    *,
    tasks: Sequence[ThreadedTask],
    run_id: str,
    errors: List[str],
) -> threading.Thread:
    sender = threading.Thread(
        target=_send_tasks,
        args=(process, tasks),
        kwargs={"run_id": run_id, "errors": errors},
        daemon=True,
    )
    sender.start()
    return sender


def _send_tasks(
    process: subprocess.Popen,
    tasks: Sequence[ThreadedTask],
    *,
    run_id: str,
    errors: List[str],
) -> None:
    if process.stdin is None:
        errors.append("worker stdin is unavailable")
        return
    try:
        for task in tasks:
            process.stdin.write(_serialize_task(task, run_id=run_id) + "\n")
            process.stdin.flush()
    except (BrokenPipeError, OSError):
        errors.append("worker input channel closed")
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass


def _serialize_task(task: ThreadedTask, *, run_id: str) -> str:
    payload = {
        "run_id": run_id,
        "ordinal": task.ordinal,
        "document_id": task.document_id,
        "text": task.text,
    }
    return json.dumps(payload, separators=(",", ":"))


def _drain_frames(
    result_fd: int,
    *,
    run_id: str,
    tasks: Sequence[ThreadedTask],
    states: Dict[int, str],
    documents: Dict[int, DocumentExecution],
    meter: MeteringProxy,
    deadline: float,
    on_checkpoint: CheckpointCallback,
) -> Tuple[str, bool]:
    from src.evaluation.worker import WorkerProtocolError, _parse_document_record, _read_record

    task_by_ordinal = {task.ordinal: task for task in tasks}
    buffer = b""
    aggregate_bytes = 0
    worker_done = False
    try:
        while not worker_done:
            frame, buffer = _read_record(result_fd, buffer=buffer, deadline=deadline)
            aggregate_bytes = _updated_aggregate_bytes(aggregate_bytes, frame=frame)
            frame_type, task = _validate_frame(
                frame,
                run_id=run_id,
                task_by_ordinal=task_by_ordinal,
                states=states,
            )
            if frame_type == FrameType.SETTLED and task is not None:
                document = _parse_document_record(
                    frame["result"],
                    ordinal=task.ordinal,
                    expected_document_id=task.document_id,
                    source_tokens=task.source_tokens,
                    usage=CostReport(()),
                    usage_complete=True,
                    usage_attribution_status=UsageAttributionStatus.UNAVAILABLE,
                    latency_seconds=float(frame["result"].get("latency_seconds", 0.0)),
                )
                documents[task.ordinal] = document
                on_checkpoint(tuple(documents[index] for index in sorted(documents)), meter.progress())
            worker_done = frame_type == FrameType.WORKER_DONE
    except TimeoutError:
        return "dataset_deadline", worker_done
    except (EOFError, WorkerProtocolError, KeyError, TypeError, ValueError):
        return "worker_protocol_error", worker_done
    return "none", worker_done


def _updated_aggregate_bytes(current: int, *, frame: Mapping[str, object]) -> int:
    from src.evaluation.worker import WorkerProtocolError

    updated = current + len(json.dumps(frame, separators=(",", ":")).encode())
    if updated > MAX_AGGREGATE_RESULT_BYTES:
        raise WorkerProtocolError(
            f"worker results exceeded aggregate limit {MAX_AGGREGATE_RESULT_BYTES} bytes"
        )
    return updated


def _validate_frame(
    frame: Mapping[str, object],
    *,
    run_id: str,
    task_by_ordinal: Mapping[int, ThreadedTask],
    states: Dict[int, str],
) -> Tuple[str, Optional[ThreadedTask]]:
    from src.evaluation.worker import WorkerProtocolError

    if frame.get("protocol_version") != PROTOCOL_VERSION or frame.get("run_id") != run_id:
        raise WorkerProtocolError("worker frame used the wrong protocol or run ID")
    frame_type = str(frame.get("frame_type"))
    if frame_type not in FrameType.all():
        raise WorkerProtocolError(f"worker returned unsupported frame type {frame_type!r}")
    if frame_type == FrameType.WORKER_DONE:
        return frame_type, None
    task = _frame_task(frame, task_by_ordinal=task_by_ordinal)
    expected = {
        FrameType.ACCEPTED: "queued",
        FrameType.STARTED: "accepted",
        FrameType.SETTLED: "started",
    }[frame_type]
    if states[task.ordinal] != expected:
        raise WorkerProtocolError(
            f"illegal worker transition {states[task.ordinal]!r} -> {frame_type!r} "
            f"for ordinal {task.ordinal}"
        )
    states[task.ordinal] = frame_type
    return frame_type, task


def _frame_task(frame: Mapping[str, object], *, task_by_ordinal: Mapping[int, ThreadedTask]) -> ThreadedTask:
    from src.evaluation.worker import WorkerProtocolError

    ordinal = frame.get("ordinal")
    if not isinstance(ordinal, int) or ordinal not in task_by_ordinal:
        raise WorkerProtocolError(f"worker returned unknown ordinal {ordinal!r}")
    task = task_by_ordinal[ordinal]
    if frame.get("document_id") != task.document_id:
        raise WorkerProtocolError(f"worker returned wrong document for ordinal {ordinal}")
    return task


def _final_termination(termination: str, *, worker_done: bool, sender_errors: Sequence[str]) -> str:
    if termination != "none":
        return termination
    if sender_errors or not worker_done:
        return "worker_protocol_error"
    return termination


def _threaded_ledger(
    tasks: Sequence[ThreadedTask],
    *,
    states: Mapping[int, str],
    documents: Mapping[int, DocumentExecution],
    termination: str,
) -> Tuple[DocumentExecution, ...]:
    return tuple(
        documents.get(task.ordinal)
        or _unfinished_document(task, state=states[task.ordinal], termination=termination)
        for task in tasks
    )


def _unfinished_document(task: ThreadedTask, *, state: str, termination: str) -> DocumentExecution:
    if state in ("queued", FrameType.ACCEPTED):
        return DocumentExecution(
            ordinal=task.ordinal,
            document_id=task.document_id,
            status=DocumentStatus.NOT_ATTEMPTED,
            source_tokens=task.source_tokens,
            usage_attribution_status=UsageAttributionStatus.UNAVAILABLE,
        )
    failure_category = termination if termination != "none" else "worker_protocol_error"
    return DocumentExecution(
        ordinal=task.ordinal,
        document_id=task.document_id,
        status=DocumentStatus.FAILED,
        source_tokens=task.source_tokens,
        usage_attribution_status=UsageAttributionStatus.UNAVAILABLE,
        failure_category=failure_category,
        error_message="threaded document execution did not settle",
        retryable=True,
    )


def _stop_worker(process: subprocess.Popen, *, deadline: float) -> None:
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=min(PROCESS_GRACE_SECONDS, max(deadline - time.monotonic(), 0.01)))
        return
    except subprocess.TimeoutExpired:
        pass
    _terminate_process_group(process)


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=PROCESS_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=PROCESS_GRACE_SECONDS)


def _close_worker_resources(*, result_fd: int, logs) -> None:
    os.close(result_fd)
    logs.close()
