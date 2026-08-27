import importlib
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Dict, Iterator, Sequence, Tuple

from src.evaluation.execution import AdmissionStrategy, AdmissionStrategyValue, RAMP_STAGES
from src.evaluation.run_results import DocumentStatus
from src.evaluation.worker_frames import (
    PARENT_PID_ENVIRONMENT,
    RESULT_FD_ENVIRONMENT,
    ChildTask,
    FrameType,
    FrameWriter,
)


def run_threaded_worker(
    module_name: str,
    *,
    run_id: str,
    max_concurrent_documents: int,
    admission_strategy: AdmissionStrategyValue,
) -> int:
    result_fd = int(os.environ[RESULT_FD_ENVIRONMENT])
    extract_pii = importlib.import_module(module_name).extract_pii
    _start_parent_watchdog()
    with os.fdopen(result_fd, "w") as result_file:
        writer = FrameWriter(result_file, run_id=run_id)
        tasks = _read_tasks(run_id=run_id, writer=writer)
        settled, stopped = _execute_tasks(
            tasks,
            extract_pii=extract_pii,
            writer=writer,
            max_concurrent_documents=max_concurrent_documents,
            admission_strategy=admission_strategy,
        )
        writer.write(FrameType.WORKER_DONE, settled=settled, admission_stopped=stopped)
    return 0


def _read_tasks(*, run_id: str, writer: FrameWriter) -> Tuple[ChildTask, ...]:
    tasks = []
    ordinals = set()
    for line in sys.stdin:
        task = _parse_task(line, run_id=run_id)
        if task.ordinal in ordinals:
            raise RuntimeError(f"worker received duplicate ordinal {task.ordinal}")
        ordinals.add(task.ordinal)
        tasks.append(task)
        writer.write(FrameType.ACCEPTED, ordinal=task.ordinal, document_id=task.document_id)
    return tuple(tasks)


def _parse_task(line: str, *, run_id: str) -> ChildTask:
    payload = json.loads(line)
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise RuntimeError("worker received a task for the wrong run")
    return ChildTask(
        run_id=run_id,
        ordinal=int(payload["ordinal"]),
        document_id=str(payload["document_id"]),
        text=str(payload["text"]),
    )


def _execute_tasks(
    tasks: Sequence[ChildTask],
    *,
    extract_pii: Callable,
    writer: FrameWriter,
    max_concurrent_documents: int,
    admission_strategy: AdmissionStrategyValue,
) -> Tuple[int, bool]:
    pending = iter(tasks)
    active: Dict[Future, ChildTask] = {}
    limit = _initial_limit(max_concurrent_documents, admission_strategy=admission_strategy)
    settled = 0
    stopped = False
    with ThreadPoolExecutor(max_workers=max_concurrent_documents) as executor:
        _fill_slots(
            active, pending=pending, executor=executor, limit=limit, extract_pii=extract_pii, writer=writer
        )
        while active:
            return_when = (
                FIRST_COMPLETED if admission_strategy == AdmissionStrategy.IMMEDIATE else ALL_COMPLETED
            )
            completed, _ = wait(active, return_when=return_when)
            healthy = True
            for future in completed:
                active.pop(future)
                record = future.result()
                settled += 1
                healthy = healthy and record["status"] == DocumentStatus.COMPLETED
            if admission_strategy == AdmissionStrategy.RAMP and healthy:
                limit = _next_ramp_limit(limit, maximum=max_concurrent_documents)
            elif not healthy:
                stopped = True
            if not stopped:
                _fill_slots(
                    active,
                    pending=pending,
                    executor=executor,
                    limit=limit,
                    extract_pii=extract_pii,
                    writer=writer,
                )
    return settled, stopped


def _fill_slots(
    active: Dict[Future, ChildTask],
    *,
    pending: Iterator[ChildTask],
    executor: ThreadPoolExecutor,
    limit: int,
    extract_pii: Callable,
    writer: FrameWriter,
) -> None:
    while len(active) < limit:
        task = next(pending, None)
        if task is None:
            return
        future = executor.submit(_execute_task, task, extract_pii=extract_pii, writer=writer)
        active[future] = task


def _execute_task(task: ChildTask, *, extract_pii: Callable, writer: FrameWriter) -> Dict[str, object]:
    from src.evaluation.worker import _execute_document

    started_at = time.monotonic()
    writer.write(FrameType.STARTED, ordinal=task.ordinal, document_id=task.document_id)
    record = _execute_document(
        {"ordinal": task.ordinal, "document_id": task.document_id, "text": task.text},
        extract_pii=extract_pii,
    )
    record["latency_seconds"] = time.monotonic() - started_at
    writer.write(FrameType.SETTLED, ordinal=task.ordinal, document_id=task.document_id, result=record)
    return record


def _initial_limit(maximum: int, *, admission_strategy: AdmissionStrategyValue) -> int:
    if admission_strategy == AdmissionStrategy.IMMEDIATE:
        return maximum
    return min(RAMP_STAGES[0], maximum)


def _next_ramp_limit(current: int, *, maximum: int) -> int:
    return min(next((stage for stage in RAMP_STAGES if stage > current), maximum), maximum)


def _start_parent_watchdog() -> None:
    parent_pid = int(os.environ[PARENT_PID_ENVIRONMENT])
    thread = threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True)
    thread.start()


def _watch_parent(parent_pid: int) -> None:
    while True:
        time.sleep(0.1)
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return
