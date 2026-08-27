import json
import io
import os
import time
from types import SimpleNamespace

import pytest

from src.cost_metering.accounting import CostReport
from src.evaluation import worker
from src.evaluation.worker import (
    DocumentTask,
    MAX_RESULT_BYTES,
    WorkerProtocolError,
    _parse_document_record,
    _read_record,
)
from src.evaluation.run_results import (
    DocumentExecution,
    DocumentStatus,
    EvaluationRun,
    UsageAttributionStatus,
)
from src.evaluation import threaded_executor
from src.evaluation.threaded_executor import (
    _send_tasks,
    _updated_aggregate_bytes,
    _updated_aggregate_predictions,
    _validate_frame,
)
from src.evaluation import threaded_worker
from src.evaluation.threaded_worker import _execute_tasks, _parse_task, _read_tasks, _watch_parent
from src.evaluation.worker_frames import (
    MAX_AGGREGATE_PREDICTIONS,
    MAX_AGGREGATE_RESULT_BYTES,
    MAX_PII_VALUE_LENGTH,
    MAX_PREDICTIONS_PER_DOCUMENT,
    MAX_VALUES_PER_FIELD,
    FrameType,
    FrameWriter,
    PROTOCOL_VERSION,
    ThreadedTask,
)


def test_worker_start_failure_returns_protocol_error(monkeypatch):
    task = DocumentTask(ordinal=0, document_id="doc", text="text", source_tokens=1, run_token="token")

    def fail_to_start(**_kwargs):
        raise OSError("cannot start worker")

    monkeypatch.setattr(worker, "_start_worker", fail_to_start)

    result = worker._run_document_task(
        task,
        module="solution",
        environment={},
        deadline=time.monotonic() + 1.0,
    )

    assert result.task == task
    assert result.failure_category == "worker_protocol_error"
    assert result.record is None


def test_worker_record_rejects_wrong_sequence_or_document():
    valid = {"ordinal": 0, "document_id": "doc", "status": "completed", "predictions": []}

    with pytest.raises(WorkerProtocolError, match="sequence"):
        _parse_document_record(
            {**valid, "ordinal": 1},
            ordinal=0,
            expected_document_id="doc",
            source_tokens=1,
            usage=CostReport(()),
            usage_complete=True,
            latency_seconds=0.1,
        )
    with pytest.raises(WorkerProtocolError, match="expected 'doc'"):
        _parse_document_record(
            {**valid, "document_id": "other"},
            ordinal=0,
            expected_document_id="doc",
            source_tokens=1,
            usage=CostReport(()),
            usage_complete=True,
            latency_seconds=0.1,
        )


def test_worker_record_rejects_unknown_status_and_invalid_predictions():
    base = {"ordinal": 0, "document_id": "doc"}

    with pytest.raises(WorkerProtocolError, match="unsupported document status"):
        _parse_document_record(
            {**base, "status": "unknown"},
            ordinal=0,
            expected_document_id="doc",
            source_tokens=1,
            usage=CostReport(()),
            usage_complete=True,
            latency_seconds=0.1,
        )
    with pytest.raises(WorkerProtocolError, match="predictions must be a list"):
        _parse_document_record(
            {**base, "status": "completed", "predictions": {}},
            ordinal=0,
            expected_document_id="doc",
            source_tokens=1,
            usage=CostReport(()),
            usage_complete=True,
            latency_seconds=0.1,
        )


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ([{}] * (MAX_PREDICTIONS_PER_DOCUMENT + 1), "people per document"),
        ([{"first_name": ["x"] * (MAX_VALUES_PER_FIELD + 1)}], "field exceeded"),
        ([{"first_name": ["x" * (MAX_PII_VALUE_LENGTH + 1)]}], "value exceeded"),
    ],
)
def test_worker_record_rejects_semantically_oversized_predictions(predictions, message):
    record = {
        "ordinal": 0,
        "document_id": "doc",
        "status": "completed",
        "predictions": predictions,
    }

    with pytest.raises(WorkerProtocolError, match=message):
        _parse_document_record(
            record,
            ordinal=0,
            expected_document_id="doc",
            source_tokens=1,
            usage=CostReport(()),
            usage_complete=True,
            latency_seconds=0.1,
        )


def test_threaded_results_reject_aggregate_prediction_exhaustion():
    document = DocumentExecution(
        ordinal=0,
        document_id="doc",
        status=DocumentStatus.COMPLETED,
        source_tokens=1,
        predictions=(),
    )

    with pytest.raises(WorkerProtocolError, match="aggregate limit"):
        _updated_aggregate_predictions(MAX_AGGREGATE_PREDICTIONS + 1, document=document)


def test_evaluation_run_reports_mixed_usage_attribution():
    run = EvaluationRun.__new__(EvaluationRun)
    object.__setattr__(
        run,
        "documents",
        (
            DocumentExecution(0, "first", DocumentStatus.COMPLETED, 1),
            DocumentExecution(
                1,
                "second",
                DocumentStatus.COMPLETED,
                1,
                usage_attribution_status=UsageAttributionStatus.UNAVAILABLE,
            ),
        ),
    )

    assert run.usage_attribution_status == "mixed"


@pytest.mark.parametrize("payload", [b"not-json\n", b"[]\n"])
def test_worker_result_channel_rejects_malformed_records(payload):
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, payload)
        with pytest.raises(WorkerProtocolError):
            _read_record(read_fd, buffer=b"", deadline=time.monotonic() + 1.0)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_worker_result_channel_rejects_oversized_record():
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(WorkerProtocolError, match="exceeded"):
            _read_record(
                read_fd,
                buffer=b"x" * (MAX_RESULT_BYTES + 1),
                deadline=time.monotonic() + 1.0,
            )
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_worker_result_channel_rejects_eof_mid_record():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, json.dumps({"ordinal": 0}).encode())
    os.close(write_fd)
    try:
        with pytest.raises(EOFError, match="before a complete record"):
            _read_record(read_fd, buffer=b"", deadline=time.monotonic() + 1.0)
    finally:
        os.close(read_fd)


def test_threaded_protocol_rejects_forged_identity_and_illegal_transition():
    task = ThreadedTask(ordinal=0, document_id="doc", text="text", source_tokens=1)
    base = {"protocol_version": PROTOCOL_VERSION, "run_id": "run", "ordinal": 0}

    with pytest.raises(WorkerProtocolError, match="wrong document"):
        _validate_frame(
            {**base, "frame_type": FrameType.ACCEPTED, "document_id": "other"},
            run_id="run",
            task_by_ordinal={0: task},
            states={0: "queued"},
        )
    with pytest.raises(WorkerProtocolError, match="illegal worker transition"):
        _validate_frame(
            {**base, "frame_type": FrameType.SETTLED, "document_id": "doc"},
            run_id="run",
            task_by_ordinal={0: task},
            states={0: "queued"},
        )


@pytest.mark.parametrize(
    "frame",
    [
        {"protocol_version": 0, "run_id": "run", "frame_type": FrameType.WORKER_DONE},
        {"protocol_version": PROTOCOL_VERSION, "run_id": "other", "frame_type": FrameType.WORKER_DONE},
        {"protocol_version": PROTOCOL_VERSION, "run_id": "run", "frame_type": "unknown"},
        {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": "run",
            "frame_type": FrameType.ACCEPTED,
            "ordinal": 2,
            "document_id": "doc",
        },
    ],
)
def test_threaded_protocol_rejects_wrong_envelope_fields(frame):
    task = ThreadedTask(ordinal=0, document_id="doc", text="text", source_tokens=1)

    with pytest.raises(WorkerProtocolError):
        _validate_frame(
            frame,
            run_id="run",
            task_by_ordinal={0: task},
            states={0: "queued"},
        )


def test_threaded_protocol_enforces_frame_and_aggregate_byte_limits():
    writer = FrameWriter(io.StringIO(), run_id="run")
    with pytest.raises(RuntimeError, match="frame exceeded"):
        writer.write(FrameType.WORKER_DONE, payload="x" * MAX_RESULT_BYTES)

    with pytest.raises(WorkerProtocolError, match="aggregate limit"):
        _updated_aggregate_bytes(
            MAX_AGGREGATE_RESULT_BYTES,
            frame={"frame_type": FrameType.WORKER_DONE},
        )


def test_threaded_task_parser_rejects_malformed_and_wrong_run_records():
    with pytest.raises(json.JSONDecodeError):
        _parse_task("not-json", run_id="run")
    with pytest.raises(RuntimeError, match="wrong run"):
        _parse_task(json.dumps({"run_id": "other"}), run_id="run")


def test_threaded_worker_rejects_duplicate_task_ordinals(monkeypatch):
    task = json.dumps({"run_id": "run", "ordinal": 0, "document_id": "doc", "text": "text"})
    monkeypatch.setattr(threaded_worker.sys, "stdin", io.StringIO(f"{task}\n{task}\n"))
    writer = SimpleNamespace(write=lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="duplicate ordinal 0"):
        _read_tasks(run_id="run", writer=writer)


def test_threaded_worker_watchdog_terminates_group_after_parent_exit(monkeypatch):
    terminated = []
    monkeypatch.setattr(threaded_worker.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        threaded_worker.os,
        "kill",
        lambda _pid, _signal: (_ for _ in ()).throw(ProcessLookupError),
    )
    monkeypatch.setattr(
        threaded_worker.os,
        "killpg",
        lambda process_group, sent_signal: terminated.append((process_group, sent_signal)),
    )
    monkeypatch.setattr(threaded_worker.os, "getpgrp", lambda: 4321)

    _watch_parent(1234)

    assert terminated == [(4321, threaded_worker.signal.SIGTERM)]


def test_threaded_sender_reports_missing_and_closed_input_channels():
    task = ThreadedTask(ordinal=0, document_id="doc", text="text", source_tokens=1)
    errors = []
    _send_tasks(SimpleNamespace(stdin=None), (task,), run_id="run", errors=errors)
    assert errors == ["worker stdin is unavailable"]

    class ClosedInput:
        def write(self, _payload):
            raise BrokenPipeError

        def close(self):
            return None

    errors = []
    _send_tasks(SimpleNamespace(stdin=ClosedInput()), (task,), run_id="run", errors=errors)
    assert errors == ["worker input channel closed"]


def test_threaded_worker_start_failure_returns_complete_unfinished_ledger(monkeypatch):
    def fail_to_start(**_kwargs):
        raise OSError("cannot start")

    monkeypatch.setattr(threaded_executor, "_start_worker", fail_to_start)

    ledger, termination = threaded_executor.run_threaded_solution_documents(
        {"doc": "text"},
        module="solution",
        meter=SimpleNamespace(),
        deadline=time.monotonic() + 1.0,
        environment={},
        source_tokens={"doc": 1},
        on_checkpoint=lambda *_args: None,
        max_concurrent_documents=1,
        admission_strategy="immediate",
        run_id="run",
    )

    assert termination == "worker_protocol_error"
    assert ledger[0].status == DocumentStatus.NOT_ATTEMPTED


def test_threaded_worker_is_cleaned_up_when_checkpoint_raises(monkeypatch):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    process = SimpleNamespace()
    sender = SimpleNamespace(join=lambda **_kwargs: None, is_alive=lambda: False)
    logs = io.StringIO()
    calls = []
    monkeypatch.setattr(
        threaded_executor,
        "_start_worker",
        lambda **_kwargs: (process, read_fd, logs),
    )
    monkeypatch.setattr(threaded_executor, "_start_sender", lambda *_args, **_kwargs: sender)
    monkeypatch.setattr(
        threaded_executor,
        "_drain_frames",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("checkpoint failed")),
    )
    monkeypatch.setattr(
        threaded_executor,
        "_stop_worker",
        lambda stopped, **_kwargs: calls.append(("stop", stopped)),
    )

    def close_resources(*, result_fd, logs):
        calls.append(("close", result_fd, logs))
        os.close(result_fd)

    monkeypatch.setattr(threaded_executor, "_close_worker_resources", close_resources)

    with pytest.raises(OSError, match="checkpoint failed"):
        threaded_executor.run_threaded_solution_documents(
            {"doc": "text"},
            module="solution",
            meter=SimpleNamespace(),
            deadline=time.monotonic() + 1.0,
            environment={},
            source_tokens={"doc": 1},
            on_checkpoint=lambda *_args: None,
            max_concurrent_documents=1,
            admission_strategy="immediate",
            run_id="run",
        )

    assert calls == [("stop", process), ("close", read_fd, logs)]


def test_threaded_cleanup_terminates_before_touching_active_sender(monkeypatch):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    process = SimpleNamespace()
    calls = []
    sender = SimpleNamespace(
        is_alive=lambda: True,
        join=lambda **_kwargs: calls.append("join"),
    )
    logs = io.StringIO()
    monkeypatch.setattr(
        threaded_executor,
        "_start_worker",
        lambda **_kwargs: (process, read_fd, logs),
    )
    monkeypatch.setattr(threaded_executor, "_start_sender", lambda *_args, **_kwargs: sender)
    monkeypatch.setattr(
        threaded_executor,
        "_drain_frames",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError),
    )
    monkeypatch.setattr(
        threaded_executor,
        "_terminate_process_group",
        lambda terminated: calls.append(("terminate", terminated)),
    )
    monkeypatch.setattr(
        threaded_executor,
        "_stop_worker",
        lambda stopped, **kwargs: calls.append(("stop", stopped, kwargs["close_input"])),
    )

    def close_resources(*, result_fd, logs):
        calls.append("close")
        os.close(result_fd)

    monkeypatch.setattr(threaded_executor, "_close_worker_resources", close_resources)

    with pytest.raises(TimeoutError):
        threaded_executor.run_threaded_solution_documents(
            {"doc": "text"},
            module="solution",
            meter=SimpleNamespace(),
            deadline=time.monotonic() + 1.0,
            environment={},
            source_tokens={"doc": 1},
            on_checkpoint=lambda *_args: None,
            max_concurrent_documents=1,
            admission_strategy="immediate",
            run_id="run",
        )

    assert calls == [
        ("terminate", process),
        "join",
        ("stop", process, False),
        "close",
    ]


def test_immediate_admission_refills_after_each_completion():
    started_at = time.monotonic()
    starts = {}

    class Writer:
        def write(self, frame_type, **fields):
            if frame_type == FrameType.STARTED:
                starts[fields["document_id"]] = time.monotonic() - started_at

    tasks = tuple(
        threaded_worker.ChildTask("run", index, name, name)
        for index, name in enumerate(("slow", "fast", "third"))
    )

    def extract(text):
        time.sleep(0.2 if text == "slow" else 0.01)
        return []

    settled, stopped = _execute_tasks(
        tasks,
        extract_pii=extract,
        writer=Writer(),
        max_concurrent_documents=2,
        admission_strategy="immediate",
    )

    assert settled == 3
    assert stopped is False
    assert starts["third"] < 0.15
