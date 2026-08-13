import json
import os
import time

import pytest

from src.cost_metering.accounting import CostReport
from src.evaluation.worker import (
    MAX_RESULT_BYTES,
    WorkerProtocolError,
    _parse_document_record,
    _read_record,
)


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
