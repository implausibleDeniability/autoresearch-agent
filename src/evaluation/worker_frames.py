import json
import threading
from dataclasses import dataclass
from typing import Tuple

RESULT_FD_ENVIRONMENT = "EVALUATION_RESULT_FD"
PARENT_PID_ENVIRONMENT = "EVALUATION_PARENT_PID"
PROTOCOL_VERSION = 1
MAX_RESULT_BYTES = 10_000_000
MAX_AGGREGATE_RESULT_BYTES = 50_000_000


class FrameType:
    ACCEPTED = "accepted"
    STARTED = "started"
    SETTLED = "settled"
    WORKER_DONE = "worker_done"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.ACCEPTED, cls.STARTED, cls.SETTLED, cls.WORKER_DONE


@dataclass(frozen=True)
class ThreadedTask:
    ordinal: int
    document_id: str
    text: str
    source_tokens: int


@dataclass(frozen=True)
class ChildTask:
    run_id: str
    ordinal: int
    document_id: str
    text: str


class FrameWriter:
    def __init__(self, result_file, *, run_id: str) -> None:
        self._result_file = result_file
        self._run_id = run_id
        self._lock = threading.Lock()

    def write(self, frame_type: str, **fields: object) -> None:
        frame = {
            "protocol_version": PROTOCOL_VERSION,
            "frame_type": frame_type,
            "run_id": self._run_id,
            **fields,
        }
        payload = json.dumps(frame, separators=(",", ":"))
        if len(payload.encode()) > MAX_RESULT_BYTES:
            raise RuntimeError(f"worker frame exceeded {MAX_RESULT_BYTES} bytes")
        with self._lock:
            self._result_file.write(payload + "\n")
            self._result_file.flush()
