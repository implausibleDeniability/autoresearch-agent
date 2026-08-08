import threading
import time
from typing import List, Optional

import httpx

from cost_accounting import CostReport, MeteringError, ModelUsage


class MeterState:
    def __init__(self, *, api_key: str, run_token: str, upstream_base_url: str) -> None:
        self.api_key = api_key
        self.run_token = run_token
        self.upstream_base_url = upstream_base_url
        self.client = httpx.Client(timeout=300.0)
        self._condition = threading.Condition()
        self._accepting = True
        self._active_requests = 0
        self._usages: List[ModelUsage] = []
        self._errors: List[str] = []

    def begin_request(self, authorization: Optional[str]) -> int:
        if authorization != f"Bearer {self.run_token}":
            return 401
        with self._condition:
            if not self._accepting:
                return 503
            self._active_requests += 1
        return 0

    def finish_request(self, *, usage: Optional[ModelUsage], error: str = "") -> None:
        with self._condition:
            if usage is not None:
                self._usages.append(usage)
            if error:
                self._errors.append(error)
            self._active_requests -= 1
            self._condition.notify_all()

    def seal_and_report(self, *, timeout: float) -> CostReport:
        deadline = time.monotonic() + timeout
        with self._condition:
            self._accepting = False
            while self._active_requests and time.monotonic() < deadline:
                self._condition.wait(deadline - time.monotonic())
            self._validate_final_state(timeout)
            return CostReport(tuple(self._usages))

    def stop_accepting(self) -> None:
        with self._condition:
            self._accepting = False

    def close(self) -> None:
        self.client.close()

    def _validate_final_state(self, timeout: float) -> None:
        if self._active_requests:
            raise MeteringError(f"{self._active_requests} metered requests did not finish within {timeout}s")
        if self._errors:
            raise MeteringError("; ".join(self._errors))
