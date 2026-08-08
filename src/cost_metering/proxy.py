import secrets
import threading
import time
from decimal import Decimal
from http.server import ThreadingHTTPServer
from typing import List, Optional

import httpx

from .accounting import CostReport, MeteringError, ModelUsage, SpendingLimitExceededError
from .http import MeterHandler

DEFAULT_SPENDING_LIMIT_USD = Decimal("0.08")
SPENDING_LIMIT_STATUS = 429


class MeterState:
    def __init__(
        self,
        *,
        api_key: str,
        run_token: str,
        upstream_base_url: str,
        spending_limit_usd: Decimal = DEFAULT_SPENDING_LIMIT_USD,
    ) -> None:
        _validate_spending_limit(spending_limit_usd)
        self.api_key = api_key
        self.run_token = run_token
        self.upstream_base_url = upstream_base_url
        self.client = httpx.Client(timeout=300.0)
        self._condition = threading.Condition()
        self._accepting = True
        self._limit_exceeded = False
        self._spending_limit_usd = spending_limit_usd
        self._observed_spending_usd = Decimal()
        self._active_requests = 0
        self._usages: List[ModelUsage] = []
        self._errors: List[str] = []

    def begin_request(self, authorization: Optional[str]) -> int:
        if authorization != f"Bearer {self.run_token}":
            return 401
        with self._condition:
            if not self._accepting:
                return SPENDING_LIMIT_STATUS if self._limit_exceeded else 503
            self._active_requests += 1
        return 0

    def finish_request(self, *, usage: Optional[ModelUsage], error: str = "") -> None:
        with self._condition:
            if usage is not None:
                self._usages.append(usage)
                self._observed_spending_usd += usage.cost_usd
                if self._observed_spending_usd > self._spending_limit_usd:
                    self._limit_exceeded = True
                    self._accepting = False
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
        if self._limit_exceeded:
            raise SpendingLimitExceededError(
                f"observed API spending ${self._observed_spending_usd:.8f} exceeded absolute run "
                f"limit ${self._spending_limit_usd:.2f}; in-flight requests may cause bounded overshoot"
            )
        if self._errors:
            raise MeteringError("; ".join(self._errors))


class MeteringProxy:
    def __init__(
        self,
        *,
        api_key: str,
        upstream_base_url: str = "https://api.openai.com",
        spending_limit_usd: Decimal = DEFAULT_SPENDING_LIMIT_USD,
    ) -> None:
        _validate_api_key(api_key)
        self._state = MeterState(
            api_key=api_key,
            run_token=secrets.token_urlsafe(32),
            upstream_base_url=upstream_base_url.rstrip("/"),
            spending_limit_usd=spending_limit_usd,
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), MeterHandler)
        self._server.state = self._state
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._closed = False

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    @property
    def run_token(self) -> str:
        return self._state.run_token

    def __enter__(self) -> "MeteringProxy":
        self._thread.start()
        return self

    def seal_and_report(self, *, timeout: float = 30.0) -> CostReport:
        report = self._state.seal_and_report(timeout=timeout)
        self.close()
        return report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._state.stop_accepting()
        self._server.shutdown()
        self._server.server_close()
        self._state.close()
        self._thread.join(timeout=5.0)

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


def _validate_api_key(api_key: str) -> None:
    if not api_key.strip():
        raise ValueError("api_key must not be empty")


def _validate_spending_limit(spending_limit_usd: Decimal) -> None:
    if spending_limit_usd <= 0:
        raise ValueError(f"spending_limit_usd must be positive, got {spending_limit_usd}")
