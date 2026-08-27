import secrets
import threading
import time
from decimal import Decimal
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import httpx

from .accounting import (
    CacheFillFailedError,
    CostReport,
    CostStatus,
    EvaluationMode,
    MeteringError,
    MeteringOutcome,
    ModelUsage,
    SpendingLimitExceededError,
)
from .http import MeterHandler
from .response_cache import CacheEntryError, CachedResponse, ResponseCache

DEFAULT_SPENDING_LIMIT_USD = Decimal("0.08")
DEFAULT_INFLIGHT_LIABILITY_LIMIT_USD = Decimal("0.08")
DEFAULT_PROXY_INFLIGHT_LIABILITY_LIMIT_USD = Decimal("1.00")
DEFAULT_MAX_UPSTREAM_REQUESTS = 150
HANDLER_HEADROOM = 16
SPENDING_LIMIT_STATUS = 429


class MeterState:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        run_token: str,
        upstream_base_url: str,
        spending_limit_usd: Decimal = DEFAULT_SPENDING_LIMIT_USD,
        max_inflight_liability_usd: Decimal = DEFAULT_INFLIGHT_LIABILITY_LIMIT_USD,
        evaluation_mode: EvaluationMode = EvaluationMode.FRESH,
        response_cache: Optional[ResponseCache] = None,
        admission_deadline: Optional[float] = None,
        max_upstream_requests: int = DEFAULT_MAX_UPSTREAM_REQUESTS,
    ) -> None:
        _validate_spending_limit(spending_limit_usd)
        _validate_spending_limit(max_inflight_liability_usd)
        evaluation_mode = _normalize_evaluation_mode(evaluation_mode)
        self.api_key = api_key or ""
        self.run_token = run_token
        self.upstream_base_url = upstream_base_url
        self.evaluation_mode = evaluation_mode
        limits = httpx.Limits(
            max_connections=max_upstream_requests,
            max_keepalive_connections=max_upstream_requests,
        )
        self.client = (
            httpx.Client(timeout=300.0, limits=limits)
            if evaluation_mode.allows_upstream and self.api_key
            else None
        )
        self._response_cache = response_cache
        self._condition = threading.Condition()
        self._run_token = run_token
        self._run_tokens = {run_token}
        self._accepting = True
        self._limit_exceeded = False
        self._spending_limit_usd = spending_limit_usd
        self._max_inflight_liability_usd = max_inflight_liability_usd
        self._admission_deadline = admission_deadline
        self._max_upstream_requests = max_upstream_requests
        self._observed_spending_usd = Decimal()
        self._reserved_spending_usd = Decimal()
        self._unknown_liability_usd = Decimal()
        self._peak_reserved_spending_usd = Decimal()
        self._reservation_wait_seconds = 0.0
        self._active_requests = 0
        self._active_upstream_requests = 0
        self._peak_active_upstream_requests = 0
        self._active_requests_by_token = {run_token: 0}
        self._usages: List[ModelUsage] = []
        self._usages_by_token: Dict[str, List[ModelUsage]] = {run_token: []}
        self._errors: List[str] = []
        self._errors_by_token: Dict[str, List[str]] = {run_token: []}
        self._cache_hits = 0
        self._cache_misses = 0
        self._live_requests = 0
        self._cache_writes = 0
        self._cache_write_errors = 0
        self._cache_errors = 0
        self._cache_fills_in_progress: set[str] = set()
        self._cache_fill_failures: set[str] = set()
        self._final_outcome: Optional[MeteringOutcome] = None

    def issue_token(self) -> str:
        with self._condition:
            run_token = secrets.token_urlsafe(32)
            self._run_tokens.add(run_token)
            self._active_requests_by_token[run_token] = 0
            self._usages_by_token[run_token] = []
            self._errors_by_token[run_token] = []
            return run_token

    def resolve_run_token(self, authorization: Optional[str]) -> Optional[str]:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        run_token = authorization.removeprefix("Bearer ")
        with self._condition:
            return run_token if run_token in self._run_tokens else None

    def begin_request(self, authorization: Optional[str], *, reservation_usd: Decimal = Decimal()) -> int:
        run_token = self.resolve_run_token(authorization)
        if run_token is None:
            return 401
        with self._condition:
            rejection_status = self._reserve_liability(reservation_usd)
            if rejection_status:
                return rejection_status
            self._active_requests += 1
            self._active_requests_by_token[run_token] += 1
        return 0

    def _reserve_liability(self, reservation_usd: Decimal) -> int:
        if reservation_usd > self._max_inflight_liability_usd:
            self._record_liability_error(
                f"request liability ${reservation_usd:.8f} exceeds in-flight limit "
                f"${self._max_inflight_liability_usd:.8f}"
            )
            return SPENDING_LIMIT_STATUS
        started_at = time.monotonic()
        while self._must_wait_for_liability(reservation_usd):
            remaining = self._admission_remaining_seconds()
            if remaining == 0:
                self._record_liability_error("request liability waited past the evaluation deadline")
                return SPENDING_LIMIT_STATUS
            self._condition.wait(remaining)
        self._reservation_wait_seconds += time.monotonic() - started_at
        if not self._accepting:
            return SPENDING_LIMIT_STATUS if self._limit_exceeded else 503
        self._reserved_spending_usd += reservation_usd
        self._peak_reserved_spending_usd = max(
            self._peak_reserved_spending_usd,
            self._reserved_spending_usd,
        )
        return 0

    def _must_wait_for_liability(self, reservation_usd: Decimal) -> bool:
        projected = self._reserved_spending_usd + reservation_usd
        return (
            self._accepting
            and reservation_usd > 0
            and self._reserved_spending_usd > 0
            and projected > self._max_inflight_liability_usd
        )

    def _admission_remaining_seconds(self) -> Optional[float]:
        if self._admission_deadline is None:
            return None
        return max(self._admission_deadline - time.monotonic(), 0.0)

    def _record_liability_error(self, error: str) -> None:
        if error not in self._errors:
            self._errors.append(error)

    def reserve_request_spending(self, reservation_usd: Decimal) -> int:
        with self._condition:
            return self._reserve_liability(reservation_usd)

    def finish_request(
        self,
        *,
        usage: Optional[ModelUsage],
        error: str = "",
        run_token: str = "",
        reservation_usd: Decimal = Decimal(),
        upstream_started: bool = False,
    ) -> None:
        run_token = run_token or self._run_token
        with self._condition:
            self._reserved_spending_usd -= reservation_usd
            if upstream_started and usage is None:
                self._unknown_liability_usd += reservation_usd
            if usage is not None:
                self._usages.append(usage)
                self._usages_by_token[run_token].append(usage)
                self._observed_spending_usd += usage.cost_usd
                if self._observed_spending_usd > self._spending_limit_usd:
                    self._limit_exceeded = True
                    self._accepting = False
            if error:
                self._errors.append(error)
                self._errors_by_token[run_token].append(error)
            self._active_requests -= 1
            self._active_requests_by_token[run_token] -= 1
            self._condition.notify_all()

    def begin_upstream_request(self) -> bool:
        with self._condition:
            while self._accepting and self._active_upstream_requests >= self._max_upstream_requests:
                remaining = self._admission_remaining_seconds()
                if remaining == 0:
                    return False
                self._condition.wait(remaining)
            if not self._accepting:
                return False
            self._active_upstream_requests += 1
            self._peak_active_upstream_requests = max(
                self._peak_active_upstream_requests,
                self._active_upstream_requests,
            )
            return True

    def finish_upstream_request(self) -> None:
        with self._condition:
            self._active_upstream_requests -= 1
            self._condition.notify_all()

    def record_error(self, run_token: str, *, error: str) -> None:
        with self._condition:
            self._errors.append(error)
            self._errors_by_token[run_token].append(error)
            self._condition.notify_all()

    def get_cached_response(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> Optional[CachedResponse]:
        if not self.evaluation_mode.reads_cache or self._response_cache is None:
            raise RuntimeError("response cache lookup is not enabled")
        try:
            response = self._response_cache.get(path=path, body=body, headers=headers)
        except CacheEntryError:
            with self._condition:
                self._cache_errors += 1
            raise
        with self._condition:
            if response is None:
                self._cache_misses += 1
            else:
                self._cache_hits += 1
        return response

    def claim_cache_fill(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> tuple[Optional[CachedResponse], Optional[str]]:
        if self.evaluation_mode is not EvaluationMode.CACHE_FILL or self._response_cache is None:
            raise RuntimeError("response cache fill is not enabled")
        request_key = self._response_cache.request_key(path=path, body=body, headers=headers)
        with self._condition:
            while request_key in self._cache_fills_in_progress:
                self._condition.wait()
            try:
                response = self._response_cache.get(path=path, body=body, headers=headers)
            except CacheEntryError:
                self._cache_errors += 1
                raise
            if response is not None:
                self._cache_hits += 1
                return response, None
            self._cache_misses += 1
            if request_key in self._cache_fill_failures:
                raise CacheFillFailedError(
                    "a previous cache-fill attempt for this exact request failed during this evaluation"
                )
            self._cache_fills_in_progress.add(request_key)
            return None, request_key

    def finish_cache_fill(self, request_key: str, *, succeeded: bool) -> None:
        with self._condition:
            self._cache_fills_in_progress.remove(request_key)
            if not succeeded:
                self._cache_fill_failures.add(request_key)
            self._condition.notify_all()

    def record_live_request(self) -> None:
        with self._condition:
            self._live_requests += 1

    def store_cached_response(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
        response: CachedResponse,
    ) -> bool:
        if self._response_cache is None:
            return False
        try:
            stored = self._response_cache.put(
                path=path,
                body=body,
                headers=headers,
                response=response,
            )
        except (OSError, TypeError, ValueError):
            with self._condition:
                self._cache_write_errors += 1
            return False
        if stored:
            with self._condition:
                self._cache_writes += 1
        return True

    def seal_and_report(self, *, timeout: float) -> CostReport:
        outcome = self.finalize(timeout=timeout)
        self._raise_for_outcome(outcome)
        return outcome.report

    def checkpoint(self, *, timeout: float) -> MeteringOutcome:
        return self._outcome(timeout=timeout, stop_accepting=False)

    def progress(self) -> MeteringOutcome:
        with self._condition:
            return self._make_outcome(active_request_count=self._active_requests)

    def token_outcome(self, run_token: str, *, timeout: float) -> MeteringOutcome:
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._condition:
            while self._active_requests_by_token[run_token] and time.monotonic() < deadline:
                self._condition.wait(max(deadline - time.monotonic(), 0.0))
            active_requests = self._active_requests_by_token[run_token]
            errors = list(self._errors_by_token[run_token])
            if active_requests:
                errors.append(
                    f"{active_requests} metered requests for token did not finish within {timeout}s"
                )
            return MeteringOutcome(
                report=CostReport(tuple(self._usages_by_token[run_token])),
                status=CostStatus.INCOMPLETE if errors else CostStatus.COMPLETE,
                errors=tuple(errors),
                active_request_count=active_requests,
                **self._cache_outcome_fields(),
            )

    def finalize(self, *, timeout: float) -> MeteringOutcome:
        return self._outcome(timeout=timeout, stop_accepting=True)

    def _outcome(self, *, timeout: float, stop_accepting: bool) -> MeteringOutcome:
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._condition:
            if self._final_outcome is not None:
                return self._final_outcome
            if stop_accepting:
                self._accepting = False
            while self._active_requests and time.monotonic() < deadline:
                self._condition.wait(max(deadline - time.monotonic(), 0.0))
            if self._final_outcome is not None:
                return self._final_outcome
            outcome = self._make_outcome(active_request_count=self._active_requests, timeout=timeout)
            if stop_accepting:
                self._final_outcome = outcome
            return outcome

    def _make_outcome(self, *, active_request_count: int, timeout: Optional[float] = None) -> MeteringOutcome:
        errors = list(self._errors)
        if active_request_count and timeout is not None:
            errors.append(f"{active_request_count} metered requests did not finish within {timeout}s")
        if self._limit_exceeded:
            errors.append(
                f"observed API spending ${self._observed_spending_usd:.8f} exceeded run "
                f"limit ${self._spending_limit_usd:.8f}; in-flight requests may cause bounded overshoot"
            )
        if self._unknown_liability_usd:
            errors.append(f"billing outcome is unknown for up to ${self._unknown_liability_usd:.8f}")
        return MeteringOutcome(
            report=CostReport(tuple(self._usages)),
            status=CostStatus.INCOMPLETE if errors else CostStatus.COMPLETE,
            errors=tuple(errors),
            active_request_count=active_request_count,
            **self._cache_outcome_fields(),
        )

    def _cache_outcome_fields(self) -> dict[str, object]:
        maximum_exposure = (
            self._observed_spending_usd + self._reserved_spending_usd + self._unknown_liability_usd
        )
        return {
            "evaluation_mode": self.evaluation_mode,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "live_requests": self._live_requests,
            "cache_writes": self._cache_writes,
            "cache_write_errors": self._cache_write_errors,
            "cache_errors": self._cache_errors,
            "reserved_api_cost_usd": self._reserved_spending_usd,
            "unknown_api_cost_liability_usd": self._unknown_liability_usd,
            "maximum_api_cost_exposure_usd": maximum_exposure,
            "peak_reserved_api_cost_usd": self._peak_reserved_spending_usd,
            "peak_active_upstream_requests": self._peak_active_upstream_requests,
            "reservation_wait_seconds": self._reservation_wait_seconds,
        }

    def stop_accepting(self) -> None:
        with self._condition:
            self._accepting = False
            self._condition.notify_all()

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def _raise_for_outcome(self, outcome: MeteringOutcome) -> None:
        spending_error = next(
            (error for error in outcome.errors if "exceeded run limit" in error),
            None,
        )
        if spending_error is not None:
            raise SpendingLimitExceededError(spending_error)
        if outcome.errors:
            raise MeteringError("; ".join(outcome.errors))


class _MeterServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, server_address, handler, *, max_handlers: int, backlog: int) -> None:
        self.request_queue_size = backlog
        self._handler_slots = threading.BoundedSemaphore(max_handlers)
        super().__init__(server_address, handler)

    def process_request(self, request, client_address) -> None:
        self._handler_slots.acquire()
        try:
            super().process_request(request, client_address)
        except Exception:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()


class MeteringProxy:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        upstream_base_url: str = "https://api.openai.com",
        spending_limit_usd: Decimal = DEFAULT_SPENDING_LIMIT_USD,
        max_inflight_liability_usd: Decimal = DEFAULT_PROXY_INFLIGHT_LIABILITY_LIMIT_USD,
        evaluation_mode: EvaluationMode | str = EvaluationMode.FRESH,
        cache_directory: Optional[Path] = None,
        admission_deadline: Optional[float] = None,
        max_upstream_requests: int = DEFAULT_MAX_UPSTREAM_REQUESTS,
    ) -> None:
        normalized_mode = _normalize_evaluation_mode(evaluation_mode)
        if normalized_mode.requires_api_key_upfront:
            _validate_api_key(api_key)
        if normalized_mode.reads_cache and cache_directory is None:
            raise ValueError(f"cache_directory is required in {normalized_mode.value} evaluation mode")
        normalized_upstream_base_url = upstream_base_url.rstrip("/")
        response_cache = (
            ResponseCache(cache_directory, upstream_base_url=normalized_upstream_base_url)
            if cache_directory is not None and normalized_mode.reads_cache
            else None
        )
        self._state = MeterState(
            api_key=api_key,
            run_token=secrets.token_urlsafe(32),
            upstream_base_url=normalized_upstream_base_url,
            spending_limit_usd=spending_limit_usd,
            max_inflight_liability_usd=max_inflight_liability_usd,
            evaluation_mode=normalized_mode,
            response_cache=response_cache,
            admission_deadline=admission_deadline,
            max_upstream_requests=max_upstream_requests,
        )
        max_handlers = max_upstream_requests + HANDLER_HEADROOM
        self._server = _MeterServer(
            ("127.0.0.1", 0),
            MeterHandler,
            max_handlers=max_handlers,
            backlog=max_handlers,
        )
        self._server.state = self._state
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    @property
    def run_token(self) -> str:
        return self._state.run_token

    def __enter__(self) -> "MeteringProxy":
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._thread.start()
        return self

    def seal_and_report(self, *, timeout: float = 30.0) -> CostReport:
        try:
            return self._state.seal_and_report(timeout=timeout)
        finally:
            self.close()

    def checkpoint(self, *, timeout: float) -> MeteringOutcome:
        return self._state.checkpoint(timeout=timeout)

    def issue_token(self) -> str:
        return self._state.issue_token()

    def progress(self) -> MeteringOutcome:
        return self._state.progress()

    def token_outcome(self, run_token: str, *, timeout: float) -> MeteringOutcome:
        return self._state.token_outcome(run_token, timeout=timeout)

    def finalize(self, *, timeout: float) -> MeteringOutcome:
        outcome = self._state.finalize(timeout=timeout)
        self.close()
        return outcome

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._state.stop_accepting()
        if self._thread is not None:
            self._server.shutdown()
        self._server.server_close()
        self._state.close()
        if self._thread is not None:
            self._thread.join(timeout=0.1)

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


def _validate_api_key(api_key: Optional[str]) -> None:
    if api_key is None or not api_key.strip():
        raise ValueError("api_key must not be empty")


def _normalize_evaluation_mode(evaluation_mode: EvaluationMode | str) -> EvaluationMode:
    try:
        return EvaluationMode(evaluation_mode)
    except ValueError as error:
        raise ValueError(
            f"unsupported evaluation_mode {evaluation_mode!r}; expected one of {EvaluationMode.all()}"
        ) from error


def _validate_spending_limit(spending_limit_usd: Decimal) -> None:
    if spending_limit_usd <= 0:
        raise ValueError(f"spending_limit_usd must be positive, got {spending_limit_usd}")
