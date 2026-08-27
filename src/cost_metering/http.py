import json
from dataclasses import dataclass
from decimal import Decimal
from http.server import BaseHTTPRequestHandler
from typing import Dict, Mapping, Optional, Protocol, cast
from urllib.parse import urlsplit

import httpx

from .accounting import (
    CacheFillFailedError,
    EvaluationMode,
    ModelUsage,
    StreamUsageParser,
    parse_response_usage,
    prepare_request,
    request_cost_upper_bound,
)
from .response_cache import CacheEntryError, CachedResponse

UPSTREAM_HEADER_BLOCKLIST = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
DOWNSTREAM_HEADER_BLOCKLIST = UPSTREAM_HEADER_BLOCKLIST | {"content-encoding"}


class MeterStateProtocol(Protocol):
    api_key: str
    upstream_base_url: str
    evaluation_mode: EvaluationMode
    client: Optional[httpx.Client]

    def resolve_run_token(self, authorization: Optional[str]) -> Optional[str]: ...

    def begin_request(self, authorization: Optional[str], *, reservation_usd: Decimal = Decimal()) -> int: ...

    def reserve_request_spending(self, reservation_usd: Decimal) -> int: ...

    def finish_request(
        self,
        *,
        usage: Optional[ModelUsage],
        error: str = "",
        run_token: str = "",
        reservation_usd: Decimal = Decimal(),
    ) -> None: ...

    def record_error(self, run_token: str, *, error: str) -> None: ...

    def get_cached_response(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> Optional[CachedResponse]: ...

    def claim_cache_fill(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> tuple[Optional[CachedResponse], Optional[str]]: ...

    def finish_cache_fill(self, request_key: str, *, succeeded: bool) -> None: ...

    def record_live_request(self) -> None: ...

    def store_cached_response(
        self,
        *,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
        response: CachedResponse,
    ) -> bool: ...


class MeterServerProtocol(Protocol):
    state: MeterStateProtocol


@dataclass(frozen=True)
class _ForwardedResponse:
    usage: Optional[ModelUsage]
    cached_response: Optional[CachedResponse]


class MeterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self._response_started = False
        state = cast(MeterServerProtocol, self.server).state
        authorization = self.headers.get("Authorization")
        run_token = state.resolve_run_token(authorization)
        if run_token is None:
            self._send_json(401, payload={"error": "metering request rejected"})
            return
        try:
            path, body, is_stream = self._prepare_request()
            reservation_usd = (
                request_cost_upper_bound(path, body)
                if state.evaluation_mode is EvaluationMode.FRESH
                else Decimal()
            )
        except Exception as caught_error:
            error = f"metering failed for {self.path}: {caught_error}"
            state.record_error(run_token, error=error)
            self._send_json(502, payload={"error": error})
            return
        rejection_status = state.begin_request(authorization, reservation_usd=reservation_usd)
        if rejection_status:
            self._send_json(rejection_status, payload={"error": "metering request rejected"})
            return
        self._observed_usage = None
        error = ""
        cache_fill_key: Optional[str] = None
        cache_fill_succeeded = False
        try:
            if state.evaluation_mode is EvaluationMode.CACHE:
                if not self._replay_cache_only(state, path=path, body=body):
                    error = "strict response cache miss"
            elif state.evaluation_mode is EvaluationMode.CACHE_FILL:
                cached_response, cache_fill_key = state.claim_cache_fill(
                    path=path,
                    body=body,
                    headers=self.headers,
                )
                if cached_response is not None:
                    self._send_cached_response(cached_response)
                elif state.client is None:
                    error = "cache-fill found a miss but OPENAI_API_KEY is unavailable"
                    self._send_openai_error(
                        400,
                        message=(
                            "Cache-fill found a miss but OPENAI_API_KEY is unavailable. "
                            "No live request was made. Load .env and rerun the same command."
                        ),
                        error_type="response_cache_fill_requires_api_key",
                    )
                else:
                    requested_reservation = request_cost_upper_bound(path, body)
                    rejection_status = state.reserve_request_spending(requested_reservation)
                    if rejection_status:
                        error = "cache-fill live request was rejected by the spending meter"
                        self._send_json(rejection_status, payload={"error": "metering request rejected"})
                    else:
                        reservation_usd = requested_reservation
                        state.record_live_request()
                        forwarded = self._forward(state, path=path, body=body, is_stream=is_stream)
                        self._observed_usage = forwarded.usage
                        if forwarded.cached_response is None:
                            error = "cache-fill live response was not successful and cacheable"
                        elif state.store_cached_response(
                            path=path,
                            body=body,
                            headers=self.headers,
                            response=forwarded.cached_response,
                        ):
                            cache_fill_succeeded = True
                        else:
                            error = (
                                "paid inference succeeded but the evaluator-owned response cache "
                                "could not persist it"
                            )
            else:
                state.record_live_request()
                forwarded = self._forward(state, path=path, body=body, is_stream=is_stream)
                self._observed_usage = forwarded.usage
        except CacheFillFailedError:
            error = "a previous cache-fill attempt for this exact request failed"
            if not self._response_started:
                self._send_openai_error(
                    502,
                    message=(
                        "A previous cache-fill attempt for this exact request failed during this "
                        "evaluation. No additional OpenAI call was made. Inspect the first failure "
                        "before rerunning."
                    ),
                    error_type="response_cache_fill_failed",
                )
        except CacheEntryError:
            error = f"response cache error for {self.path}"
            if not self._response_started:
                self._send_openai_error(
                    400,
                    message=(
                        "The evaluator-owned response cache entry is invalid. No OpenAI call was "
                        "made. Escalate for human cache repair; do not modify the cache."
                    ),
                    error_type="response_cache_error",
                )
        except Exception as caught_error:
            error = f"metering failed for {self.path}: {caught_error}"
            if not self._response_started:
                self._send_json(502, payload={"error": error})
        finally:
            if cache_fill_key is not None:
                state.finish_cache_fill(cache_fill_key, succeeded=cache_fill_succeeded)
            state.finish_request(
                usage=self._observed_usage,
                error=error,
                run_token=run_token,
                reservation_usd=reservation_usd,
            )

    def _prepare_request(self) -> tuple[str, bytes, bool]:
        path = urlsplit(self.path).path
        body, is_stream = prepare_request(path, self._read_body())
        return path, body, is_stream

    def _replay_cache_only(self, state: MeterStateProtocol, *, path: str, body: bytes) -> bool:
        response = state.get_cached_response(path=path, body=body, headers=self.headers)
        if response is None:
            self._send_openai_error(
                400,
                message=(
                    "--cache found no exact response. No OpenAI call was made. Run with "
                    "--cache-fill to fill it; misses may spend API budget and require "
                    "OPENAI_API_KEY."
                ),
                error_type="response_cache_miss",
            )
            return False
        self._send_cached_response(response)
        return True

    def _send_cached_response(self, response: CachedResponse) -> None:
        self.send_response(response.status_code)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.content)))
        self.end_headers()
        self._response_started = True
        self.wfile.write(response.content)

    def _forward(
        self, state: MeterStateProtocol, *, path: str, body: bytes, is_stream: bool
    ) -> _ForwardedResponse:
        if state.client is None:
            raise RuntimeError(f"upstream client is unavailable in {state.evaluation_mode.value} mode")
        headers = _upstream_headers(self.headers, api_key=state.api_key)
        url = f"{state.upstream_base_url}{path}"
        with state.client.stream("POST", url, content=body, headers=headers) as response:
            if is_stream:
                return self._relay_stream(response, path=path)
            return self._relay_response(response, path=path)

    def _relay_response(self, response: httpx.Response, *, path: str) -> _ForwardedResponse:
        content = response.read()
        usage = parse_response_usage(path, content) if response.is_success else None
        self._observed_usage = usage
        self._send_response(response, content=content)
        cached_response = _cached_response(response, content=content) if usage is not None else None
        return _ForwardedResponse(usage=usage, cached_response=cached_response)

    def _relay_stream(self, response: httpx.Response, *, path: str) -> _ForwardedResponse:
        self._start_chunked_response(response)
        parser = StreamUsageParser(path=path)
        content = bytearray()
        client_connected = True
        for chunk in response.iter_bytes():
            content.extend(chunk)
            parser.feed(chunk)
            if client_connected:
                client_connected = self._write_chunk(chunk)
        self._finish_chunks(client_connected)
        usage = parser.finish() if response.is_success else None
        cached_response = _cached_response(response, content=bytes(content)) if usage is not None else None
        return _ForwardedResponse(usage=usage, cached_response=cached_response)

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError(f"request body is empty for {self.path}")
        return self.rfile.read(content_length)

    def _send_response(self, response: httpx.Response, *, content: bytes) -> None:
        self.send_response(response.status_code)
        self._send_headers(response.headers, content_length=len(content))
        self.end_headers()
        self._response_started = True
        self.wfile.write(content)

    def _start_chunked_response(self, response: httpx.Response) -> None:
        self.send_response(response.status_code)
        self._send_headers(response.headers)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._response_started = True

    def _send_headers(self, headers: httpx.Headers, *, content_length: Optional[int] = None) -> None:
        for name, value in headers.multi_items():
            if name.lower() not in DOWNSTREAM_HEADER_BLOCKLIST:
                self.send_header(name, value)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))

    def _write_chunk(self, chunk: bytes) -> bool:
        try:
            self.wfile.write(f"{len(chunk):X}\r\n".encode())
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            return True
        except OSError:
            return False

    def _finish_chunks(self, client_connected: bool) -> None:
        if not client_connected:
            return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass

    def _send_json(self, status: int, *, payload: Mapping[str, object]) -> None:
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self._response_started = True
        self.wfile.write(content)

    def _send_openai_error(self, status: int, *, message: str, error_type: str) -> None:
        self._send_json(
            status,
            payload={
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": error_type,
                }
            },
        )

    def log_message(self, format_string, *args) -> None:
        pass


def _upstream_headers(headers: Mapping[str, str], *, api_key: str) -> Dict[str, str]:
    forwarded = {
        name: value for name, value in headers.items() if name.lower() not in UPSTREAM_HEADER_BLOCKLIST
    }
    forwarded["Authorization"] = f"Bearer {api_key}"
    return forwarded


def _cached_response(response: httpx.Response, *, content: bytes) -> CachedResponse:
    return CachedResponse(
        status_code=response.status_code,
        content_type=response.headers.get("Content-Type", "application/octet-stream"),
        content=content,
    )
