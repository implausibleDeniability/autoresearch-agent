import json
import socket
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from openai import BadRequestError, OpenAI
from pydantic import BaseModel

from src.cost_metering.accounting import (
    CHAT_COMPLETIONS_PATH,
    CostReport,
    CostStatus,
    EvaluationMode,
    MeteringError,
    MeteringOutcome,
    ModelUsage,
    SpendingLimitExceededError,
    StreamUsageParser,
    cost_is_comparable,
    parse_response_usage,
    prepare_request,
    request_cost_upper_bound,
)
from src.cost_metering.proxy import SPENDING_LIMIT_STATUS, MeteringProxy, MeterState
from src.cost_metering.http import MAX_REQUEST_BODY_BYTES
from src.cost_metering.response_cache import CachedResponse, ResponseCache


class _Answer(BaseModel):
    answer: str


class _UpstreamServer(ThreadingHTTPServer):
    request_queue_size = 200

    def __init__(self) -> None:
        self.requests = []
        super().__init__(("127.0.0.1", 0), _UpstreamHandler)

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        server = self.server
        content_length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(content_length))
        server.requests.append((self.path, self.headers["Authorization"], payload))
        time.sleep(payload.get("metadata", {}).get("delay_seconds", 0))
        if payload.get("metadata", {}).get("omit_usage"):
            self._send_json(_chat_response(usage=None))
        elif payload.get("stream"):
            self._send_stream(payload["model"], responses=self.path != CHAT_COMPLETIONS_PATH)
        elif self.path == CHAT_COMPLETIONS_PATH:
            self._send_json(_chat_response())
        else:
            self._send_json(_responses_response())

    def _send_json(self, payload) -> None:
        content = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_stream(self, model: str, *, responses: bool) -> None:
        if responses:
            self._send_responses_stream(model)
            return
        events = [
            {
                "id": "chatcmpl-test",
                "choices": [],
                "created": 0,
                "model": model,
                "object": "chat.completion.chunk",
                "usage": None,
            },
            {
                "id": "chatcmpl-test",
                "choices": [],
                "created": 0,
                "model": model,
                "object": "chat.completion.chunk",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            },
        ]
        content = (
            b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_responses_stream(self, model: str) -> None:
        response = _responses_response()
        response["model"] = model
        event = {"type": "response.completed", "response": response}
        content = f"data: {json.dumps(event)}\r\n\r\ndata: [DONE]\r\n\r\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string, *args) -> None:
        pass


@pytest.fixture
def upstream_server():
    server = _UpstreamServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def test_structured_chat_completion_is_forwarded_and_metered(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:
        client = OpenAI(api_key=meter.run_token, base_url=meter.base_url)

        # operate
        completion = client.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": "answer"}],
            response_format=_Answer,
        )
        report = meter.seal_and_report()

    # check
    assert completion.choices[0].message.parsed == _Answer(answer="ok")
    assert report.total_usd == Decimal("0.000325")
    assert upstream_server.requests[0][1] == "Bearer real-key"


def test_chat_stream_forces_terminal_usage_and_meters_it(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:
        client = OpenAI(api_key=meter.run_token, base_url=meter.base_url)

        # operate
        chunks = list(
            client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": "answer"}],
                stream=True,
            )
        )
        report = meter.seal_and_report()

    # check
    assert chunks[-1].usage.total_tokens == 110
    assert report.total_usd == Decimal("0.000325")
    assert upstream_server.requests[0][2]["stream_options"] == {"include_usage": True}


def test_responses_api_usage_is_metered(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:

        # operate
        response = httpx.post(
            f"{meter.base_url}/responses",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json={"model": "gpt-4o-mini-2024-07-18", "input": "answer"},
        )
        report = meter.seal_and_report()

    # check
    assert response.status_code == 200
    assert report.total_usd == Decimal("0.0000165")


def test_streaming_responses_api_with_crlf_is_metered(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:

        # operate
        with httpx.stream(
            "POST",
            f"{meter.base_url}/responses",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json={"model": "gpt-4o-mini-2024-07-18", "input": "answer", "stream": True},
        ) as response:
            content = b"".join(response.iter_bytes())
        outcome = meter.finalize(timeout=5.0)

    # check
    assert response.status_code == 200
    assert b"response.completed" in content
    assert outcome.report.total_usd == Decimal("0.0000165")
    assert len(outcome.request_receipts) == 1
    assert outcome.request_receipts[0].request_ordinal == 0
    assert outcome.request_receipts[0].replayed is False
    assert len(outcome.request_receipts[0].request_key) == 64
    assert len(outcome.request_receipts[0].response_content_sha256) == 64


def test_cache_fill_response_is_written_once_and_replayed_without_upstream(tmp_path, upstream_server):
    cache_directory = tmp_path / "responses"
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [{"role": "user", "content": "answer"}],
    }
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_directory,
    ) as meter:
        live_response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        live_outcome = meter.finalize(timeout=5.0)

    request_count = len(upstream_server.requests)
    with MeteringProxy(
        evaluation_mode="cache",
        upstream_base_url=upstream_server.base_url,
        cache_directory=cache_directory,
    ) as meter:
        cached_response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        cached_outcome = meter.finalize(timeout=5.0)

    assert cached_response.status_code == live_response.status_code == 200
    assert cached_response.headers["content-type"] == live_response.headers["content-type"]
    assert cached_response.content == live_response.content
    assert len(upstream_server.requests) == request_count
    assert live_outcome.live_requests == 1
    assert live_outcome.cache_writes == 1
    assert cached_outcome.cache_hits == 1
    assert cached_outcome.cache_misses == 0
    assert cached_outcome.live_requests == 0
    assert cached_outcome.report.total_usd == Decimal()
    assert len(live_outcome.request_receipts) == len(cached_outcome.request_receipts) == 1
    live_receipt = live_outcome.request_receipts[0]
    cached_receipt = cached_outcome.request_receipts[0]
    assert live_receipt.request_key == cached_receipt.request_key
    assert live_receipt.response_content_sha256 == cached_receipt.response_content_sha256
    assert live_receipt.replayed is False
    assert cached_receipt.replayed is True


def test_receipts_preserve_document_and_request_order_under_concurrency(upstream_server):
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:
        first_token = meter.issue_token(document_ordinal=8)
        second_token = meter.issue_token(document_ordinal=3)

        def request(token, label, delay):
            return httpx.post(
                f"{meter.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": "gpt-4o-2024-08-06",
                    "messages": [{"role": "user", "content": label}],
                    "metadata": {"delay_seconds": delay},
                },
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(request, first_token, "first", 0.1),
                executor.submit(request, first_token, "second", 0.0),
                executor.submit(request, second_token, "third", 0.0),
            ]
            assert all(future.result().status_code == 200 for future in futures)
        outcome = meter.finalize(timeout=5.0)

    identities = [(receipt.document_ordinal, receipt.request_ordinal) for receipt in outcome.request_receipts]
    assert identities == [(3, 0), (8, 0), (8, 1)]


def test_cache_fill_hit_replays_without_credentials_or_live_call(tmp_path, upstream_server):
    cache_directory = tmp_path / "responses"
    payload = {"model": "gpt-4o-2024-08-06", "messages": []}
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_directory,
    ) as meter:
        primed = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        meter.finalize(timeout=5.0)

    request_count = len(upstream_server.requests)
    with MeteringProxy(
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_directory,
    ) as meter:
        replayed = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        outcome = meter.finalize(timeout=5.0)

    assert replayed.content == primed.content
    assert len(upstream_server.requests) == request_count
    assert outcome.cache_hits == 1
    assert outcome.cache_misses == 0
    assert outcome.live_requests == 0


def test_cache_fill_miss_without_credentials_fails_without_live_call(tmp_path, upstream_server):
    with MeteringProxy(
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=tmp_path / "responses",
    ) as meter:
        response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json={"model": "gpt-4o-2024-08-06", "messages": []},
        )
        outcome = meter.finalize(timeout=5.0)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "response_cache_fill_requires_api_key"
    assert upstream_server.requests == []
    assert outcome.cache_misses == 1
    assert outcome.live_requests == 0
    assert outcome.status == "incomplete"


def test_fresh_bypasses_cache_reads_and_writes(tmp_path, upstream_server):
    cache_directory = tmp_path / "responses"
    payload = {"model": "gpt-4o-2024-08-06", "messages": []}
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_directory,
    ) as meter:
        httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        meter.finalize(timeout=5.0)
    cache_snapshot = {path: path.read_bytes() for path in cache_directory.rglob("*.json")}
    request_count = len(upstream_server.requests)

    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="fresh",
        cache_directory=cache_directory,
    ) as meter:
        response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        outcome = meter.finalize(timeout=5.0)

    assert response.status_code == 200
    assert len(upstream_server.requests) == request_count + 1
    assert outcome.live_requests == 1
    assert outcome.cache_hits == 0
    assert outcome.cache_misses == 0
    assert outcome.cache_writes == 0
    assert {path: path.read_bytes() for path in cache_directory.rglob("*.json")} == cache_snapshot


def test_concurrent_cache_fill_misses_share_one_live_request(tmp_path, upstream_server):
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [],
        "metadata": {"delay_seconds": 0.1},
    }
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=tmp_path / "responses",
    ) as meter:

        def request():
            return httpx.post(
                f"{meter.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {meter.run_token}"},
                json=payload,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(executor.map(lambda _: request(), range(8)))
        outcome = meter.finalize(timeout=5.0)

    assert [response.status_code for response in responses] == [200] * 8
    assert len(upstream_server.requests) == 1
    assert outcome.cache_misses == 1
    assert outcome.cache_hits == 7
    assert outcome.live_requests == 1
    assert outcome.cache_writes == 1


def test_concurrent_failed_cache_fills_do_not_retry_live_request(tmp_path, upstream_server):
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [],
        "metadata": {"delay_seconds": 0.1, "omit_usage": True},
    }
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=tmp_path / "responses",
    ) as meter:

        def request():
            return httpx.post(
                f"{meter.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {meter.run_token}"},
                json=payload,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(executor.map(lambda _: request(), range(8)))
        outcome = meter.finalize(timeout=5.0)

    assert [response.status_code for response in responses] == [502] * 8
    assert len(upstream_server.requests) == 1
    assert outcome.cache_misses == 8
    assert outcome.cache_hits == 0
    assert outcome.live_requests == 1
    assert outcome.cache_writes == 0
    assert outcome.status == "incomplete"


def test_high_cardinality_cache_fill_claims_are_released_after_success_and_failure(tmp_path):
    state = MeterState(
        api_key="real-key",
        run_token="run-token",
        upstream_base_url="http://127.0.0.1:1",
        evaluation_mode=EvaluationMode.CACHE_FILL,
        response_cache=ResponseCache(
            tmp_path / "responses",
            upstream_base_url="http://127.0.0.1:1",
        ),
    )

    for index in range(200):
        response, request_key = state.claim_cache_fill(
            path=CHAT_COMPLETIONS_PATH,
            body=json.dumps({"model": "gpt-4o", "messages": [], "seed": index}).encode(),
            headers={},
        )
        assert response is None
        assert request_key is not None
        state.finish_cache_fill(request_key, succeeded=index % 2 == 0)

    assert state._cache_fills_in_progress == set()
    state.close()


def test_cached_miss_is_not_retried_or_forwarded(tmp_path, upstream_server):
    with MeteringProxy(
        evaluation_mode="cache",
        upstream_base_url=upstream_server.base_url,
        cache_directory=tmp_path / "responses",
    ) as meter:
        client = OpenAI(api_key=meter.run_token, base_url=meter.base_url)

        with pytest.raises(BadRequestError, match="found no exact response"):
            client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": "answer"}],
            )
        outcome = meter.finalize(timeout=5.0)

    assert upstream_server.requests == []
    assert outcome.cache_hits == 0
    assert outcome.cache_misses == 1
    assert outcome.live_requests == 0
    assert outcome.status == "incomplete"
    assert outcome.errors == ("strict response cache miss",)


def test_streaming_response_is_cached_and_replayed_exactly(tmp_path, upstream_server):
    cache_directory = tmp_path / "responses"
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [{"role": "user", "content": "answer"}],
        "stream": True,
    }
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_directory,
    ) as meter:
        with httpx.stream(
            "POST",
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        ) as response:
            live_content = b"".join(response.iter_bytes())
        live_outcome = meter.finalize(timeout=5.0)

    request_count = len(upstream_server.requests)
    with MeteringProxy(
        evaluation_mode="cache",
        upstream_base_url=upstream_server.base_url,
        cache_directory=cache_directory,
    ) as meter:
        with httpx.stream(
            "POST",
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        ) as response:
            cached_content = b"".join(response.iter_bytes())
        cached_outcome = meter.finalize(timeout=5.0)

    assert cached_content == live_content
    assert len(upstream_server.requests) == request_count
    assert live_outcome.cache_writes == 1
    assert cached_outcome.cache_hits == 1


def test_usage_invalid_success_is_not_cached(tmp_path, upstream_server):
    cache_directory = tmp_path / "responses"
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [{"role": "user", "content": "answer"}],
        "metadata": {"omit_usage": True},
    }
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_directory,
    ) as meter:
        live_response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        live_outcome = meter.finalize(timeout=5.0)

    with MeteringProxy(
        evaluation_mode="cache",
        upstream_base_url=upstream_server.base_url,
        cache_directory=cache_directory,
    ) as meter:
        cached_response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        cached_outcome = meter.finalize(timeout=5.0)

    assert live_response.status_code == 502
    assert live_outcome.cache_writes == 0
    assert cached_response.status_code == 400
    assert cached_outcome.cache_misses == 1


@pytest.mark.parametrize("evaluation_mode", ["cache-fill", "cache"])
def test_malformed_cache_entry_fails_closed_without_upstream(tmp_path, upstream_server, evaluation_mode):
    cache_directory = tmp_path / "responses"
    payload = {"model": "gpt-4o-2024-08-06", "messages": []}
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_directory,
    ) as meter:
        response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        meter.finalize(timeout=5.0)
    assert response.status_code == 200
    entry = next(cache_directory.rglob("*.json"))
    entry.write_text("malformed")
    request_count = len(upstream_server.requests)

    with MeteringProxy(
        api_key="real-key" if evaluation_mode == "cache-fill" else None,
        evaluation_mode=evaluation_mode,
        upstream_base_url=upstream_server.base_url,
        cache_directory=cache_directory,
    ) as meter:
        response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json=payload,
        )
        outcome = meter.finalize(timeout=5.0)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "response_cache_error"
    assert len(upstream_server.requests) == request_count
    assert outcome.cache_errors == 1
    assert outcome.status == "incomplete"


@pytest.mark.parametrize(
    ("mode", "cache_hits", "result_is_complete", "expected"),
    [
        (EvaluationMode.FRESH, 0, True, True),
        (EvaluationMode.CACHE_FILL, 0, True, True),
        (EvaluationMode.CACHE_FILL, 1, True, False),
        (EvaluationMode.CACHE, 1, True, False),
        (EvaluationMode.FRESH, 0, False, False),
        (EvaluationMode.CACHE_FILL, 0, False, False),
    ],
)
def test_cost_comparability_requires_complete_uncached_evidence(
    mode, cache_hits, result_is_complete, expected
):
    outcome = MeteringOutcome(
        CostReport(()),
        CostStatus.COMPLETE if result_is_complete else CostStatus.INCOMPLETE,
        evaluation_mode=mode,
        cache_hits=cache_hits,
    )

    assert cost_is_comparable(outcome, result_is_complete=result_is_complete) is expected


def test_cache_write_failure_preserves_response_and_invalidates_cache_fill(tmp_path, upstream_server):
    cache_path = tmp_path / "not-a-directory"
    cache_path.write_text("blocked")
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        evaluation_mode="cache-fill",
        cache_directory=cache_path,
    ) as meter:
        response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json={"model": "gpt-4o-2024-08-06", "messages": []},
        )
        outcome = meter.finalize(timeout=5.0)

    assert response.status_code == 200
    assert outcome.status == "incomplete"
    assert any("could not persist" in error for error in outcome.errors)
    assert outcome.cache_writes == 0
    assert outcome.cache_write_errors == 1


def test_response_cache_uses_complete_request_identity_and_atomic_owner_only_files(tmp_path):
    directory = tmp_path / "responses"
    cache = ResponseCache(directory, upstream_base_url="HTTP://LOCALHOST:80/")
    path = "/v1/chat/completions"
    headers = {
        "idempotency-key": "attempt-1",
        "openai-beta": "structured-outputs=v1",
        "openai-organization": "org-1",
        "openai-project": "project-1",
    }
    first = CachedResponse(200, "application/json", b'{"answer":"first"}')
    replacement = CachedResponse(200, "application/json", b'{"answer":"replacement"}')

    assert cache.put(path=path, body=b'{"b":2,"a":1}', headers=headers, response=first)
    assert not cache.put(path=path, body=b'{"a":1,"b":2}', headers=headers, response=replacement)
    assert cache.get(path=path, body=b'{"a":1,"b":2}', headers=headers) == first
    assert cache.get(path="/v1/responses", body=b'{"a":1,"b":2}', headers=headers) is None
    assert cache.get(path=path, body=b'{"a":1,"b":3}', headers=headers) is None
    for header_name in headers:
        changed = dict(headers)
        changed[header_name] = "different"
        assert cache.get(path=path, body=b'{"a":1,"b":2}', headers=changed) is None
    other_origin = ResponseCache(directory, upstream_base_url="http://localhost:81")
    assert other_origin.get(path=path, body=b'{"a":1,"b":2}', headers=headers) is None

    concurrent_body = b'{"input":"concurrent"}'
    responses = [CachedResponse(200, "application/json", str(index).encode()) for index in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        stored = list(
            executor.map(
                lambda response: cache.put(
                    path="/v1/responses",
                    body=concurrent_body,
                    headers={},
                    response=response,
                ),
                responses,
            )
        )
    assert stored.count(True) == 1
    assert cache.get(path="/v1/responses", body=concurrent_body, headers={}) in responses
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.rglob("*.json"))


def test_successful_response_without_usage_invalidates_run(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:
        client = OpenAI(api_key=meter.run_token, base_url=meter.base_url)

        # operate
        with pytest.raises(Exception):
            client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": "answer"}],
                metadata={"omit_usage": True},
            )

        # check
        with pytest.raises(MeteringError, match="omitted usage"):
            meter.seal_and_report()


def test_unknown_model_invalidates_run(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:

        # operate
        response = httpx.post(
            f"{meter.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json={"model": "unknown", "messages": []},
        )

        # check
        assert response.status_code == 502
        with pytest.raises(MeteringError, match="unsupported model"):
            meter.seal_and_report()


def test_concurrent_requests_are_all_metered(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:

        def request():
            return httpx.post(
                f"{meter.base_url}/responses",
                headers={"Authorization": f"Bearer {meter.run_token}"},
                json={"model": "gpt-4o-mini-2024-07-18", "input": "answer"},
            )

        # operate
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(lambda _: request(), range(4)))
        report = meter.seal_and_report()

    # check
    assert [response.status_code for response in responses] == [200] * 4
    assert len(report.usages) == 4


def test_proxy_admits_one_hundred_fifty_bounded_upstream_requests(upstream_server):
    payload = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [],
        "max_completion_tokens": 1,
        "metadata": {"delay_seconds": 0.05},
    }
    with MeteringProxy(
        api_key="real-key",
        upstream_base_url=upstream_server.base_url,
        max_upstream_requests=150,
    ) as meter:

        def request():
            return httpx.post(
                f"{meter.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {meter.run_token}"},
                json=payload,
                timeout=10.0,
            )

        with ThreadPoolExecutor(max_workers=150) as executor:
            responses = list(executor.map(lambda _: request(), range(150)))
        outcome = meter.finalize(timeout=5.0)

    assert [response.status_code for response in responses] == [200] * 150
    assert len(outcome.report.usages) == 150
    assert 1 < outcome.peak_active_upstream_requests <= 150


def test_wrong_run_token_is_rejected_without_polluting_report(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:

        # operate
        response = httpx.post(
            f"{meter.base_url}/responses",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "gpt-4o-mini-2024-07-18", "input": "answer"},
        )
        report = meter.seal_and_report()

    # check
    assert response.status_code == 401
    assert report.usages == ()


def test_oversized_proxy_request_is_rejected_without_upstream_call(upstream_server):
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:
        host, port = meter._server.server_address
        with socket.create_connection((host, port), timeout=2.0) as connection:
            request = (
                "POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Authorization: Bearer {meter.run_token}\r\n"
                f"Content-Length: {MAX_REQUEST_BODY_BYTES + 1}\r\n"
                "Connection: close\r\n\r\n"
            )
            connection.sendall(request.encode())
            response = connection.recv(4096)
        outcome = meter.finalize(timeout=1.0)

    assert response.startswith(b"HTTP/1.1 502")
    assert upstream_server.requests == []
    assert outcome.status == "incomplete"


def test_hosted_tool_invalidates_run(upstream_server):
    # setup
    with MeteringProxy(api_key="real-key", upstream_base_url=upstream_server.base_url) as meter:

        # operate
        response = httpx.post(
            f"{meter.base_url}/responses",
            headers={"Authorization": f"Bearer {meter.run_token}"},
            json={
                "model": "gpt-4o-2024-08-06",
                "input": "search",
                "tools": [{"type": "web_search"}],
            },
        )

        # check
        assert response.status_code == 502
        with pytest.raises(MeteringError, match="unsupported hosted tool"):
            meter.seal_and_report()


def test_default_pricing_rejects_automatic_service_tier():
    body = json.dumps({"model": "gpt-4o", "messages": [], "service_tier": "auto"}).encode()

    with pytest.raises(MeteringError, match="unsupported service_tier"):
        prepare_request(CHAT_COMPLETIONS_PATH, body)


def test_local_function_tools_remain_available():
    body = json.dumps(
        {
            "model": "gpt-4o",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        }
    ).encode()

    prepared, is_stream = prepare_request(CHAT_COMPLETIONS_PATH, body)

    assert json.loads(prepared)["tools"][0]["function"]["name"] == "lookup"
    assert is_stream is False


def test_request_cost_reservation_uses_requested_output_limit():
    small = json.dumps(
        {"model": "gpt-4o-mini-2024-07-18", "messages": [], "max_completion_tokens": 10}
    ).encode()
    large = json.dumps(
        {"model": "gpt-4o-mini-2024-07-18", "messages": [], "max_completion_tokens": 100}
    ).encode()

    assert request_cost_upper_bound(CHAT_COMPLETIONS_PATH, large) > request_cost_upper_bound(
        CHAT_COMPLETIONS_PATH, small
    )


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_request_cost_reservation_rejects_invalid_output_limit(value):
    body = json.dumps(
        {"model": "gpt-4o-mini-2024-07-18", "messages": [], "max_completion_tokens": value}
    ).encode()

    with pytest.raises(MeteringError, match="invalid maximum output tokens"):
        request_cost_upper_bound(CHAT_COMPLETIONS_PATH, body)


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "completion_tokens": 1},
        {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    ],
)
def test_invalid_usage_fails_closed(usage):
    content = json.dumps(_chat_response(usage=usage)).encode()

    with pytest.raises(MeteringError, match="invalid usage|cached input exceeds"):
        parse_response_usage(CHAT_COMPLETIONS_PATH, content)


def test_stream_missing_usage_fails_closed():
    parser = StreamUsageParser(path=CHAT_COMPLETIONS_PATH)
    parser.feed(b"data: [DONE]\n\n")

    with pytest.raises(MeteringError, match="omitted usage"):
        parser.finish()


def test_stream_duplicate_usage_fails_closed():
    parser = StreamUsageParser(path=CHAT_COMPLETIONS_PATH)
    event = f"data: {json.dumps(_chat_response())}\n\n".encode()

    parser.feed(event)

    with pytest.raises(MeteringError, match="usage more than once"):
        parser.feed(event)


def test_seal_waits_for_admitted_request_and_rejects_new_requests():
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    assert state.begin_request("Bearer run-token") == 0
    state.stop_accepting()

    with ThreadPoolExecutor(max_workers=1) as executor:
        report_future = executor.submit(state.seal_and_report, timeout=1.0)
        assert state.begin_request("Bearer run-token") == 503
        state.finish_request(usage=None)
        report = report_future.result()
    state.close()

    assert report.usages == ()


def test_zero_cost_cache_lookup_is_admitted_while_live_spending_is_reserved():
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    prior_usage = ModelUsage(
        model="gpt-4o-2024-08-06",
        input_tokens=30_000,
        cached_input_tokens=0,
        output_tokens=0,
    )
    assert state.begin_request("Bearer run-token") == 0
    state.finish_request(usage=prior_usage)
    assert state.begin_request("Bearer run-token", reservation_usd=Decimal("0.01")) == 0

    assert state.begin_request("Bearer run-token") == 0

    state.finish_request(usage=None)
    state.finish_request(usage=None, reservation_usd=Decimal("0.01"))
    outcome = state.finalize(timeout=1.0)
    state.close()
    assert outcome.status == "complete"


def test_observed_spending_over_limit_rejects_retries_and_fails_run():
    # setup
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    usage = ModelUsage(
        model="gpt-4o-2024-08-06",
        input_tokens=40_000,
        cached_input_tokens=0,
        output_tokens=0,
    )
    assert state.begin_request("Bearer run-token") == 0

    # operate
    state.finish_request(usage=usage)

    # check
    assert state.begin_request("Bearer run-token") == SPENDING_LIMIT_STATUS
    with pytest.raises(SpendingLimitExceededError, match=r"\$0\.10000000.*limit \$0\.08000000"):
        state.seal_and_report(timeout=1.0)
    state.close()


def test_in_flight_requests_are_counted_before_bounded_overshoot_failure():
    # setup
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    usage = ModelUsage(
        model="gpt-4o-2024-08-06",
        input_tokens=20_000,
        cached_input_tokens=0,
        output_tokens=0,
    )
    assert state.begin_request("Bearer run-token") == 0
    assert state.begin_request("Bearer run-token") == 0

    # operate
    state.finish_request(usage=usage)
    state.finish_request(usage=usage)

    # check
    with pytest.raises(SpendingLimitExceededError, match=r"\$0\.10000000.*bounded overshoot"):
        state.seal_and_report(timeout=1.0)
    state.close()


def test_structured_finalization_preserves_observed_usage_on_metering_error():
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    usage = ModelUsage(
        model="gpt-4o-mini-2024-07-18",
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=10,
    )
    assert state.begin_request("Bearer run-token") == 0
    state.finish_request(usage=usage, error="provider response could not be validated")

    outcome = state.finalize(timeout=1.0)
    state.close()

    assert outcome.status == "incomplete"
    assert outcome.report.usages == (usage,)
    assert outcome.report.total_usd > 0
    assert outcome.errors == ("provider response could not be validated",)


def test_finalization_outcome_is_immutable_when_active_request_finishes_late():
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    usage = ModelUsage(
        model="gpt-4o-mini-2024-07-18",
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=10,
    )
    assert state.begin_request("Bearer run-token") == 0

    first = state.finalize(timeout=0.0)
    state.finish_request(usage=usage)
    second = state.finalize(timeout=1.0)
    state.close()

    assert first is second
    assert first.status == "incomplete"
    assert first.active_request_count == 1
    assert first.report.usages == ()


def test_checkpoint_keeps_meter_open_for_later_requests():
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    usage = ModelUsage(
        model="gpt-4o-mini-2024-07-18",
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=10,
    )

    checkpoint = state.checkpoint(timeout=0.0)
    assert state.begin_request("Bearer run-token") == 0
    state.finish_request(usage=usage)
    final = state.finalize(timeout=1.0)
    state.close()

    assert checkpoint.status == "complete"
    assert checkpoint.report.usages == ()
    assert final.report.usages == (usage,)


def test_document_tokens_isolate_usage_and_metering_errors():
    # setup
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    first_token = state.issue_token()
    second_token = state.issue_token()
    first_usage = ModelUsage("gpt-4o-mini-2024-07-18", 100, 0, 10)
    second_usage = ModelUsage("gpt-4o-mini-2024-07-18", 200, 0, 20)

    # operate
    assert state.begin_request(f"Bearer {first_token}") == 0
    state.finish_request(usage=first_usage, run_token=first_token)
    assert state.begin_request(f"Bearer {first_token}") == 0
    state.finish_request(
        usage=first_usage,
        error="first document metering failed",
        run_token=first_token,
    )
    assert state.begin_request(f"Bearer {second_token}") == 0
    state.finish_request(usage=second_usage, run_token=second_token)
    first = state.token_outcome(first_token, timeout=0.0)
    second = state.token_outcome(second_token, timeout=0.0)
    state.close()

    # check
    assert first.status == "incomplete"
    assert first.report.usages == (first_usage, first_usage)
    assert first.errors == ("first document metering failed",)
    assert second.status == "complete"
    assert second.report.usages == (second_usage,)
    assert second.errors == ()


def test_request_reservations_bound_concurrent_spending():
    # setup
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    first_token = state.issue_token()
    second_token = state.issue_token()
    reservation = Decimal("0.06")
    usage = ModelUsage("gpt-4o-mini-2024-07-18", 50_000, 0, 0)
    entered = threading.Event()
    assert state.begin_request(f"Bearer {first_token}", reservation_usd=reservation) == 0

    # operate
    with ThreadPoolExecutor(max_workers=1) as executor:
        second = executor.submit(
            _begin_reserved_request,
            state,
            second_token,
            reservation,
            entered,
        )
        assert entered.wait(timeout=1.0)
        assert not second.done()
        state.finish_request(
            usage=usage,
            run_token=first_token,
            reservation_usd=reservation,
        )
        assert second.result(timeout=1.0) == 0
    state.finish_request(
        usage=usage,
        run_token=second_token,
        reservation_usd=reservation,
    )
    outcome = state.finalize(timeout=1.0)
    state.close()

    # check
    assert outcome.status == "complete"
    assert outcome.report.usages == (usage, usage)


def test_settled_spend_does_not_consume_inflight_liability_capacity():
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    first_token = state.issue_token()
    second_token = state.issue_token()
    usage = ModelUsage("gpt-4o-2024-08-06", 20_000, 0, 0)
    assert state.begin_request("Bearer run-token") == 0
    state.finish_request(usage=usage)
    assert state.begin_request(f"Bearer {first_token}", reservation_usd=Decimal("0.01")) == 0

    second_status = state.begin_request(
        f"Bearer {second_token}",
        reservation_usd=Decimal("0.06"),
    )

    assert second_status == 0
    state.finish_request(usage=None, run_token=first_token, reservation_usd=Decimal("0.01"))
    state.finish_request(usage=None, run_token=second_token, reservation_usd=Decimal("0.06"))
    state.close()


def test_unknown_billing_retains_liability_and_prevents_final_cost():
    state = MeterState(api_key="real-key", run_token="run-token", upstream_base_url="http://127.0.0.1:1")
    reservation = Decimal("0.06")
    assert state.begin_request("Bearer run-token", reservation_usd=reservation) == 0

    state.finish_request(
        usage=None,
        reservation_usd=reservation,
        upstream_started=True,
    )
    outcome = state.finalize(timeout=1.0)
    state.close()

    assert outcome.status == "incomplete"
    assert outcome.cost_is_final is False
    assert outcome.reserved_api_cost_usd == Decimal()
    assert outcome.unknown_api_cost_liability_usd == reservation
    assert outcome.maximum_api_cost_exposure_usd == reservation


def test_single_request_above_liability_limit_fails_without_waiting():
    state = MeterState(
        api_key="real-key",
        run_token="run-token",
        upstream_base_url="http://127.0.0.1:1",
        max_inflight_liability_usd=Decimal("0.05"),
    )

    status = state.begin_request("Bearer run-token", reservation_usd=Decimal("0.06"))
    outcome = state.finalize(timeout=0.0)
    state.close()

    assert status == SPENDING_LIMIT_STATUS
    assert outcome.status == "incomplete"
    assert "exceeds in-flight limit" in outcome.errors[0]


def _begin_reserved_request(
    state: MeterState,
    run_token: str,
    reservation: Decimal,
    entered: threading.Event,
) -> int:
    entered.set()
    return state.begin_request(f"Bearer {run_token}", reservation_usd=reservation)


def _chat_response(*, usage="default"):
    payload = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": '{"answer":"ok"}', "refusal": None, "role": "assistant"},
            }
        ],
        "created": 0,
        "model": "gpt-4o-2024-08-06",
        "object": "chat.completion",
    }
    if usage == "default":
        payload["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
    elif usage is not None:
        payload["usage"] = usage
    return payload


def _responses_response():
    return {
        "id": "resp-test",
        "model": "gpt-4o-mini-2024-07-18",
        "object": "response",
        "output": [],
        "status": "completed",
        "usage": {
            "input_tokens": 80,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 90,
        },
    }
