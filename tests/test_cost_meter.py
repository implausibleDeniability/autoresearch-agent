import json
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel

from cost_accounting import (
    CHAT_COMPLETIONS_PATH,
    MeteringError,
    StreamUsageParser,
    parse_response_usage,
    prepare_request,
)
from cost_meter import MeteringProxy
from meter_state import MeterState


class _Answer(BaseModel):
    answer: str


class _UpstreamServer(ThreadingHTTPServer):
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
        report = meter.seal_and_report()

    # check
    assert response.status_code == 200
    assert b"response.completed" in content
    assert report.total_usd == Decimal("0.0000165")


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
