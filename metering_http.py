import json
from http.server import BaseHTTPRequestHandler
from typing import Dict, Mapping, Optional, Protocol, cast
from urllib.parse import urlsplit

import httpx

from cost_accounting import ModelUsage, StreamUsageParser, parse_response_usage, prepare_request
from meter_state import MeterState

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


class MeterServerProtocol(Protocol):
    state: MeterState


class MeterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        state = cast(MeterServerProtocol, self.server).state
        rejection_status = state.begin_request(self.headers.get("Authorization"))
        if rejection_status:
            self._send_json(rejection_status, payload={"error": "metering request rejected"})
            return
        usage = None
        error = ""
        try:
            usage = self._forward(state)
        except Exception as caught_error:
            error = f"metering failed for {self.path}: {caught_error}"
            self._send_json(502, payload={"error": error})
        finally:
            state.finish_request(usage=usage, error=error)

    def _forward(self, state: MeterState) -> Optional[ModelUsage]:
        path = urlsplit(self.path).path
        body, is_stream = prepare_request(path, self._read_body())
        headers = _upstream_headers(self.headers, api_key=state.api_key)
        url = f"{state.upstream_base_url}{path}"
        with state.client.stream("POST", url, content=body, headers=headers) as response:
            if is_stream:
                return self._relay_stream(response, path=path)
            return self._relay_response(response, path=path)

    def _relay_response(self, response: httpx.Response, *, path: str) -> Optional[ModelUsage]:
        content = response.read()
        usage = parse_response_usage(path, content) if response.is_success else None
        self._send_response(response, content=content)
        return usage

    def _relay_stream(self, response: httpx.Response, *, path: str) -> Optional[ModelUsage]:
        self._start_chunked_response(response)
        parser = StreamUsageParser(path=path)
        client_connected = True
        for chunk in response.iter_bytes():
            parser.feed(chunk)
            if client_connected:
                client_connected = self._write_chunk(chunk)
        self._finish_chunks(client_connected)
        return parser.finish() if response.is_success else None

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError(f"request body is empty for {self.path}")
        return self.rfile.read(content_length)

    def _send_response(self, response: httpx.Response, *, content: bytes) -> None:
        self.send_response(response.status_code)
        self._send_headers(response.headers, content_length=len(content))
        self.end_headers()
        self.wfile.write(content)

    def _start_chunked_response(self, response: httpx.Response) -> None:
        self.send_response(response.status_code)
        self._send_headers(response.headers)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

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

    def _send_json(self, status: int, *, payload: Mapping[str, str]) -> None:
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string, *args) -> None:
        pass


def _upstream_headers(headers: Mapping[str, str], *, api_key: str) -> Dict[str, str]:
    forwarded = {
        name: value for name, value in headers.items() if name.lower() not in UPSTREAM_HEADER_BLOCKLIST
    }
    forwarded["Authorization"] = f"Bearer {api_key}"
    return forwarded
