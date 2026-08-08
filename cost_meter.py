import secrets
import threading
from http.server import ThreadingHTTPServer

from cost_accounting import CostReport
from meter_state import MeterState
from metering_http import MeterHandler


class MeteringProxy:
    def __init__(self, *, api_key: str, upstream_base_url: str = "https://api.openai.com") -> None:
        _validate_api_key(api_key)
        self._state = MeterState(
            api_key=api_key,
            run_token=secrets.token_urlsafe(32),
            upstream_base_url=upstream_base_url.rstrip("/"),
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
