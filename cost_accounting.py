import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping, Optional, Tuple, cast

from model_pricing import model_price, supported_models

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"
SUPPORTED_PATHS = (CHAT_COMPLETIONS_PATH, RESPONSES_PATH)
SUPPORTED_SERVICE_TIERS = (None, "default")
SUPPORTED_TOOL_TYPES = ("function",)


class MeteringError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelUsage:
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> Decimal:
        price = model_price(self.model)
        uncached = self.input_tokens - self.cached_input_tokens
        total = uncached * price.input_per_million
        total += self.cached_input_tokens * price.cached_input_per_million
        total += self.output_tokens * price.output_per_million
        return total / Decimal(1_000_000)


@dataclass(frozen=True)
class CostReport:
    usages: Tuple[ModelUsage, ...]

    @property
    def total_usd(self) -> Decimal:
        return sum((usage.cost_usd for usage in self.usages), Decimal())

    def cost_per_million_source_tokens(self, source_tokens: int) -> Decimal:
        if source_tokens <= 0:
            raise ValueError(f"source_tokens must be positive, got {source_tokens}")
        return self.total_usd * Decimal(1_000_000) / Decimal(source_tokens)


class StreamUsageParser:
    def __init__(self, *, path: str) -> None:
        self._path = path
        self._buffer = b""
        self._usage: Optional[ModelUsage] = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        self._buffer = self._buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            self._consume_event(event)

    def finish(self) -> ModelUsage:
        if self._buffer.strip():
            self._consume_event(self._buffer)
        if self._usage is None:
            raise MeteringError(f"successful streaming response for {self._path} omitted usage")
        return self._usage

    def _consume_event(self, event: bytes) -> None:
        lines = (line[5:].strip() for line in event.splitlines() if line.startswith(b"data:"))
        data = b"\n".join(lines)
        if not data or data == b"[DONE]":
            return
        usage = usage_from_payload(self._path, json.loads(data))
        if usage is None:
            return
        if self._usage is not None:
            raise MeteringError(f"stream for {self._path} returned usage more than once")
        self._usage = usage


def prepare_request(path: str, body: bytes) -> Tuple[bytes, bool]:
    if path not in SUPPORTED_PATHS:
        raise MeteringError(f"unsupported OpenAI endpoint {path!r}; expected one of {SUPPORTED_PATHS}")
    payload = json.loads(body)
    _validate_request(payload)
    is_stream = payload.get("stream") is True
    if path == CHAT_COMPLETIONS_PATH and is_stream:
        stream_options = dict(payload.get("stream_options") or {})
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
    return json.dumps(payload, separators=(",", ":")).encode(), is_stream


def parse_response_usage(path: str, content: bytes) -> ModelUsage:
    usage = usage_from_payload(path, json.loads(content))
    if usage is None:
        raise MeteringError(f"successful response for {path} omitted usage")
    return usage


def usage_from_payload(path: str, payload: Mapping[str, object]) -> Optional[ModelUsage]:
    if path == RESPONSES_PATH and payload.get("type") == "response.completed":
        payload = cast(Mapping[str, object], payload["response"])
    raw_usage = payload.get("usage")
    if raw_usage is None:
        return None
    usage = cast(Mapping[str, object], raw_usage)
    model = str(payload["model"])
    if path == CHAT_COMPLETIONS_PATH:
        details = cast(Mapping[str, object], usage.get("prompt_tokens_details") or {})
        return _make_usage(
            model=model,
            input_tokens=usage["prompt_tokens"],
            cached_input_tokens=details.get("cached_tokens", 0),
            output_tokens=usage["completion_tokens"],
        )
    details = cast(Mapping[str, object], usage.get("input_tokens_details") or {})
    return _make_usage(
        model=model,
        input_tokens=usage["input_tokens"],
        cached_input_tokens=details.get("cached_tokens", 0),
        output_tokens=usage["output_tokens"],
    )


def _validate_request(payload: Mapping[str, object]) -> None:
    model = payload.get("model")
    if model not in supported_models():
        raise MeteringError(f"unsupported model {model!r}; expected one of {supported_models()}")
    service_tier = payload.get("service_tier")
    if service_tier not in SUPPORTED_SERVICE_TIERS:
        raise MeteringError(f"unsupported service_tier {service_tier!r}; expected default pricing")
    _validate_tools(payload.get("tools", ()))


def _validate_tools(raw_tools: object) -> None:
    tools = cast(list, raw_tools)
    unsupported = [tool.get("type") for tool in tools if tool.get("type") not in SUPPORTED_TOOL_TYPES]
    if unsupported:
        raise MeteringError(
            f"unsupported hosted tool types {unsupported}; only local function tools have complete pricing"
        )


def _make_usage(
    *, model: str, input_tokens: object, cached_input_tokens: object, output_tokens: object
) -> ModelUsage:
    values = (input_tokens, cached_input_tokens, output_tokens)
    if not all(isinstance(value, int) and value >= 0 for value in values):
        raise MeteringError(
            f"invalid usage for model {model!r}: input={input_tokens}, "
            f"cached={cached_input_tokens}, output={output_tokens}"
        )
    if cached_input_tokens > input_tokens:
        raise MeteringError(
            f"cached input exceeds total input for model {model!r}: {cached_input_tokens}>{input_tokens}"
        )
    model_price(model)
    return ModelUsage(
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )
