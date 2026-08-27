import json
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Mapping, Optional, Tuple, cast

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"
SUPPORTED_PATHS = (CHAT_COMPLETIONS_PATH, RESPONSES_PATH)
SUPPORTED_SERVICE_TIERS = (None, "default")
SUPPORTED_TOOL_TYPES = ("function",)
PRICE_TABLE_VERSION = "2026-08-08"
DEFAULT_MAX_OUTPUT_TOKENS = 16_384
TOKENS_PER_MILLION = Decimal(1_000_000)


class CostStatus:
    PENDING = "pending"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.PENDING, cls.COMPLETE, cls.INCOMPLETE


CostStatusValue = Literal["pending", "complete", "incomplete"]


class EvaluationMode(StrEnum):
    CACHE_FILL = "cache-fill"
    FRESH = "fresh"
    CACHE = "cache"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return tuple(mode.value for mode in cls)

    @property
    def reads_cache(self) -> bool:
        return self in {self.CACHE_FILL, self.CACHE}

    @property
    def allows_upstream(self) -> bool:
        return self in {self.CACHE_FILL, self.FRESH}

    @property
    def writes_cache(self) -> bool:
        return self is self.CACHE_FILL

    @property
    def requires_api_key_upfront(self) -> bool:
        return self is self.FRESH


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


GPT_4O_PRICE = ModelPrice(Decimal("2.50"), Decimal("1.25"), Decimal("10.00"))
GPT_4O_MINI_PRICE = ModelPrice(Decimal("0.15"), Decimal("0.075"), Decimal("0.60"))
MODEL_PRICES = {
    "gpt-4o": GPT_4O_PRICE,
    "gpt-4o-2024-08-06": GPT_4O_PRICE,
    "gpt-4o-2024-11-20": GPT_4O_PRICE,
    "gpt-4o-mini": GPT_4O_MINI_PRICE,
    "gpt-4o-mini-2024-07-18": GPT_4O_MINI_PRICE,
}


def model_price(model: str) -> ModelPrice:
    if model not in MODEL_PRICES:
        raise ValueError(f"no price configured for model {model!r}")
    return MODEL_PRICES[model]


def supported_models() -> Tuple[str, ...]:
    return tuple(MODEL_PRICES)


class MeteringError(RuntimeError):
    pass


class SpendingLimitExceededError(MeteringError):
    pass


class CacheFillFailedError(MeteringError):
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

    @property
    def input_tokens(self) -> int:
        return sum(usage.input_tokens for usage in self.usages)

    @property
    def cached_input_tokens(self) -> int:
        return sum(usage.cached_input_tokens for usage in self.usages)

    @property
    def output_tokens(self) -> int:
        return sum(usage.output_tokens for usage in self.usages)


@dataclass(frozen=True)
class MeteringOutcome:
    report: CostReport
    status: CostStatusValue
    errors: Tuple[str, ...] = ()
    active_request_count: int = 0
    evaluation_mode: EvaluationMode = EvaluationMode.FRESH
    cache_hits: int = 0
    cache_misses: int = 0
    live_requests: int = 0
    cache_writes: int = 0
    cache_write_errors: int = 0
    cache_errors: int = 0
    reserved_api_cost_usd: Decimal = Decimal()
    unknown_api_cost_liability_usd: Decimal = Decimal()
    maximum_api_cost_exposure_usd: Decimal = Decimal()
    peak_reserved_api_cost_usd: Decimal = Decimal()
    peak_active_upstream_requests: int = 0
    reservation_wait_seconds: float = 0.0

    @property
    def cost_is_final(self) -> bool:
        return self.status == CostStatus.COMPLETE and self.unknown_api_cost_liability_usd == 0


def cost_is_comparable(outcome: MeteringOutcome, *, result_is_complete: bool) -> bool:
    return result_is_complete and (
        outcome.evaluation_mode is EvaluationMode.FRESH
        or (outcome.evaluation_mode is EvaluationMode.CACHE_FILL and outcome.cache_hits == 0)
    )


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


def request_cost_upper_bound(path: str, body: bytes) -> Decimal:
    payload = cast(Mapping[str, object], json.loads(body))
    price = model_price(str(payload["model"]))
    output_tokens = _maximum_output_tokens(path, payload=payload)
    input_cost = Decimal(len(body)) * price.input_per_million
    output_cost = Decimal(output_tokens) * price.output_per_million
    return (input_cost + output_cost) / TOKENS_PER_MILLION


def _maximum_output_tokens(path: str, *, payload: Mapping[str, object]) -> int:
    field_names = (
        ("max_completion_tokens", "max_tokens") if path == CHAT_COMPLETIONS_PATH else ("max_output_tokens",)
    )
    value = next((payload[name] for name in field_names if payload.get(name) is not None), None)
    if value is None:
        return DEFAULT_MAX_OUTPUT_TOKENS
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MeteringError(f"invalid maximum output tokens {value!r} for {path}")
    return value


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
