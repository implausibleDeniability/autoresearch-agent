from dataclasses import dataclass
from decimal import Decimal
from typing import Tuple

PRICE_TABLE_VERSION = "2026-08-08"


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


MODEL_PRICES = {
    "gpt-4o": ModelPrice(
        input_per_million=Decimal("2.50"),
        cached_input_per_million=Decimal("1.25"),
        output_per_million=Decimal("10.00"),
    ),
    "gpt-4o-2024-08-06": ModelPrice(
        input_per_million=Decimal("2.50"),
        cached_input_per_million=Decimal("1.25"),
        output_per_million=Decimal("10.00"),
    ),
    "gpt-4o-2024-11-20": ModelPrice(
        input_per_million=Decimal("2.50"),
        cached_input_per_million=Decimal("1.25"),
        output_per_million=Decimal("10.00"),
    ),
    "gpt-4o-mini": ModelPrice(
        input_per_million=Decimal("0.15"),
        cached_input_per_million=Decimal("0.075"),
        output_per_million=Decimal("0.60"),
    ),
    "gpt-4o-mini-2024-07-18": ModelPrice(
        input_per_million=Decimal("0.15"),
        cached_input_per_million=Decimal("0.075"),
        output_per_million=Decimal("0.60"),
    ),
}


def model_price(model: str) -> ModelPrice:
    if model not in MODEL_PRICES:
        raise ValueError(f"no price configured for model {model!r}")
    return MODEL_PRICES[model]


def supported_models() -> Tuple[str, ...]:
    return tuple(MODEL_PRICES)
