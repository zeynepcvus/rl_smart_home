from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

BASE_HOURLY_PRICES = np.array(
    [
        0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 1.2, 1.2,
        2.5, 2.5, 2.5, 2.5,
        2.5, 2.5, 1.5, 1.5,
        2.5, 2.5, 2.5, 2.5,
        1.5, 1.2, 1.0, 0.8,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class PriceConfig:
    noise_pct: float = 0.05
    min_price: float = 0.05


def get_price_category_from_value(price: float) -> str:
    if price <= 1.0:
        return "cheap"
    if price <= 1.8:
        return "normal"
    return "expensive"


def get_price_category(hour: int, profile: Sequence[float] | None = None) -> str:
    if profile is None:
        price = float(BASE_HOURLY_PRICES[hour % 24])
    else:
        price = float(profile[hour % 24])
    return get_price_category_from_value(price)


def build_daily_price_profile(
    rng: np.random.Generator,
    config: PriceConfig | None = None,
    base_prices: np.ndarray | None = None,
) -> np.ndarray:
    config = config or PriceConfig()
    base = np.array(base_prices if base_prices is not None else BASE_HOURLY_PRICES, dtype=np.float32)

    if config.noise_pct <= 0:
        return base.copy()

    noise = rng.uniform(-config.noise_pct, config.noise_pct, size=24).astype(np.float32)
    profile = base * (1.0 + noise)
    profile = np.clip(profile, config.min_price, None)
    return np.round(profile, 3)
