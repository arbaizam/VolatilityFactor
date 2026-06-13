"""Configuration helpers for volatility factor model runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class FactorConfig:
    """One configured volatility factor portfolio."""

    volatility_factor_id: str
    tickers: tuple[str, ...]
    ndays: tuple[int, ...]
    weights: tuple[float, ...] | None = None


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load the project YAML configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_weights(value: Any) -> tuple[float, ...] | None:
    """Normalize YAML weight values.

    The legacy configuration uses the literal string ``"None"`` in places, so
    this intentionally accepts both real null values and the string spelling.
    """

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return tuple(float(item) for item in value)


def iter_factor_configs(config: dict[str, Any]) -> Iterable[FactorConfig]:
    """Yield factor definitions from the legacy ``var_settings`` section."""

    for factor_id, values in config.get("var_settings", {}).items():
        yield FactorConfig(
            volatility_factor_id=str(factor_id),
            tickers=tuple(str(ticker) for ticker in values["tickers"]),
            ndays=tuple(int(day) for day in values["ndays"]),
            weights=normalize_weights(values.get("weights")),
        )

