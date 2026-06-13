"""Portfolio-level pricing transforms shared by VaR methods."""

from __future__ import annotations

from functools import cached_property
from typing import Sequence

import numpy as np
import pandas as pd

from volatility_factor.data_sources import normalize_pricing_data


def coerce_tickers(value: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize ticker input to a tuple of strings."""

    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple | list):
        return tuple(str(ticker) for ticker in value)
    raise ValueError(f"Tickers must be a string, tuple, or list. Got {type(value)}")


def coerce_weights(tickers: Sequence[str], value: Sequence[float] | None) -> np.ndarray:
    """Normalize and validate portfolio weights."""

    if len(tickers) == 1:
        return np.array([1.0], dtype=np.float64)
    if value is None:
        return np.array([1.0 / len(tickers)] * len(tickers), dtype=np.float64)

    weights = np.array(value, dtype=np.float64)
    if len(weights) != len(tickers):
        raise ValueError("Weights must have the same length as tickers")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must add to one. Weights totaled {weights.sum()}")
    return weights


class PortfolioPricingData:
    """Historical price and return matrices for one configured portfolio."""

    def __init__(
        self,
        tickers: str | Sequence[str],
        pricing_data: pd.DataFrame,
        weights: Sequence[float] | None = None,
        volatility_factor_id: str | None = None,
    ):
        self.tickers = coerce_tickers(tickers)
        self.weights = coerce_weights(self.tickers, weights)
        self.volatility_factor_id = volatility_factor_id or "Unknown"
        self.pricing_data = normalize_pricing_data(pricing_data, tickers=self.tickers)
        self._validate_tickers_present()

    @property
    def single_index(self) -> bool:
        return len(self.tickers) == 1

    def _validate_tickers_present(self) -> None:
        available = set(self.pricing_data["Ticker"].unique())
        missing = [ticker for ticker in self.tickers if ticker not in available]
        if missing:
            raise ValueError(f"Pricing data is missing requested tickers: {missing}")

    @cached_property
    def price_matrix(self) -> pd.DataFrame:
        df = self.pricing_data.pivot(index="Date", values="Price", columns="Ticker")
        df = pd.DataFrame(df.rename_axis(None, axis=1))[list(self.tickers)]
        return df.sort_index()

    @cached_property
    def price_return_matrix(self) -> pd.DataFrame:
        return (self.price_matrix.diff() / self.price_matrix.shift(1)).dropna()

    @cached_property
    def wtd_price_return(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.dot(self.price_return_matrix, self.weights),
            columns=["Wtd_Price_Return"],
            index=self.price_return_matrix.index,
        )

    @cached_property
    def wtd_price(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.dot(self.price_matrix, self.weights),
            columns=["Wtd_Price"],
            index=self.price_matrix.index,
        )

    @cached_property
    def log_price_matrix(self) -> pd.DataFrame:
        return np.log(self.price_matrix)

    @cached_property
    def log_return_matrix(self) -> pd.DataFrame:
        return self.log_price_matrix.diff().dropna()

    @cached_property
    def wtd_log_return(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.dot(self.log_return_matrix, self.weights),
            columns=["Wtd_Log_Return"],
            index=self.log_return_matrix.index,
        )

    @cached_property
    def index_description(self) -> str:
        df = self.pricing_data.groupby("Ticker")["Description"].first()
        values = [f"{df.loc[ticker]} ({ticker})" for ticker in self.tickers if ticker in df.index]
        return ";".join(values)

    @cached_property
    def ticker_description(self) -> str:
        return ";".join(str(ticker) for ticker in self.tickers)

    @cached_property
    def weight_description(self) -> str:
        if self.single_index:
            return "1.0"
        return ";".join(str(weight) for weight in self.weights)

