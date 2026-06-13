"""Backtesting calculations for VaR outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import pandas as pd

from volatility_factor.portfolio import PortfolioPricingData


@dataclass(frozen=True)
class BacktestSummary:
    volatility_factor_id: str
    tickers: str
    weights: str
    start_date: str | None
    end_date: str | None
    liquidation_days: int
    var_value: float
    var_is_log: bool
    confidence_level: float | None
    observations: int
    exceptions: int
    exception_rate: float
    expected_exception_rate: float | None
    min_shortfall: float
    mean_shortfall: float

    def to_record(self) -> dict:
        return self.__dict__.copy()


class VaRBacktest(PortfolioPricingData):
    """Data-injected replacement for the legacy ``VarBacktest`` calculations."""

    def __init__(
        self,
        tickers: str | Sequence[str],
        pricing_data: pd.DataFrame,
        n_liquidation: int,
        weights: Sequence[float] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        var_value: float | None = None,
        var_is_log: bool = True,
        forward_shift: bool = False,
        confidence_level: float | None = None,
        volatility_factor_id: str | None = None,
    ):
        super().__init__(
            tickers=tickers,
            weights=weights,
            pricing_data=pricing_data,
            volatility_factor_id=volatility_factor_id,
        )
        self.n_liquidation = int(n_liquidation)
        self.start_date = self._validate_optional_date(start_date, "start_date")
        self.end_date = self._validate_optional_date(end_date, "end_date")
        self.var_value = var_value
        self.var_is_log = var_is_log
        self.forward_shift = forward_shift
        self.confidence_level = confidence_level

    @staticmethod
    def _validate_optional_date(value: str | None, field_name: str) -> str | None:
        if value is None:
            return None
        if re.match(r"^[\d]{4}-[\d]{2}-[\d]{2}$", value):
            return value
        raise ValueError(f"{field_name} must be None or formatted as YYYY-MM-DD")

    @property
    def var_pct(self) -> float:
        if self.var_value is None:
            raise ValueError("var_value is required for exception and shortfall calculations")
        if self.var_is_log:
            return float(-(np.exp(-self.var_value / 100) - 1))
        return float(self.var_value / 100)

    @property
    def nday_price_return(self) -> pd.Series:
        df = self.price_matrix.sort_index().copy()
        if self.forward_shift:
            pct_chg = df.shift(-self.n_liquidation) / df - 1
        else:
            pct_chg = df / df.shift(self.n_liquidation) - 1

        if self.single_index:
            values = pct_chg.iloc[:, 0].values
        else:
            values = pct_chg.dot(self.weights)
        series = pd.Series(data=values, index=df.index, name="nday_price_return")
        return series[self.start_date : self.end_date].dropna()

    @property
    def nday_log_return(self) -> pd.Series:
        df = self.log_price_matrix.sort_index().copy()
        if self.forward_shift:
            log_chg = df.shift(-self.n_liquidation) - df
        else:
            log_chg = df - df.shift(self.n_liquidation)

        if self.single_index:
            values = log_chg.iloc[:, 0].values
        else:
            values = log_chg.dot(self.weights)
        series = pd.Series(data=values, index=df.index, name="nday_log_return")
        return series[self.start_date : self.end_date].dropna()

    @property
    def exception_series(self) -> pd.Series:
        series = (-self.var_pct > self.nday_price_return).astype(int)
        series.name = "exception"
        return series

    @property
    def shortfall_series(self) -> pd.Series:
        series = self.nday_price_return + self.var_pct
        series.name = "shortfall"
        return series

    @property
    def durations(self) -> list[int]:
        breaks = np.array(self.exception_series)
        breaks = np.insert(breaks, 0, 1)
        breaks = np.append(breaks, 1)
        return list(np.diff(np.where(breaks == 1))[0, 1:-1])

    def summary(self) -> BacktestSummary:
        exceptions = int(self.exception_series.sum())
        observations = int(self.exception_series.shape[0])
        expected_exception_rate = (
            float(1 - self.confidence_level) if self.confidence_level is not None else None
        )
        shortfall = self.shortfall_series
        return BacktestSummary(
            volatility_factor_id=self.volatility_factor_id,
            tickers=self.ticker_description,
            weights=self.weight_description,
            start_date=self.start_date,
            end_date=self.end_date,
            liquidation_days=self.n_liquidation,
            var_value=float(self.var_value),
            var_is_log=self.var_is_log,
            confidence_level=self.confidence_level,
            observations=observations,
            exceptions=exceptions,
            exception_rate=float(exceptions / observations) if observations else np.nan,
            expected_exception_rate=expected_exception_rate,
            min_shortfall=float(shortfall.min()) if not shortfall.empty else np.nan,
            mean_shortfall=float(shortfall.mean()) if not shortfall.empty else np.nan,
        )

