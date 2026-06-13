"""Shared helpers for VaR method classes."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from volatility_factor.portfolio import PortfolioPricingData
from volatility_factor.results import (
    CONFIDENCE_LEVELS,
    MethodResult,
    arithmetic_var_from_log_quantile,
    log_var_from_log_quantile,
)


class BaseVaRMethod(PortfolioPricingData):
    """Base class for VaR methods using historical log returns."""

    def __init__(
        self,
        tickers: str | Sequence[str],
        weights: Sequence[float] | None,
        pricing_data: pd.DataFrame,
        sample_start: str,
        sample_end: str,
        volatility_factor_id: str | None = None,
    ):
        super().__init__(
            tickers=tickers,
            weights=weights,
            pricing_data=pricing_data,
            volatility_factor_id=volatility_factor_id,
        )
        self.sample_start = self._validate_date(sample_start, "sample_start")
        self.sample_end = self._validate_date(sample_end, "sample_end")

    @staticmethod
    def _validate_date(value: str, field_name: str) -> str:
        if re.match(r"^[\d]{4}-[\d]{2}-[\d]{2}$", value):
            return value
        raise ValueError(f"{field_name} must be formatted as string (YYYY-MM-DD)")

    @property
    def constituent_returns(self) -> pd.DataFrame:
        return self.log_return_matrix[self.sample_start : self.sample_end].dropna()

    @property
    def portfolio_returns(self) -> pd.Series:
        returns = self.wtd_log_return[self.sample_start : self.sample_end].iloc[:, 0].dropna()
        returns.name = "portfolio_log_return"
        return returns

    def horizon_portfolio_returns(self, n_liquidation: int) -> pd.Series:
        if n_liquidation < 1:
            raise ValueError("n_liquidation must be at least 1")
        returns = self.portfolio_returns.rolling(n_liquidation).sum().dropna()
        if returns.empty:
            raise ValueError("Sample period is too short for requested liquidation horizon")
        returns.name = f"{n_liquidation}_day_portfolio_log_return"
        return returns

    def result_from_log_quantiles(
        self,
        method: str,
        n_liquidation: int,
        log_quantiles: Mapping[float, float],
        method_variant: str | None = None,
        parameters: Mapping | None = None,
        diagnostics: Mapping | None = None,
    ) -> MethodResult:
        var_by_confidence = {
            level: arithmetic_var_from_log_quantile(value)
            for level, value in log_quantiles.items()
        }
        log_var_by_confidence = {
            level: log_var_from_log_quantile(value) for level, value in log_quantiles.items()
        }
        return MethodResult(
            volatility_factor_id=self.volatility_factor_id,
            tickers=self.ticker_description,
            weights=self.weight_description,
            method=method,
            method_variant=method_variant,
            sample_start=self.sample_start,
            sample_end=self.sample_end,
            liquidation_days=n_liquidation,
            var_by_confidence=var_by_confidence,
            log_var_by_confidence=log_var_by_confidence,
            parameters=parameters or {},
            diagnostics=diagnostics or {},
        )


def confidence_levels_or_default(
    confidence_levels: Sequence[float] | None,
) -> tuple[float, ...]:
    if confidence_levels is None:
        return CONFIDENCE_LEVELS
    return tuple(float(level) for level in confidence_levels)


def historical_log_quantiles(
    returns: pd.Series,
    confidence_levels: Sequence[float] | None = None,
) -> dict[float, float]:
    levels = confidence_levels_or_default(confidence_levels)
    return {
        level: float(returns.quantile(1 - level, interpolation="higher"))
        for level in levels
    }


def covariance_portfolio_variance(returns: pd.DataFrame, weights: np.ndarray) -> float:
    covariance = returns.cov().values
    return float(weights.dot(covariance).dot(weights))

