"""Filtered historical simulation VaR."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from volatility_factor.methods.common import BaseVaRMethod, confidence_levels_or_default
from volatility_factor.results import MethodResult, arithmetic_var_from_log_quantile, log_var_from_log_quantile


def ewma_volatility_series(returns: pd.Series, lambda_: float) -> pd.Series:
    """Return EWMA conditional volatility for a univariate return series."""

    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1")
    if returns.empty:
        raise ValueError("returns cannot be empty")

    variance = float(returns.var(ddof=1))
    values = []
    for value in returns:
        variance = lambda_ * variance + (1 - lambda_) * float(value) ** 2
        values.append(np.sqrt(variance))
    return pd.Series(values, index=returns.index, name="ewma_volatility")


class FilteredHistoricalSimulationVaR(BaseVaRMethod):
    """Filtered historical simulation using EWMA standardized residuals."""

    def __init__(self, *args, random_seed: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = random_seed

    def run(
        self,
        n_liquidation: int,
        confidence_levels: Sequence[float] | None = None,
        lambda_: float = 0.94,
        n_sims: int = 10000,
    ) -> MethodResult:
        levels = confidence_levels_or_default(confidence_levels)
        returns = self.portfolio_returns
        volatility = ewma_volatility_series(returns, lambda_)
        standardized = (returns / volatility).replace([np.inf, -np.inf], np.nan).dropna()
        if standardized.empty:
            raise ValueError("No standardized residuals available for filtered historical VaR")

        latest_sigma = float(volatility.iloc[-1])
        random_state = np.random.RandomState(self.random_seed)
        sampled = random_state.choice(
            standardized.values,
            size=(n_sims, n_liquidation),
            replace=True,
        )
        simulated_log_returns = pd.Series((sampled * latest_sigma).sum(axis=1))
        log_quantiles = {
            level: float(simulated_log_returns.quantile(1 - level, interpolation="higher"))
            for level in levels
        }

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
            method="filtered_historical_simulation",
            method_variant=f"lambda_{lambda_}",
            sample_start=self.sample_start,
            sample_end=self.sample_end,
            liquidation_days=n_liquidation,
            var_by_confidence=var_by_confidence,
            log_var_by_confidence=log_var_by_confidence,
            parameters={
                "lambda": lambda_,
                "simulations": n_sims,
                "random_seed": self.random_seed,
                "confidence_levels": levels,
            },
            diagnostics={
                "standardized_residuals": int(standardized.shape[0]),
                "latest_daily_sigma": latest_sigma,
            },
        )

