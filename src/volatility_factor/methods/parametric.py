"""Parametric variance-covariance VaR."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from volatility_factor.methods.common import BaseVaRMethod, confidence_levels_or_default
from volatility_factor.results import MethodResult


class ParametricVaR(BaseVaRMethod):
    """Delta-normal variance-covariance VaR using constituent log-return covariance."""

    def run(
        self,
        n_liquidation: int,
        confidence_levels: Sequence[float] | None = None,
        include_mean: bool = False,
    ) -> MethodResult:
        returns = self.constituent_returns
        levels = confidence_levels_or_default(confidence_levels)
        mean_daily = float(self.weights.dot(returns.mean().values)) if include_mean else 0.0
        covariance = returns.cov().values
        variance_daily = float(self.weights.dot(covariance).dot(self.weights))
        if variance_daily < 0:
            raise ValueError("Portfolio variance is negative; covariance matrix is invalid")

        sigma_horizon = float(np.sqrt(variance_daily * n_liquidation))
        mean_horizon = mean_daily * n_liquidation
        log_quantiles = {
            level: float(mean_horizon + stats.norm.ppf(1 - level) * sigma_horizon)
            for level in levels
        }
        return self.result_from_log_quantiles(
            method="parametric_variance_covariance",
            n_liquidation=n_liquidation,
            log_quantiles=log_quantiles,
            parameters={"include_mean": include_mean, "confidence_levels": levels},
            diagnostics={
                "portfolio_daily_mean": mean_daily,
                "portfolio_daily_variance": variance_daily,
                "portfolio_daily_volatility": float(np.sqrt(variance_daily)),
            },
        )
