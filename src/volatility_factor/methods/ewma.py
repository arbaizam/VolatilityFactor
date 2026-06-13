"""EWMA / RiskMetrics VaR."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from volatility_factor.methods.common import BaseVaRMethod, confidence_levels_or_default
from volatility_factor.results import MethodResult


def ewma_covariance_matrix(returns, lambda_: float) -> np.ndarray:
    """Compute the terminal EWMA covariance matrix using zero-mean returns."""

    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1")
    if returns.empty:
        raise ValueError("returns cannot be empty")

    covariance = returns.cov().values
    for row in returns.values:
        vector = row.reshape(-1, 1)
        covariance = lambda_ * covariance + (1 - lambda_) * vector.dot(vector.T)
    return covariance


class EWMAVaR(BaseVaRMethod):
    """RiskMetrics-style EWMA VaR with default lambda 0.94."""

    def run(
        self,
        n_liquidation: int,
        confidence_levels: Sequence[float] | None = None,
        lambda_: float = 0.94,
        include_mean: bool = False,
    ) -> MethodResult:
        returns = self.constituent_returns
        levels = confidence_levels_or_default(confidence_levels)
        covariance = ewma_covariance_matrix(returns, lambda_)
        variance_daily = float(self.weights.dot(covariance).dot(self.weights))
        if variance_daily < 0:
            raise ValueError("Portfolio variance is negative; covariance matrix is invalid")

        mean_daily = float(self.weights.dot(returns.mean().values)) if include_mean else 0.0
        sigma_horizon = float(np.sqrt(variance_daily * n_liquidation))
        mean_horizon = mean_daily * n_liquidation
        log_quantiles = {
            level: float(mean_horizon + stats.norm.ppf(1 - level) * sigma_horizon)
            for level in levels
        }
        return self.result_from_log_quantiles(
            method="ewma",
            method_variant=f"lambda_{lambda_}",
            n_liquidation=n_liquidation,
            log_quantiles=log_quantiles,
            parameters={
                "lambda": lambda_,
                "include_mean": include_mean,
                "confidence_levels": levels,
            },
            diagnostics={
                "portfolio_daily_mean": mean_daily,
                "portfolio_daily_variance": variance_daily,
                "portfolio_daily_volatility": float(np.sqrt(variance_daily)),
            },
        )

