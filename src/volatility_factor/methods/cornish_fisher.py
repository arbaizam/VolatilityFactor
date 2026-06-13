"""Cornish-Fisher modified VaR."""

from __future__ import annotations

from typing import Sequence

from scipy import stats

from volatility_factor.methods.common import BaseVaRMethod, confidence_levels_or_default
from volatility_factor.results import MethodResult


class CornishFisherVaR(BaseVaRMethod):
    """Modified VaR using skewness and excess kurtosis adjustments."""

    def run(
        self,
        n_liquidation: int,
        confidence_levels: Sequence[float] | None = None,
    ) -> MethodResult:
        returns = self.horizon_portfolio_returns(n_liquidation)
        if returns.shape[0] < 4:
            raise ValueError("At least four observations are required for Cornish-Fisher VaR")

        levels = confidence_levels_or_default(confidence_levels)
        mean = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        skewness = float(stats.skew(returns, bias=False))
        excess_kurtosis = float(stats.kurtosis(returns, fisher=True, bias=False))

        log_quantiles = {}
        for level in levels:
            z = float(stats.norm.ppf(1 - level))
            z_cf = (
                z
                + ((z**2 - 1) * skewness / 6)
                + ((z**3 - 3 * z) * excess_kurtosis / 24)
                - ((2 * z**3 - 5 * z) * skewness**2 / 36)
            )
            log_quantiles[level] = float(mean + z_cf * sigma)

        return self.result_from_log_quantiles(
            method="cornish_fisher",
            n_liquidation=n_liquidation,
            log_quantiles=log_quantiles,
            parameters={"confidence_levels": levels},
            diagnostics={
                "observations": int(returns.shape[0]),
                "mean": mean,
                "standard_deviation": sigma,
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
            },
        )

