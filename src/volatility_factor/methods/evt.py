"""Extreme value theory VaR using peaks over threshold."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from volatility_factor.methods.common import BaseVaRMethod, confidence_levels_or_default
from volatility_factor.results import MethodResult, arithmetic_var_from_log_quantile, log_var_from_log_quantile


class EVTPeaksOverThresholdVaR(BaseVaRMethod):
    """EVT VaR using a generalized Pareto fit to tail losses."""

    def run(
        self,
        n_liquidation: int,
        confidence_levels: Sequence[float] | None = None,
        threshold_quantile: float = 0.95,
        min_exceedances: int = 20,
    ) -> MethodResult:
        if not 0 < threshold_quantile < 1:
            raise ValueError("threshold_quantile must be between 0 and 1")

        levels = confidence_levels_or_default(confidence_levels)
        horizon_returns = self.horizon_portfolio_returns(n_liquidation)
        losses = -horizon_returns
        threshold = float(losses.quantile(threshold_quantile, interpolation="higher"))
        exceedances = losses[losses > threshold] - threshold
        if exceedances.shape[0] < min_exceedances:
            raise ValueError(
                f"EVT requires at least {min_exceedances} exceedances; got {exceedances.shape[0]}"
            )

        shape, loc, scale = stats.genpareto.fit(exceedances.values, floc=0)
        n_obs = losses.shape[0]
        n_exceed = exceedances.shape[0]
        threshold_tail_probability = n_exceed / n_obs

        log_quantiles = {}
        fallback_levels = []
        for level in levels:
            target_tail_probability = 1 - level
            if target_tail_probability >= threshold_tail_probability:
                loss_var = float(losses.quantile(level, interpolation="higher"))
                fallback_levels.append(level)
            else:
                ratio = threshold_tail_probability / target_tail_probability
                if abs(shape) < 1e-8:
                    loss_var = threshold + scale * np.log(ratio)
                else:
                    loss_var = threshold + (scale / shape) * (ratio**shape - 1)
            log_quantiles[level] = float(-loss_var)

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
            method="evt_peaks_over_threshold",
            method_variant=f"threshold_{threshold_quantile}",
            sample_start=self.sample_start,
            sample_end=self.sample_end,
            liquidation_days=n_liquidation,
            var_by_confidence=var_by_confidence,
            log_var_by_confidence=log_var_by_confidence,
            parameters={
                "threshold_quantile": threshold_quantile,
                "min_exceedances": min_exceedances,
                "confidence_levels": levels,
            },
            diagnostics={
                "observations": int(n_obs),
                "threshold": threshold,
                "exceedances": int(n_exceed),
                "gpd_shape": float(shape),
                "gpd_location": float(loc),
                "gpd_scale": float(scale),
                "fallback_historical_levels": fallback_levels,
            },
        )

