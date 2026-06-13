"""Historical and circular block bootstrap VaR methods."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from volatility_factor.methods.common import (
    BaseVaRMethod,
    confidence_levels_or_default,
    historical_log_quantiles,
)
from volatility_factor.results import MethodResult


class HistoricalSimulationVaR(BaseVaRMethod):
    """Empirical historical simulation VaR using rolling n-day log returns."""

    def run(
        self,
        n_liquidation: int,
        confidence_levels: Sequence[float] | None = None,
    ) -> MethodResult:
        horizon_returns = self.horizon_portfolio_returns(n_liquidation)
        log_quantiles = historical_log_quantiles(horizon_returns, confidence_levels)
        return self.result_from_log_quantiles(
            method="historical_simulation",
            n_liquidation=n_liquidation,
            log_quantiles=log_quantiles,
            parameters={"confidence_levels": confidence_levels_or_default(confidence_levels)},
            diagnostics={"observations": int(horizon_returns.shape[0])},
        )


class CircularBlockBootstrapVaR(BaseVaRMethod):
    """Circular block bootstrap VaR ported from the legacy implementation."""

    def __init__(self, *args, random_seed: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = random_seed

    @staticmethod
    def _get_boot_quantile(boot_i: np.ndarray, n_liquidation: int, VaRp: float) -> float:
        df = pd.DataFrame(boot_i)
        pc_nday = df.rolling(n_liquidation).sum().dropna()
        var = pc_nday.quantile(q=1 - VaRp, interpolation="higher")
        return float(1 - np.exp(var.iloc[0]))

    def _bootstrap_array(self, nboot: int) -> tuple[np.ndarray, int, int, int]:
        training_data = self.portfolio_returns
        sample = np.append(training_data, training_data)
        sample_size = len(training_data)
        block_size = int(np.floor(np.sqrt(sample_size)))
        block_n = int(np.ceil(sample_size / block_size))

        sample_block = np.empty([sample_size, block_size])
        for i in range(sample_size):
            sample_block[i,] = sample[i : i + block_size]

        random_state = np.random.RandomState(self.random_seed)
        boot_index = random_state.randint(sample_size, size=(nboot, block_n))
        boot_all = sample_block[boot_index]
        boot_array = boot_all.reshape((nboot, (boot_all.shape[1] * boot_all.shape[2])))[
            :, :sample_size
        ]
        return boot_array, sample_size, block_size, block_n

    def circular_block_bootstrap(
        self,
        n_liquidation: int,
        nboot: int = 1000,
        VaRp: float = 0.99,
    ) -> dict:
        boot_array, _, _, _ = self._bootstrap_array(nboot)
        quantile_boot_all = np.apply_along_axis(
            self._get_boot_quantile,
            axis=1,
            arr=boot_array,
            n_liquidation=n_liquidation,
            VaRp=VaRp,
        )
        quantile_boot_mean = float(quantile_boot_all.mean() * 100)
        return {
            "Tickers": self.ticker_description,
            "Weights": "1" if self.single_index else ";".join(str(x) for x in self.weights),
            "Method": "CircularBootstrap",
            "SampleStart": self.sample_start,
            "SampleEnd": self.sample_end,
            "VaRp": str(round(VaRp * 100)) + "Q",
            "NDays": n_liquidation,
            "BootstrapReplications": nboot,
            "VaR": quantile_boot_mean,
        }

    def run(
        self,
        n_liquidation: int,
        nboot: int = 1000,
        confidence_levels: Sequence[float] | None = None,
    ) -> MethodResult:
        levels = confidence_levels_or_default(confidence_levels)
        boot_array, sample_size, block_size, block_n = self._bootstrap_array(nboot)
        var_by_confidence = {}
        log_var_by_confidence = {}
        for level in levels:
            boot_vars = np.apply_along_axis(
                self._get_boot_quantile,
                axis=1,
                arr=boot_array,
                n_liquidation=n_liquidation,
                VaRp=level,
            )
            var_pct = float(boot_vars.mean() * 100)
            var_by_confidence[level] = var_pct
            log_var_by_confidence[level] = float(-np.log(1 - var_pct / 100) * 100)

        return MethodResult(
            volatility_factor_id=self.volatility_factor_id,
            tickers=self.ticker_description,
            weights=self.weight_description,
            method="circular_block_bootstrap",
            sample_start=self.sample_start,
            sample_end=self.sample_end,
            liquidation_days=n_liquidation,
            var_by_confidence=var_by_confidence,
            log_var_by_confidence=log_var_by_confidence,
            parameters={
                "bootstrap_replications": nboot,
                "confidence_levels": levels,
                "random_seed": self.random_seed,
            },
            diagnostics={
                "sample_size": sample_size,
                "block_size": block_size,
                "block_count": block_n,
            },
        )

    # Backward-compatible alias for current notebooks and regression harnesses.
    CircularBlockBootstrap = circular_block_bootstrap

