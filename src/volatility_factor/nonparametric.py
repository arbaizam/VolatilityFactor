"""Non-parametric VaR: circular block bootstrap."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from volatility_factor._utils import deprecated_alias, validate_date
from volatility_factor.data import VarData

__all__ = ["NonParametricVar"]


class NonParametricVar(VarData):
    """Non-parametric VaR from the weighted log returns in a sample window.

    Example::

        NonParametricVar.from_id("agy_cmo", sample_start="2016-01-01", sample_end="2024-12-31") \\
            .circular_block_bootstrap(n_liquidation=10, nboot=10_000)
    """

    def __init__(
        self,
        tickers: str | Sequence[str],
        weights: Sequence[float] | float | None,
        sample_start: str,
        sample_end: str,
        **kwargs: Any,
    ):
        super().__init__(tickers, weights, **kwargs)
        self.sample_start = sample_start
        self.sample_end = sample_end

    @property
    def sample_start(self) -> str:
        return self._sample_start

    @sample_start.setter
    def sample_start(self, value: str) -> None:
        self._sample_start = validate_date(value, "sample_start")

    @property
    def sample_end(self) -> str:
        return self._sample_end

    @sample_end.setter
    def sample_end(self, value: str) -> None:
        self._sample_end = validate_date(value, "sample_end")

    def with_sample(self, sample_start: str, sample_end: str) -> NonParametricVar:
        """Change the sample window and return ``self``."""
        self.sample_start = sample_start
        self.sample_end = sample_end
        return self

    @property
    def training_data(self) -> pd.DataFrame:
        """Weighted log returns in the sample window."""
        return self.wtd_log_return[self.sample_start : self.sample_end]

    @staticmethod
    def _get_boot_quantile(boot_i: np.ndarray, n_liquidation: int, var_p: float) -> pd.Series:
        """Arithmetic VaR (decimal) of one bootstrap sample."""
        df = pd.DataFrame(boot_i)
        pc_nday = df.rolling(n_liquidation).sum().dropna()
        var = pc_nday.quantile(q=1 - var_p, interpolation="higher")
        var = 1 - np.exp(var)
        return var

    def circular_block_bootstrap(
        self, n_liquidation: int, nboot: int = 1000, var_p: float = 0.99, random_seed: int = 1
    ) -> dict[str, Any]:
        """VaR from a circular block bootstrap of the sample returns.

        Blocks have length ``floor(sqrt(n))`` and wrap around the end of the sample.
        Each bootstrap sample is built from randomly chosen blocks, trimmed to ``n``
        observations; its ``n_liquidation``-day VaR is computed, and the mean across
        ``nboot`` samples is returned (in percent).

        Args:
            n_liquidation: liquidation horizon in days.
            nboot: number of bootstrap samples.
            var_p: VaR confidence level.
            random_seed: seed for block selection. Defaults to 1.
        """
        sample = np.append(self.training_data, self.training_data)
        sample_size = len(self.training_data)
        block_size = int(np.floor(np.sqrt(sample_size)))
        block_n = int(np.ceil(sample_size / block_size))

        sample_block = np.empty([sample_size, block_size])
        for i in range(sample_size):
            sample_block[i,] = sample[i : i + block_size]

        random_state = np.random.RandomState(random_seed)
        boot_index = random_state.randint(sample_size, size=(nboot, block_n))
        boot_all = sample_block[boot_index]
        boot_array = boot_all.reshape((nboot, (boot_all.shape[1] * boot_all.shape[2])))[
            :, :sample_size
        ]

        quantile_boot_all = np.apply_along_axis(
            self._get_boot_quantile, axis=1, arr=boot_array, n_liquidation=n_liquidation,
            var_p=var_p,
        )
        quantile_boot_mean = quantile_boot_all.mean() * 100

        return {
            "Tickers": self.ticker_description,
            "Weights": self.weight_description,
            "Method": "CircularBootstrap",
            "SampleStart": self.sample_start,
            "SampleEnd": self.sample_end,
            "VaRp": str(round(var_p * 100)) + "Q",
            "NDays": n_liquidation,
            "BootstrapReplications": nboot,
            "VaR": quantile_boot_mean,
        }

    @deprecated_alias("circular_block_bootstrap")
    def CircularBlockBootstrap(  # noqa: N802 - legacy name
        self, n_liquidation: int, nboot: int = 1000, VaRp: float = 0.99  # noqa: N803
    ) -> dict[str, Any]:
        return self.circular_block_bootstrap(n_liquidation, nboot=nboot, var_p=VaRp)
