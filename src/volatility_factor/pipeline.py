"""Config-driven model runs (the loops from ``volatility_factor_main.ipynb``).

Typical use::

    run = run_garch_var().save()          # GARCH VaR for every configured factor
    run.backtest().save()                 # backtest every VaR over its forecast window
    run_circular_bootstrap().save()       # circular block bootstrap VaR
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from volatility_factor.backtesting import BacktestRun, run_backtests
from volatility_factor.config import Config, FactorConfig, get_config
from volatility_factor.data import PricingData, get_data
from volatility_factor.garch import GarchSim
from volatility_factor.nonparametric import NonParametricVar
from volatility_factor.periods import get_stress_range_multi, get_stress_range_single
from volatility_factor.results import to_long

__all__ = [
    "VAR_TYPES",
    "BootstrapRun",
    "GarchRun",
    "run_circular_bootstrap",
    "run_garch_var",
]

#: ``garch_settings.var_type`` value -> (shock type, ``Method`` label prefix).
#: They always run in this order.
VAR_TYPES = {
    "long_term": ("custom_normal", "long_term_var"),
    "normal": ("normal", "normal_var"),
    "short_term": ("custom_bootstrap", "short_term_var"),
    "stress_var": ("custom_bootstrap", "stress_var"),
}
STRESS_WINDOW_DAYS = 126
STRESS_ALPHA = 0.99


def _output_file(output_file: str | os.PathLike | None, config: Config) -> Path:
    path = Path(output_file) if output_file else config.output_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class GarchRun:
    """Output of :func:`run_garch_var`.

    Attributes:
        results: one row per simulated VaR (the wide layout saved to Excel).
        qa: sample-coverage QA failures (factors skipped for a window).
        config: the config used.
        data: the pricing data used.
    """

    results: pd.DataFrame
    qa: pd.DataFrame
    config: Config
    data: PricingData

    @property
    def long(self) -> pd.DataFrame:
        """Results melted to one row per VaR quantile (see :func:`to_long`)."""
        return to_long(self.results)

    def save(self, output_file: str | os.PathLike | None = None) -> GarchRun:
        """Write ``<output_file>.xlsx``, ``<output_file>.parquet`` and ``QA1_<name>.xlsx``."""
        path = _output_file(output_file, self.config)
        self.results.to_excel(f"{path}.xlsx", index=False)
        self.long.to_parquet(f"{path}.parquet", compression="snappy")
        self.qa.to_excel(path.parent / f"QA1_{path.name}.xlsx", index=False)
        return self

    def backtest(self, **kwargs: Any) -> BacktestRun:
        """Backtest these results (see :func:`run_backtests`)."""
        kwargs.setdefault("data", self.data)
        kwargs.setdefault("config", self.config)
        return run_backtests(self.results, **kwargs)


@dataclass(frozen=True)
class BootstrapRun:
    """Output of :func:`run_circular_bootstrap`."""

    results: pd.DataFrame
    config: Config

    def save(self, output_file: str | os.PathLike | None = None) -> BootstrapRun:
        """Write ``circular_bootstrap_results_<date suffix>.xlsx`` next to the output file."""
        path = _output_file(output_file, self.config)
        self.results.to_excel(
            path.parent / f"circular_bootstrap_results_{path.name.split('_')[-1]}.xlsx",
            index=False,
        )
        return self


def _coverage_failure(
    garch: GarchSim, factor: FactorConfig, sample_start: str, sample_end: str
) -> dict[str, str] | None:
    """QA: prices must cover the sample window (allowing four days at each end)."""
    first_price_date = garch.price_matrix.first_valid_index() - pd.offsets.Day(4)
    last_price_date = garch.price_matrix.last_valid_index() + pd.offsets.Day(4)
    if first_price_date > pd.to_datetime(sample_start) or last_price_date < pd.to_datetime(sample_end):
        return {
            "Volatility_Factor_ID": factor.volatility_factor_id,
            "Tickers": garch.ticker_description,
            "SampleStart": sample_start,
            "SampleEnd": sample_end,
            "FirstPriceDate": first_price_date.strftime("%Y-%m-%d"),
            "LastPriceDate": last_price_date.strftime("%Y-%m-%d"),
        }
    return None


def _stress_window(garch: GarchSim) -> list[str]:
    returns = garch.log_return_matrix[garch.sample_start : garch.sample_end].copy()
    if garch.single_index is True:
        return get_stress_range_single(start=garch.sample_start, end=garch.sample_end,
                                       ndays=STRESS_WINDOW_DAYS, returns=returns,
                                       alpha=STRESS_ALPHA)
    return get_stress_range_multi(start=garch.sample_start, end=garch.sample_end,
                                  ndays=STRESS_WINDOW_DAYS, returns=returns,
                                  weights=garch.weights, alpha=STRESS_ALPHA)


def run_garch_var(
    config: Config | str | os.PathLike | None = None,
    data: Any = None,
    *,
    verbose: bool = True,
) -> GarchRun:
    """GARCH simulation VaR for every factor, liquidation period, sample window and
    starting-value quantile in the config.

    For each combination, the configured ``var_type`` values run in this order:

    * ``long_term``: ``custom_normal`` shocks
    * ``normal``: ``normal`` shocks
    * ``short_term``: ``custom_bootstrap`` from the last six months of the sample
    * ``stress_var``: ``custom_bootstrap`` from the most volatile 126-day window

    Starting values come from the last twelve months of the sample at each configured
    quantile. Windows whose prices don't cover the sample are skipped and reported in
    ``qa``.

    Args:
        config: config (path or :class:`Config`); defaults to the active config. Use
            ``Config.with_garch(...)`` / ``with_factors(...)`` to change run sizes.
        data: pricing data; defaults to the config's ``pricing_data``.
        verbose: print progress.
    """
    cfg = get_config(config)
    data = get_data(data, cfg)
    g = cfg.garch
    sample_range, forecast_range = cfg.periods()

    records: list[dict[str, Any]] = []
    qa: list[dict[str, Any]] = []
    for factor in cfg.factors.values():
        models: dict[int, GarchSim] = {}  # fitted once per sample window, then reused
        stress: dict[int, list[str]] = {}
        for nday in factor.ndays:
            horizon2 = g.horizon2 + nday - 1
            for idx, (forecast_start, forecast_end) in enumerate(forecast_range):
                sample_start, sample_end = sample_range[idx]
                if idx not in models:
                    models[idx] = GarchSim(factor.ticker_list, factor.weight_list, sample_start,
                                           sample_end, data=data, config=cfg,
                                           volatility_factor_id=factor.volatility_factor_id)
                garch = models[idx]
                for start_point_quantile in g.start_point_quantiles:
                    failure = _coverage_failure(garch, factor, sample_start, sample_end)
                    if failure is not None:
                        if verbose:
                            print(f"{factor.ticker_list} failed QA check: "
                                  f"{sample_start} to {sample_end}")
                        qa.append(failure)
                        continue

                    start_point_date = (
                        pd.to_datetime(sample_end) - pd.offsets.MonthBegin(12)
                    ).strftime("%Y-%m-%d")
                    start_vals = [start_point_date, start_point_quantile]
                    for var_type, (shock_type, label) in VAR_TYPES.items():
                        if var_type not in g.var_type:
                            continue
                        boot: dict[str, str] = {}
                        if var_type == "short_term":
                            boot = {
                                "bootstrap_start": (
                                    pd.to_datetime(sample_end) - pd.offsets.MonthBegin(6)
                                ).strftime("%Y-%m-%d"),
                                "bootstrap_end": sample_end,
                            }
                        elif var_type == "stress_var":
                            if idx not in stress:
                                stress[idx] = _stress_window(garch)
                            boot = {"bootstrap_start": stress[idx][0],
                                    "bootstrap_end": stress[idx][1]}
                        result = garch.simulate(
                            horizon1=g.horizon1, horizon2=horizon2, ndays=nday, nsims=g.nsims,
                            shock_type=shock_type, start_vals=start_vals, **boot,
                        ).with_metadata(
                            Volatility_Factor_ID=factor.volatility_factor_id,
                            ForecastStart=forecast_start,
                            ForecastEnd=forecast_end,
                            Method=f"{label}_{round(start_point_quantile * 100)}",
                        )
                        records.append(result.to_dict())

                    if verbose:
                        print(f"{factor.ticker_list}: n-days: {nday} | starting quantile: "
                              f"{start_point_quantile} | sample: {sample_start} to "
                              f"{sample_end} -- Completed")

    return GarchRun(
        results=pd.DataFrame.from_dict(records),
        qa=pd.DataFrame.from_dict(qa),
        config=cfg,
        data=data,
    )


def run_circular_bootstrap(
    config: Config | str | os.PathLike | None = None,
    data: Any = None,
    *,
    nboot: int = 10_000,
    var_p: float = 0.99,
) -> BootstrapRun:
    """Circular block bootstrap VaR for every factor and liquidation period, using the
    most recent sample window only."""
    cfg = get_config(config)
    data = get_data(data, cfg)
    sample_start, sample_end = cfg.periods()[0][0]

    rows = []
    for factor in cfg.factors.values():
        model = NonParametricVar(factor.ticker_list, factor.weight_list, sample_start,
                                 sample_end, data=data, config=cfg,
                                 volatility_factor_id=factor.volatility_factor_id)
        for nday in factor.ndays:
            vol = model.circular_block_bootstrap(n_liquidation=nday, nboot=nboot, var_p=var_p)
            vol.update({"Volatility_Factor_ID": factor.volatility_factor_id})
            rows.append(vol)
    return BootstrapRun(results=pd.DataFrame.from_dict(rows), config=cfg)
