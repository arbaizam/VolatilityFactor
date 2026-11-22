"""Config-driven model runs: every configured factor, window and setting in one call.

Typical use::

    run = run_garch_var().save()          # GARCH VaR for every configured factor
    run.backtest().save()                 # backtest every VaR over its forecast window
    run_circular_bootstrap().save()       # circular block bootstrap VaR
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from volatility_factor.backtesting import BacktestRun, run_backtests
from volatility_factor.config import Config, FactorConfig, get_config
from volatility_factor.data import PricingData, get_data
from volatility_factor.garch import GarchSim
from volatility_factor.nonparametric import NonParametricVar
from volatility_factor.results import to_long


def _output_file(output_file: str | os.PathLike | None, config: Config) -> Path:
    """The output file stem (given, or from the config), creating its folder."""
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
        """Write the Excel and parquet results and the sample-coverage workbook.

        Raises:
            ValueError: when there are no results. The QA workbook records the run.
        """
        path = _output_file(output_file, self.config)
        qa_file = path.parent / f"QA1_{path.name}.xlsx"
        with TemporaryDirectory(dir=path.parent) as directory:
            staged = Path(directory)
            qa_stage = staged / qa_file.name
            self.qa.to_excel(qa_stage, index=False)
            if self.results.empty:
                qa_stage.replace(qa_file)
                raise ValueError(f"No VaR results to publish. Sample coverage report: {qa_file}")

            excel_stage = staged / f"{path.name}.xlsx"
            parquet_stage = staged / f"{path.name}.parquet"
            self.results.to_excel(excel_stage, index=False)
            self.long.to_parquet(parquet_stage, compression="snappy")
            excel_stage.replace(path.parent / excel_stage.name)
            parquet_stage.replace(path.parent / parquet_stage.name)
            qa_stage.replace(qa_file)
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
    """Require aligned prices at the sample boundaries and usable returns in the sample."""
    prices = garch.price_matrix
    aligned = (np.isfinite(prices) & prices.gt(0)).all(axis=1) & prices.index.notna()
    dates = prices.index[aligned]
    allowance = pd.offsets.Day(4)
    first_price_date = last_price_date = ""
    if len(dates):
        first = dates[0] - allowance
        last = dates[-1] + allowance
        start, end = pd.Timestamp(sample_start), pd.Timestamp(sample_end)
        in_sample = (prices.index >= start) & (prices.index <= end)
        has_returns = (aligned & aligned.shift(fill_value=False) & in_sample).any()
        if first <= start and last >= end and has_returns:
            return None
        first_price_date, last_price_date = first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")
    return {
        "Volatility_Factor_ID": factor.volatility_factor_id,
        "Tickers": garch.ticker_description,
        "SampleStart": sample_start,
        "SampleEnd": sample_end,
        "FirstPriceDate": first_price_date,
        "LastPriceDate": last_price_date,
    }


def run_garch_var(
    config: Config | str | os.PathLike | None = None,
    data: Any = None,
    *,
    verbose: bool = True,
) -> GarchRun:
    """GARCH simulation VaR for every factor, liquidation period, sample window and
    starting-value quantile in the config.

    Simulations use ``custom_normal`` shocks and starting values from the last twelve
    months of the sample at each configured quantile. Windows with incomplete sample
    coverage are skipped and reported in ``qa``.

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
                    result = garch.simulate(
                        horizon1=g.horizon1, horizon2=horizon2, ndays=nday, nsims=g.nsims,
                        shock_type="custom_normal", start_vals=start_vals,
                    ).with_metadata(
                        Volatility_Factor_ID=factor.volatility_factor_id,
                        ForecastStart=forecast_start,
                        ForecastEnd=forecast_end,
                        Method=f"long_term_var_{round(start_point_quantile * 100)}",
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
