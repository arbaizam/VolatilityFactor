"""Quarterly performance monitoring: F-tests, exception counts and shortfalls.

These functions hold the logic from ``monitoring_main.ipynb``. Each returns a DataFrame
in the same layout the notebook saved to Excel.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from volatility_factor.backtesting import VarBacktest, log_likelihood_test
from volatility_factor.config import FactorConfig, get_config
from volatility_factor.data import VarData, get_data
from volatility_factor.garch import GarchSim

__all__ = [
    "EXCEPTION_ADJUSTMENT",
    "MonitoringPeriods",
    "f_monitor",
    "f_test_report",
    "nday_exceptions",
    "one_day_exceptions",
    "shortfall_report",
]

# Exceptions removed when the independence test rejects (keyed by total exceptions).
# Copied verbatim from the monitoring notebook. Note: the entries for 28-30 (1) and 36 (36)
# look like typos for 10 and 13; they are kept as-is pending confirmation.
EXCEPTION_ADJUSTMENT = {
    0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 4, 12: 4, 13: 4,
    14: 5, 15: 5, 16: 5, 17: 5, 18: 6, 19: 6, 20: 6, 21: 7, 22: 7, 23: 8, 24: 8, 25: 8,
    26: 9, 27: 9, 28: 1, 29: 1, 30: 1, 31: 11, 32: 11, 33: 12, 34: 12, 35: 13, 36: 36,
    37: 14, 38: 14, 39: 15, 40: 15,
}  # fmt: skip
MAX_EXCEPTION_ADJUSTMENT = 15  # used above 40 exceptions

Window = tuple[str, str]


@dataclass(frozen=True)
class MonitoringPeriods:
    """The sample window, its last year, and the quarters that follow it.

    Example::

        p = MonitoringPeriods.from_sample_end("2024-12-31", sample_years=9)
        p.full        # ('2016-01-01', '2024-12-31')
        p.start       # ('2024-01-01', '2024-12-31')
        p.quarter(1)  # ('2025-01-01', '2025-03-31')
    """

    full: Window
    start: Window
    quarters: tuple[Window, ...]

    @classmethod
    def from_sample_end(
        cls, sample_end: str, sample_years: int = 9, n_quarters: int = 5
    ) -> MonitoringPeriods:
        end = pd.Timestamp(sample_end)
        fmt = "%Y-%m-%d"
        full = ((end - pd.DateOffset(years=sample_years) + pd.Timedelta(days=1)).strftime(fmt),
                end.strftime(fmt))
        start = ((end - pd.DateOffset(years=1) + pd.Timedelta(days=1)).strftime(fmt),
                 end.strftime(fmt))
        quarters = []
        for k in range(n_quarters):
            q_start = end + pd.Timedelta(days=1) + pd.DateOffset(months=3 * k)
            q_end = q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
            quarters.append((q_start.strftime(fmt), q_end.strftime(fmt)))
        return cls(full=full, start=start, quarters=tuple(quarters))

    @classmethod
    def from_config(cls, config: Any = None, n_quarters: int = 5) -> MonitoringPeriods:
        """Use ``garch_settings.last_sample_date`` and ``sample_years``."""
        g = get_config(config).garch
        return cls.from_sample_end(g.last_sample_date, g.sample_years, n_quarters)

    def quarter(self, k: int) -> Window:
        """The ``k``-th quarter after the sample (1-based)."""
        return self.quarters[k - 1]


def _factor_weights(factor: FactorConfig, use_config_weights: bool) -> list[float] | None:
    return factor.weight_list if use_config_weights else None


# --------------------------------------------------------------------------- #
# F-test
# --------------------------------------------------------------------------- #
def f_monitor(
    tickers: str | Sequence[str],
    period_full: Window,
    period_start: Window,
    period_new: Window,
    alpha: float = 0.99,
    *,
    weights: Sequence[float] | None = None,
    data: Any = None,
) -> dict[str, float]:
    """F-statistics comparing out-of-sample variance with in-sample variance.

    Returns:
        ``Q-vs-Full`` (new period vs full sample), ``Q-vs-1Y`` (vs last year of sample),
        ``Q-vs-Max`` (vs the largest rolling variance of the same length in the sample),
        each with its ``_CV`` critical value at ``alpha``.
    """
    df = VarData(tickers, weights, data=data).wtd_log_return

    full = df.loc[period_full[0] : period_full[1]]
    start = df.loc[period_start[0] : period_start[1]]
    new = df.loc[period_new[0] : period_new[1]]

    return {
        "Q-vs-Full": new.values.var(ddof=1) / full.values.var(ddof=1),
        "Q-vs-Full_CV": scipy.stats.f.ppf(q=alpha, dfn=len(new) - 1, dfd=len(full) - 1),
        "Q-vs-1Y": new.values.var(ddof=1) / start.values.var(ddof=1),
        "Q-vs-1Y_CV": scipy.stats.f.ppf(q=alpha, dfn=len(new) - 1, dfd=len(start) - 1),
        "Q-vs-Max": new.values.var(ddof=1)
        / full.rolling(len(new)).var().dropna().quantile(1).values[0],
        "Q-vs-Max_CV": scipy.stats.f.ppf(q=alpha, dfn=len(new) - 1, dfd=len(new) - 1),
    }


def f_test_report(
    periods: MonitoringPeriods,
    quarters: Iterable[int] = (1, 2, 3),
    *,
    alpha: float = 0.99,
    config: Any = None,
    data: Any = None,
    use_config_weights: bool = True,
) -> pd.DataFrame:
    """F-statistics for every configured factor and quarter (columns prefixed by quarter).

    Args:
        use_config_weights: weight portfolios as configured. The original notebook used
            equal weights here; pass ``False`` to reproduce that.
    """
    cfg = get_config(config)
    data = get_data(data, cfg)
    frames = []
    for qtr in quarters:
        results = []
        for factor in cfg.factors.values():
            stats = f_monitor(
                factor.ticker_list, periods.full, periods.start, periods.quarter(qtr), alpha,
                weights=_factor_weights(factor, use_config_weights), data=data,
            )
            results.append({"ID": factor.volatility_factor_id, **stats})
        df = pd.DataFrame.from_dict(results)
        df.columns = [str(qtr) + col if col != "ID" else col for col in df.columns]
        frames.append(df.set_index("ID"))
    return pd.concat(frames, axis=1)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
def _monitoring_var(
    factor_or_tickers: FactorConfig | str | Sequence[str],
    weights: Sequence[float] | None,
    ndays: int,
    periods: MonitoringPeriods,
    start_vals: Sequence,
    *,
    horizon1: int,
    horizon2: int,
    nsims: int,
    var_level: float,
    data: Any,
) -> float:
    tickers = (
        factor_or_tickers.ticker_list
        if isinstance(factor_or_tickers, FactorConfig)
        else factor_or_tickers
    )
    result = GarchSim(tickers, weights, *periods.full, data=data).simulate(
        horizon1=horizon1, horizon2=horizon2 + ndays - 1, ndays=ndays, nsims=nsims,
        shock_type="custom_normal", start_vals=start_vals,
    )
    return result.var(var_level)


def one_day_exceptions(
    periods: MonitoringPeriods,
    quarters: Iterable[int] = (1, 2, 3, 4),
    *,
    horizon1: int = 63,
    horizon2: int = 252,
    nsims: int = 10_000,
    start_quantile: float = 0.50,
    var_level: float = 0.95,
    config: Any = None,
    data: Any = None,
    use_config_weights: bool = True,
) -> pd.DataFrame:
    """One-day VaR exceptions per factor, counted by quarter.

    Args:
        use_config_weights: backtest with the configured weights (the VaR always uses
            them). The original notebook backtested with equal weights; pass ``False`` to
            reproduce that.
    """
    cfg = get_config(config)
    data = get_data(data, cfg)
    quarters = list(quarters)
    rows = []
    for factor in cfg.factors.values():
        nday_var = _monitoring_var(
            factor, factor.weight_list, 1, periods, [periods.start[0], start_quantile],
            horizon1=horizon1, horizon2=horizon2, nsims=nsims, var_level=var_level,
            data=data,
        )
        base = VarData(factor.ticker_list, _factor_weights(factor, use_config_weights), data=data)
        counts = [
            base.backtest(1, var=nday_var, start_date=periods.quarter(q)[0],
                          end_date=periods.quarter(q)[1], var_is_log=False).break_series.sum()
            for q in quarters
        ]
        row = {"ID": factor.volatility_factor_id, "1-Day_VaR": nday_var}
        row.update({f"Exception_Q{q}": c for q, c in zip(quarters, counts, strict=True)})
        row["Total_Exceptions"] = np.sum(counts)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c.startswith("Exception_Q")] + ["Total_Exceptions"]:
        df[col] = df[col].astype(np.int64)
    return df


def exception_adjustment(total_exceptions: int, independent: bool) -> int:
    """Exceptions to remove for clustering (0 when the independence test does not reject)."""
    if independent:
        return 0
    if total_exceptions > 40:
        return MAX_EXCEPTION_ADJUSTMENT
    return EXCEPTION_ADJUSTMENT.get(total_exceptions)


def nday_exceptions(
    periods: MonitoringPeriods,
    quarter: int = 3,
    *,
    horizon1: int = 63,
    horizon2: int = 252,
    nsims: int = 10_000,
    start_quantile: float = 0.50,
    var_level: float = 0.95,
    config: Any = None,
    data: Any = None,
    use_config_weights: bool = True,
) -> pd.DataFrame:
    """N-day exceptions from the start of quarter 2 through the end of ``quarter``.

    Includes the independence test and the clustering adjustment from
    :data:`EXCEPTION_ADJUSTMENT`. See :func:`one_day_exceptions` for
    ``use_config_weights``.
    """
    cfg = get_config(config)
    data = get_data(data, cfg)
    window = (periods.quarter(2)[0], periods.quarter(quarter)[1])
    frames = []
    for factor in cfg.factors.values():
        for ndays in factor.ndays:
            nday_var = _monitoring_var(
                factor, factor.weight_list, ndays, periods, [periods.start[0], start_quantile],
                horizon1=horizon1, horizon2=horizon2, nsims=nsims, var_level=var_level,
                data=data,
            )
            bt = VarBacktest(factor.ticker_list, ndays,
                             weights=_factor_weights(factor, use_config_weights),
                             start_date=window[0], end_date=window[1], var=nday_var,
                             var_is_log=False, data=data)
            ind = log_likelihood_test(bt.break_series)
            total = bt.break_series.sum()
            adjustment = exception_adjustment(total, ind.get("Independence") != "RejectNull")
            frames.append(pd.DataFrame({
                "Volatility_Factor_ID": factor.volatility_factor_id,
                "Tickers": ", ".join(factor.tickers) if len(factor.tickers) > 1 else factor.ticker_list,
                "Weights": ", ".join(str(x) for x in factor.weights) if factor.weights is not None else "1",
                "NDays": ndays,
                "NDayVaR": nday_var,
                "ForecastStart": bt.start_date,
                "ForecastEnd": bt.end_date,
                "TotalExceptions": total,
                "IndependenceTest": ind.get("Independence"),
                "IndependenceAdjustment": adjustment,
                "TotalExceptionsAdjusted": total - adjustment,
            }, index=[0]))  # fmt: skip
    return pd.concat(frames)


def shortfall_report(
    tickers: str | Sequence[str],
    periods: MonitoringPeriods,
    quarter: int = 3,
    *,
    weights: Sequence[float] | None = None,
    ndays: int = 10,
    start_vals: Sequence = ("2022-01-01", 0.50),
    horizon1: int = 63,
    horizon2: int = 252,
    nsims: int = 10_000,
    var_level: float = 0.95,
    data: Any = None,
) -> pd.DataFrame:
    """Daily shortfalls from the start of quarter 2 through the end of ``quarter``."""
    data = get_data(data)
    nday_var = _monitoring_var(
        tickers, weights, ndays, periods, list(start_vals), horizon1=horizon1,
        horizon2=horizon2, nsims=nsims, var_level=var_level, data=data,
    )
    bt = VarBacktest(tickers, ndays, weights=weights, start_date=periods.quarter(2)[0],
                     end_date=periods.quarter(quarter)[1], var=nday_var, var_is_log=False,
                     data=data)
    return bt.shortfall_series.to_frame(name=bt.ticker_description)
