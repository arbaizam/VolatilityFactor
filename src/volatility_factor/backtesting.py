"""VaR backtesting: exceptions, shortfalls, and the Kupiec / Christoffersen tests."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.stats

from volatility_factor._utils import validate_date
from volatility_factor.config import get_config
from volatility_factor.data import VarData, get_data
from volatility_factor.results import LOG_VAR_COLUMNS, VAR_COLUMNS

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["BacktestRun", "VarBacktest", "log_likelihood_test", "run_backtests"]

_UNSET = object()


def log_likelihood_test(
    break_ser: Sequence[int] | pd.Series, converge_rate: float = 0.01, alpha: float = 0.99
) -> dict[str, Any]:
    """Kupiec (1995) unconditional coverage and Christoffersen (1998) independence tests.

    Args:
        break_ser: 1 for an exception (break), 0 otherwise, in date order.
        converge_rate: expected exception rate under the null.
        alpha: confidence level for the chi-squared critical values.

    Returns:
        Dict with ``Exceptions``, ``LR_uc``, ``LR_ind``, ``LR_cc``, the chi-squared
        critical values, ``Independence`` (``"RejectNull"`` or ``None``), ``alpha`` and
        ``converge_rate`` (both in percent).

    Note:
        This reproduces the original implementation exactly, including two details that
        differ from the textbook test: the first observation is counted once more in the
        transition counts, and ``pi0 = T01 / (T - N)``.
    """
    values = np.asarray(break_ser)
    T = len(values)
    N = values.sum()

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")

        # Kupiec (1995)
        if N == T:
            LR_uc = float("inf")
        else:
            LR_uc = -2 * np.log((1 - converge_rate) ** (T - N) * (converge_rate**N)) + 2 * np.log(
                (1 - N / T) ** (T - N) * (N / T) ** N
            )

        # Christoffersen (1998)
        T00, T01, T10, T11 = 0, 0, 0, 0
        for i in range(T - 1):
            if i == 0:
                if values[0] == 0:
                    T00 = T00 + 1
                else:
                    T01 = T01 + 1
            if values[i] == 0:
                if values[i + 1] == 0:
                    T00 = T00 + 1
                else:
                    T01 = T01 + 1
            elif values[i] == 1:
                if values[i + 1] == 0:
                    T10 = T10 + 1
                else:
                    T11 = T11 + 1

        if N == T:
            LR_ind = float("inf")
        elif N == 0:
            LR_ind = 0
        else:
            pi0 = 1 if T00 == 0 else T01 / (T - N)
            pi = N / T
            pi1 = T11 / N
            LR_ind = -2 * np.log((1 - pi) ** (T00 + T10) * pi ** (T01 + T11)) + 2 * np.log(
                (1 - pi0) ** T00 * pi0**T01 * (1 - pi1) ** T10 * pi1**T11
            )

        LR_cc = LR_uc + LR_ind

    chi2_1 = scipy.stats.chi2.ppf(alpha, 1)
    chi2_2 = scipy.stats.chi2.ppf(alpha, 2)

    return {
        "Exceptions": values.sum(),
        "LR_uc": round(LR_uc, 3),
        "LR_ind": round(LR_ind, 3),
        "LR_cc": round(LR_cc, 3),
        f"chi2(1) {int(alpha * 100)}%": round(chi2_1, 3),
        f"chi2(2) {int(alpha * 100)}%": round(chi2_2, 3),
        "Independence": "RejectNull" if LR_ind > chi2_1 else None,
        "alpha": round(alpha * 100, 1),
        "converge_rate": round(converge_rate * 100, 1),
    }


def _setting(name: str, validator=None) -> property:
    """A property that validates its value and clears cached backtest results."""
    private = f"_{name}"

    def getter(self):
        return getattr(self, private)

    def setter(self, value):
        if validator is not None:
            value = validator(value, name)
        setattr(self, private, value)
        self._reset_backtest()

    return property(getter, setter)


def _optional_date(value, name):
    return validate_date(value, name, optional=True)


class VarBacktest(VarData):
    """Backtest a VaR level against realised n-day price changes.

    Args:
        tickers: index ticker(s).
        n_liquidation: days in the liquidation period.
        weights: portfolio weights (``None`` for equal weights).
        start_date, end_date: backtest window (``YYYY-MM-DD``); ``None`` means open-ended.
        var: VaR to test, in percent (2.0 means 2%).
        var_is_log: ``True`` if ``var`` is a log return; it is converted to arithmetic.
        forward_shift: ``False`` dates each n-day change at its end; ``True`` dates it at
            its start. With ``n_liquidation=1`` and prices 100 on 1-Jan and 105 on 2-Jan,
            the 5% change is dated 2-Jan when ``False`` and 1-Jan when ``True``.
        data, config, volatility_factor_id: see :class:`~volatility_factor.data.VarData`.

    Example::

        bt = VarData.from_id("agy_cmo").backtest(10, var=3.0, start_date="2020-01-01")
        bt.independence_test()
        bt.backtest_plot().show()
    """

    _BACKTEST_CACHE = (
        "var_pct",
        "nday_price_return",
        "nday_log_return",
        "break_series",
        "shortfall_series",
        "duration",
    )

    start_date = _setting("start_date", _optional_date)
    end_date = _setting("end_date", _optional_date)
    n_liquidation = _setting("n_liquidation")
    var = _setting("var")
    var_is_log = _setting("var_is_log")
    forward_shift = _setting("forward_shift")

    def __init__(
        self,
        tickers: str | Sequence[str],
        n_liquidation: int | None = None,
        weights: Sequence[float] | float | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        var: float | None = None,
        var_is_log: bool = True,
        forward_shift: bool = False,
        *,
        VaR: Any = _UNSET,  # noqa: N803 - deprecated keyword
        VaR_is_log: Any = _UNSET,  # noqa: N803 - deprecated keyword
        **kwargs: Any,
    ):
        if VaR is not _UNSET or VaR_is_log is not _UNSET:
            warnings.warn(
                "VarBacktest(VaR=..., VaR_is_log=...) is deprecated; use var= and var_is_log=.",
                DeprecationWarning,
                stacklevel=2,
            )
            var = var if VaR is _UNSET else VaR
            var_is_log = var_is_log if VaR_is_log is _UNSET else VaR_is_log
        super().__init__(tickers, weights, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.n_liquidation = n_liquidation
        self.var = var
        self.var_is_log = var_is_log
        self.forward_shift = forward_shift

    def _reset_backtest(self) -> None:
        for name in self._BACKTEST_CACHE:
            self.__dict__.pop(name, None)

    # ------------------------------------------------------------------ chaining
    def with_var(self, var: float, var_is_log: bool | None = None) -> VarBacktest:
        """Set the VaR to test (and optionally whether it is a log return); returns ``self``."""
        self.var = var
        if var_is_log is not None:
            self.var_is_log = var_is_log
        return self

    def with_window(self, start_date: str | None, end_date: str | None) -> VarBacktest:
        """Set the backtest window; returns ``self``."""
        self.start_date = start_date
        self.end_date = end_date
        return self

    # --------------------------------------------------------------- calculations
    @cached_property
    def var_pct(self) -> float:
        """VaR as a decimal arithmetic return (log VaR is converted first)."""
        if self.var is None:
            raise ValueError("var is required (pass var=... or call with_var())")
        if self.var_is_log:
            return -(np.exp(-self.var / 100) - 1)
        return self.var / 100

    def _nday_change(self, frame: pd.DataFrame, log: bool) -> pd.Series:
        df = frame.sort_index().copy()
        n = self.n_liquidation
        if log:
            chg = df.shift(-n) - df if self.forward_shift else df - df.shift(n)
        else:
            chg = df.shift(-n) / df - 1 if self.forward_shift else df / df.shift(n) - 1
        if self.single_index is True:
            chg = pd.Series(data=chg.iloc[:, 0].values, index=df.index)
        else:
            chg = pd.Series(data=chg.dot(self.weights), index=df.index)
        return chg[self.start_date : self.end_date].dropna()

    @cached_property
    def nday_price_return(self) -> pd.Series:
        """N-day percentage change (weighted for a portfolio)."""
        return self._nday_change(self.price_matrix, log=False)

    @cached_property
    def nday_log_return(self) -> pd.Series:
        """N-day log change (weighted for a portfolio)."""
        return self._nday_change(self.log_price_matrix, log=True)

    @cached_property
    def break_series(self) -> pd.Series:
        """1 where the n-day change is below ``-var_pct`` (an exception), else 0."""
        return (-self.var_pct > self.nday_price_return) * 1

    @cached_property
    def shortfall_series(self) -> pd.Series:
        """N-day change plus VaR: negative = shortfall (break), positive = no break."""
        return self.nday_price_return + self.var_pct

    @cached_property
    def duration(self) -> list[int]:
        """Days between consecutive breaks."""
        breaks = np.array(self.break_series)
        breaks = np.insert(breaks, 0, 1)
        breaks = np.append(breaks, 1)
        return list(np.diff(np.where(breaks == 1))[0, 1:-1])

    @property
    def exceptions(self) -> int:
        """Number of breaks in the window."""
        return int(self.break_series.sum())

    def independence_test(self, converge_rate: float = 0.01, alpha: float = 0.99) -> dict[str, Any]:
        """:func:`log_likelihood_test` on :attr:`break_series`."""
        return log_likelihood_test(self.break_series, converge_rate=converge_rate, alpha=alpha)

    # ------------------------------------------------------------------- plotting
    def backtest_plot(self, title_text: str = "Standard", **kwargs: Any) -> go.Figure:
        """See :func:`volatility_factor.plotting.backtest_plot`."""
        from volatility_factor.plotting import backtest_plot

        return backtest_plot(self, title_text=title_text, **kwargs)

    def price_change_plot(
        self, title_text: str = "Standard", weighted: bool = True, **kwargs: Any
    ) -> go.Figure:
        """See :func:`volatility_factor.plotting.price_change_plot`."""
        from volatility_factor.plotting import price_change_plot

        return price_change_plot(self, title_text=title_text, weighted=weighted, **kwargs)

    def price_plot(
        self, title_text: str = "Standard", weighted: bool = False, **kwargs: Any
    ) -> go.Figure:
        """See :func:`volatility_factor.plotting.price_plot`."""
        from volatility_factor.plotting import price_plot

        return price_plot(self, title_text=title_text, weighted=weighted,
                          start_date=self.start_date, end_date=self.end_date, **kwargs)


# --------------------------------------------------------------------------- #
# Batch backtesting of model results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestRun:
    """Output of :func:`run_backtests`.

    Attributes:
        detail: one row per date per (result row, VaR quantile).
        independence: one row per (result row, VaR quantile) with the test statistics.
    """

    detail: pd.DataFrame
    independence: pd.DataFrame

    def save(self, output_file: str | os.PathLike | None = None, config: Any = None) -> BacktestRun:
        """Write ``backtesting_detail_<name>.parquet`` and ``independence_tests_<name>.xlsx``.

        ``output_file`` defaults to the config's ``loc_output / output_filename``.
        """
        output_file = Path(output_file) if output_file else get_config(config).output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.detail.to_parquet(
            output_file.parent / f"backtesting_detail_{output_file.name}.parquet",
            compression="snappy",
        )
        self.independence.to_excel(
            output_file.parent / f"independence_tests_{output_file.name}.xlsx", index=False
        )
        return self


def _as_date_str(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def run_backtests(
    results: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    data: Any = None,
    config: Any = None,
    converge_rate: float = 0.01,
    alpha: float = 0.99,
    pad_incomplete: bool = True,
) -> BacktestRun:
    """Backtest every VaR in wide GARCH results over its forecast window.

    Each row of ``results`` (as produced by :func:`volatility_factor.pipeline.run_garch_var`
    or read back from the saved workbook) is unpivoted to one row per ``VaR_50/75/95/99``
    and backtested with ``var_is_log=False`` and ``forward_shift=True``.

    Args:
        results: wide results with at least ``Volatility_Factor_ID, Tickers, Weights,
            ForecastStart, ForecastEnd, Method, NDays`` and the ``VaR_*`` columns.
        data, config: pricing data / config (see :class:`~volatility_factor.data.VarData`).
        converge_rate, alpha: passed to :func:`log_likelihood_test`.
        pad_incomplete: when prices end inside a forecast window, extend the detail with
            calendar dates through the window end (no prices or returns on those dates).
            This replaces the hard-coded "UPDATE" dates in the original notebook.

    Returns:
        :class:`BacktestRun` with ``detail`` and ``independence`` frames.
    """
    data = get_data(data, config)
    wide = pd.DataFrame.from_records(results) if not isinstance(results, pd.DataFrame) else results
    id_vars = [c for c in wide.columns if c not in VAR_COLUMNS and c not in LOG_VAR_COLUMNS]
    long = pd.melt(wide, id_vars=id_vars, value_vars=VAR_COLUMNS, var_name="VaRQ", value_name="VaR")

    portfolios: dict[tuple, VarData] = {}
    bt_frames = []
    ind_dicts = []
    for row in long.itertuples(index=False):
        tickers = str(row.Tickers).split(";")
        weights = [float(w) for w in str(row.Weights).split(";")]
        forecast_start = _as_date_str(row.ForecastStart)
        forecast_end = _as_date_str(row.ForecastEnd)
        varq = str(row.VaRQ)[-2:] + "Q"

        key = (tuple(tickers), tuple(weights), row.Volatility_Factor_ID)
        if key not in portfolios:
            portfolios[key] = VarData(tickers, weights, data=data, config=config,
                                      volatility_factor_id=row.Volatility_Factor_ID)
        bt = portfolios[key].backtest(
            int(row.NDays), var=row.VaR, start_date=forecast_start, end_date=forecast_end,
            var_is_log=False, forward_shift=True,
        )

        df = pd.DataFrame(data=None, index=bt.price_matrix[forecast_start:forecast_end].index)
        if df.shape[0] == 0:
            df = pd.DataFrame(data=None, index=pd.date_range(start=forecast_start, end=forecast_end))
        elif pad_incomplete:
            last_price_date = bt.price_matrix.index.max()
            window_end = pd.Timestamp(forecast_end)
            if df.index.max() == last_price_date and last_price_date < window_end:
                pad = pd.date_range(start=last_price_date + pd.Timedelta(days=1), end=window_end)
                df = pd.concat([df, pd.DataFrame(data=None, index=pad)])

        ind_test = bt.independence_test(converge_rate=converge_rate, alpha=alpha)
        ind_test.update({
            "Volatility_Factor_ID": bt.volatility_factor_id,
            "ForecastStart": _as_date_str(bt.start_date),
            "ForecastEnd": _as_date_str(bt.end_date),
            "Method": row.Method,
            "NDays": bt.n_liquidation,
            "VaRQ": varq,
        })
        ind_dicts.append(ind_test)

        df["Date"] = df.index
        df["Volatility_Factor_ID"] = bt.volatility_factor_id
        df["ForecastStart"] = _as_date_str(bt.start_date)
        df["ForecastEnd"] = _as_date_str(bt.end_date)
        df["Method"] = row.Method
        df["NDays"] = bt.n_liquidation
        df["VaRQ"] = varq
        df["VaR_Pct"] = bt.var_pct
        df["Price"] = bt.wtd_price
        df["Pct_Change_Actual"] = bt.nday_price_return
        df["Breaks"] = bt.break_series
        df["ShortfallPct"] = bt.shortfall_series
        df["RejectNullIndependence"] = np.where(ind_test.get("Independence") == "RejectNull", 1, 0)
        bt_frames.append(df)

    detail = pd.concat(bt_frames).reset_index(drop=True) if bt_frames else pd.DataFrame()
    return BacktestRun(detail=detail, independence=pd.DataFrame.from_dict(ind_dicts))
