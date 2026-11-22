"""Sample / forecast windows and stress-period selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_periods(
    last_sample_date: str,
    periods: int,
    forecast_mos: int,
    forecast_lag: int,
    sample_yrs: int = 9,
) -> tuple[list[list[str]], list[list[str]]]:
    """Sample windows and their forecast windows, most recent first.

    Args:
        last_sample_date: ``YYYY-MM-DD`` of the last sample date.
        periods: number of sample/forecast pairs to return.
        forecast_mos: months in each forecast window.
        forecast_lag: months between the sample end and the forecast start.
        sample_yrs: years in each sample window.

    Returns:
        ``(sample_periods, forecast_periods)``, each a list of ``[start, end]`` strings.
    """
    sample_end_dates = pd.date_range(
        end=last_sample_date, periods=periods, freq=pd.offsets.MonthEnd(forecast_mos)
    )

    sample_periods = []
    forecast_periods = []
    for end_date in sample_end_dates:
        sample_start = end_date - pd.offsets.MonthBegin(sample_yrs * 12)
        sample_end = sample_start + pd.offsets.MonthEnd(sample_yrs * 12)
        forecast_start = sample_end + pd.offsets.MonthBegin(forecast_lag + 1)
        forecast_end = forecast_start + pd.offsets.MonthEnd(forecast_mos)

        sample_periods.append([sample_start.strftime("%Y-%m-%d"), sample_end.strftime("%Y-%m-%d")])
        forecast_periods.append(
            [forecast_start.strftime("%Y-%m-%d"), forecast_end.strftime("%Y-%m-%d")]
        )

    return sample_periods[::-1], forecast_periods[::-1]


def get_stress_range_multi(
    returns: pd.DataFrame,
    start: str,
    end: str,
    weights: np.ndarray,
    ndays: int,
    alpha: float = 0.99,
) -> list[str]:
    """The ``ndays`` window whose portfolio volatility is at the ``alpha`` quantile.

    For each rolling ``ndays`` window of constituent log returns, the portfolio standard
    deviation is computed from the window's constituent volatilities and the full-period
    correlation matrix.

    Returns:
        ``[start, end]`` of the selected window.
    """
    period_start = pd.Series(returns[start:end].index).to_list()

    rho = returns.loc[start:end].corr()
    std = []
    stress_range = []
    for idx in range(len(period_start[: -(ndays - 1)])):
        start_date = period_start[idx]
        end_date = pd.Series(returns[start_date:].index).shift(-(ndays - 1)).min()
        m_sigma = np.diag(returns.loc[start_date:end_date].std())
        sigma2 = m_sigma.dot(rho).dot(m_sigma)
        sigma2_weight = weights.dot(sigma2).dot(weights)
        std.append(np.sqrt(sigma2_weight))
        stress_range.append([start_date, end_date])
    return _select_stress_window(std, stress_range, alpha)


def get_stress_range_single(
    returns: pd.DataFrame,
    start: str,
    end: str,
    ndays: int,
    alpha: float = 0.99,
) -> list[str]:
    """Single-index version of :func:`get_stress_range_multi`."""
    period_start = pd.Series(returns[start:end].index).to_list()

    std = []
    stress_range = []
    for idx in range(len(period_start[: -(ndays - 1)])):
        start_date = period_start[idx]
        end_date = pd.Series(returns[start_date:].index).shift(-(ndays - 1)).min()
        std.append(returns[start_date:end_date].values.squeeze().std())
        stress_range.append([start_date, end_date])
    return _select_stress_window(std, stress_range, alpha)


def get_stress_range(
    returns: pd.DataFrame,
    start: str,
    end: str,
    ndays: int,
    weights: np.ndarray | None = None,
    alpha: float = 0.99,
) -> list[str]:
    """Dispatch to the single- or multi-index version based on ``returns`` columns."""
    if returns.shape[1] == 1:
        return get_stress_range_single(returns, start, end, ndays=ndays, alpha=alpha)
    return get_stress_range_multi(returns, start, end, weights=weights, ndays=ndays, alpha=alpha)


def _select_stress_window(std: list, stress_range: list, alpha: float) -> list[str]:
    """Pick the window whose volatility is at the ``alpha`` quantile.

    Args:
        std: volatility of each candidate window.
        stress_range: ``[start, end]`` of each candidate window.
        alpha: quantile to select (``higher`` interpolation).

    Returns:
        ``[start, end]`` as ``YYYY-MM-DD`` strings.
    """
    std_ser = pd.Series(std)
    stdqq = std_ser.quantile(alpha, interpolation="higher")
    period_idx = std_ser[std_ser == stdqq].index.astype(int).to_list()[0]
    stress_period = stress_range[period_idx]
    return [
        pd.to_datetime(stress_period[0]).strftime("%Y-%m-%d"),
        pd.to_datetime(stress_period[1]).strftime("%Y-%m-%d"),
    ]
