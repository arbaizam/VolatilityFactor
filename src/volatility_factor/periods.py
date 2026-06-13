"""Date period and stress-window utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def create_periods(
    last_sample_date: str,
    periods: int,
    forecast_mos: int,
    forecast_lag: int,
    sample_yrs: int = 9,
) -> Tuple[List[list[str]], List[list[str]]]:
    """Create sample and forecast windows using the legacy notebook convention."""

    sample_end_dates = pd.date_range(
        end=last_sample_date,
        periods=periods,
        freq=pd.offsets.MonthEnd(forecast_mos),
    )
    sample_periods = []
    forecast_periods = []
    for end_date in sample_end_dates:
        sample_start = end_date - pd.offsets.MonthBegin(sample_yrs * 12)
        sample_end = sample_start + pd.offsets.MonthEnd(sample_yrs * 12)
        forecast_start = sample_end + pd.offsets.MonthBegin(forecast_lag + 1)
        forecast_end = forecast_start + pd.offsets.MonthEnd(forecast_mos)

        sample_periods.append([sample_start.strftime("%Y-%m-%d"), sample_end.strftime("%Y-%m-%d")])
        forecast_periods.append([forecast_start.strftime("%Y-%m-%d"), forecast_end.strftime("%Y-%m-%d")])

    return sample_periods[::-1], forecast_periods[::-1]


def get_stress_range_multi(
    returns: pd.DataFrame,
    start: str,
    end: str,
    weights: np.ndarray,
    ndays: int,
    alpha: float = 0.99,
) -> list[str]:
    """Return the n-day stress period selected by weighted covariance volatility."""

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

    std_ser = pd.Series(std)
    stdqq = std_ser.quantile(alpha, interpolation="higher")
    period_idx = std_ser[std_ser == stdqq].index.astype(int).to_list()[0]
    stress_period = stress_range[period_idx]
    return [
        pd.to_datetime(stress_period[0]).strftime("%Y-%m-%d"),
        pd.to_datetime(stress_period[1]).strftime("%Y-%m-%d"),
    ]


def get_stress_range_single(
    returns: pd.DataFrame,
    start: str,
    end: str,
    ndays: int,
    alpha: float = 0.99,
) -> list[str]:
    """Return the n-day stress period selected by univariate volatility."""

    period_start = pd.Series(returns[start:end].index).to_list()
    std = []
    stress_range = []
    for idx in range(len(period_start[: -(ndays - 1)])):
        start_date = period_start[idx]
        end_date = pd.Series(returns[start_date:].index).shift(-(ndays - 1)).min()
        std.append(returns[start_date:end_date].values.squeeze().std())
        stress_range.append([start_date, end_date])

    std_ser = pd.Series(std)
    stdqq = std_ser.quantile(alpha, interpolation="higher")
    period_idx = std_ser[std_ser == stdqq].index.astype(int).to_list()[0]
    stress_period = stress_range[period_idx]
    return [
        pd.to_datetime(stress_period[0]).strftime("%Y-%m-%d"),
        pd.to_datetime(stress_period[1]).strftime("%Y-%m-%d"),
    ]
