import pandas as pd
import numpy as np
from typing import Tuple, List



def create_periods(last_sample_date, periods: int, forecast_mos: int, forecast_lag: int, sample_yrs:int = 9) -> Tuple[List[str],List[str]]:
    """Creates a list of lists of sample dates and corresponding forecast dates 

    Args:
        last_sample_date: YYYY-MM-DD of the last sample date
        periods: Number of distinct sample/forecast periods to return
        forecast_mos: Number of months in forecast period
        forecast_lag: Lag (in months) between the sample end date and forecast date
        sample_yrs: Number of years for sample period. Defaults to 9.

    Returns:
        Tuple of lists: the first containing list of sample dates and the second containing a list of 
        the corresponding forecast dates.  
    """
    sample_end_dates = pd.date_range(end = last_sample_date, periods = periods, freq = f"{forecast_mos}M")

    #Use list created above for iteration to create return objects
    sample_periods = []
    forecast_periods = []
    for end_date in sample_end_dates:
        sample_start = end_date - pd.offsets.MonthBegin(sample_yrs * 12)
        sample_end = sample_start + pd.offsets.MonthEnd(sample_yrs * 12)
        forecast_start = sample_end + pd.offsets.MonthBegin(forecast_lag + 1)
        forecast_end = forecast_start + pd.offsets.MonthEnd(forecast_mos)
        
        #append each date to return lists
        sample_periods.append([sample_start.strftime("%Y-%m-%d"), sample_end.strftime("%Y-%m-%d")])
        forecast_periods.append([forecast_start.strftime("%Y-%m-%d"), forecast_end.strftime("%Y-%m-%d")])

    return sample_periods[::-1], forecast_periods[::-1] #reverse lists



def get_stress_range_multi(returns: pd.DataFrame, start: str, end: str,  weights: np.array, ndays: int, alpha: float = 0.99) -> List[str]:
    """Given a dataframe of portfolio log returns - where each column in the portfolio contains a constituents daily 
    log retutns - calculates a series of rolling n-day standard deviations of portfolio returns.  From the series 
    of n-day standard deviations, returns the corresponding nday period at the specified quantile.

    Args:
        returns: dataframe of one-day log returns
        start: first date (inclusive) of returns to include.
        end: last date (inclusive) of returns to include.
        weights: portfolio weights
        ndays: Number of days for each n-day standard deviation to calcuate
        alpha: Quantile of standard deviation to select. Defaults to 0.99.

    Returns:
        Two-element list [start, end] corresponding to the desired quantile of nday standard deviations.
    """
    period_start = pd.Series(returns[start:end].index).to_list()

    Rho = returns.loc[start:end].corr()
    std = []
    stress_range = []
    for idx in range(len(period_start[:-(ndays-1)])):
        start_date = period_start[idx]
        end_date = pd.Series(returns[start_date:].index).shift(-(ndays-1)).min()
        M_sigma = np.diag(returns.loc[start_date:end_date].std())
        Sigma2 = M_sigma.dot(Rho).dot(M_sigma)
        sigma2_weight = weights.dot(Sigma2).dot(weights)
        std.append(np.sqrt(sigma2_weight))
        stress_range.append([start_date, end_date])
    std_ser = pd.Series(std)
    stdqq = std_ser.quantile(alpha, interpolation = "higher")
    period_idx = std_ser[std_ser == stdqq].index.astype(int).to_list()[0]
    stress_period = stress_range[period_idx]
    
    return [pd.to_datetime(stress_period[0]).strftime("%Y-%m-%d"), pd.to_datetime(stress_period[1]).strftime("%Y-%m-%d")]


def get_stress_range_single(returns: pd.DataFrame, start: str, end: str,  ndays: int, alpha: float = 0.99) -> List[str]:
    """Given a dataframe of a single index log returns, calculates a series of rolling n-day standard deviations of returns.  
    From the series of n-day standard deviations, returns the corresponding nday period at the specified quantile.

    Args:
        returns: dataframe of one-day log returns
        start: first date (inclusive) of returns to include.
        end: last date (inclusive) of returns to include.
        ndays: Number of days for each n-day standard deviation to calcuate
        alpha: Quantile of standard deviation to select. Defaults to 0.99.

    Returns:
        Two-element list [start, end] corresponding to the desired quantile of nday standard deviations.
    """
    period_start = pd.Series(returns[start:end].index).to_list()

    std = []
    stress_range = []
    for idx in range(len(period_start[:-(ndays-1)])):
        start_date = period_start[idx]
        end_date = pd.Series(returns[start_date:].index).shift(-(ndays-1)).min()
        std.append(returns[start_date:end_date].values.squeeze().std())
        stress_range.append([start_date, end_date])
    std_ser = pd.Series(std)
    stdqq = std_ser.quantile(alpha, interpolation = "higher")
    period_idx = std_ser[std_ser == stdqq].index.astype(int).to_list()[0]
    stress_period = stress_range[period_idx]
    
    return [pd.to_datetime(stress_period[0]).strftime("%Y-%m-%d"), pd.to_datetime(stress_period[1]).strftime("%Y-%m-%d")]


