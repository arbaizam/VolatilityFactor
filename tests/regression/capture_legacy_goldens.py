"""Capture reference ("golden") outputs from the legacy, pre-refactor modules.

This script is kept for provenance only. It imports the legacy root-level modules
(``VaRData``, ``VaRMethods``, ``VaRBacktesting``, ``util_functions``), which exist
only at commit ``d0d285f``. To re-run it::

    git worktree add ../vf-legacy d0d285f
    cd ../vf-legacy
    python <path-to-this-file> --out <path-to>/tests/fixtures/golden

The notebook loops below are copied from ``volatility_factor_main.ipynb``,
``backtesting_function_with_independence.ipynb`` and ``monitoring_main.ipynb`` with
only these edits: file writes are removed, run sizes come from the test config, and
the backtest notebook's hard-coded "UPDATE" padding dates are set to the fixture
data's last date (what a user would type when running against this data).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import VaRData  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_CSV = FIXTURES / "pricing_sample.csv"
TEST_CONFIG = FIXTURES / "config_test.yml"

FIXTURE_DATA = pd.read_csv(
    FIXTURE_CSV, parse_dates=["Date"], dtype={"Price": np.float64}, thousands=","
).sort_values(by=["Ticker", "Date"])
FULL_DATA = VaRData.DATA
VaRData.DATA = FIXTURE_DATA  # every legacy class reads this module global

with open(TEST_CONFIG) as f:
    TEST_CONFIGS = yaml.safe_load(f)
VaRData.CONFIGS = TEST_CONFIGS  # used by VarData.volatility_factor_id

from VaRBacktesting import VarBacktest  # noqa: E402
from VaRMethods import GarchSim, NonParametricVar  # noqa: E402
from util_functions import (  # noqa: E402
    create_periods,
    get_stress_range_multi,
    get_stress_range_single,
)

PORTFOLIOS = {
    "single": (["M0A0"], None),
    "pair": (["G6P0", "G9P0"], [0.5, 0.5]),
    "three": (["G1P0", "G2P0", "CMOS"], [0.5, 0.3, 0.2]),
}
SAMPLE = ("2016-01-01", "2024-12-31")
START_VALS = ["2024-01-01", 0.5]
BOOT = ("2024-07-01", "2024-12-31")


def legacy_loglikelihood_test(break_ser, converge_rate=0.01, alpha=0.99):
    """Verbatim copy of LogLikelihood_test from the backtesting notebook."""
    T = len(break_ser)
    N = break_ser.sum()

    # Kupiec(1995)
    if N == T:
        LR_uc = float("inf")
    else:
        LR_uc = -2 * np.log((1-converge_rate)**(T-N)*(converge_rate**N)) + 2 * np.log((1-N/T)**(T-N)*(N/T)**N)

    # Christoffersen (1998)
    df_break = break_ser
    T00,T01,T10,T11 = 0,0,0,0
    for i in range(len(df_break)-1):
        if i == 0:
            if df_break[0] == 0:
                T00 = T00 + 1
            else:
                T01 = T01 + 1
        if df_break[i] == 0:
            if df_break[i+1] == 0:
                T00 = T00 + 1
            else:
                T01 = T01 + 1
        elif df_break[i] == 1:
            if df_break[i+1] == 0:
                T10 = T10 + 1
            else:
                T11 = T11 + 1

    if N == T:
        LR_ind = float("inf")
    elif N == 0:
        LR_ind = 0
    else:
        if T00 == 0:
            pi0 = 1
        else:
            pi0 = T01 / (T - N)
        pi  = N / T
        pi1 = T11 / N
        LR_ind = -2 * np.log((1-pi)**(T00+T10) * pi**(T01+T11)) + 2 * np.log((1-pi0)**T00 * pi0**T01 * (1-pi1)**T10 * pi1**T11)

    LR_cc = LR_uc + LR_ind

    chi2_1 = scipy.stats.chi2.ppf(alpha,1)
    chi2_2 = scipy.stats.chi2.ppf(alpha,2)

    result = {
        'Exceptions': break_ser.sum(),
        'LR_uc':round(LR_uc,3),
        'LR_ind':round(LR_ind,3),
        'LR_cc':round(LR_cc,3),
        f'chi2(1) {int(alpha*100)}%':round(chi2_1,3),
        f'chi2(2) {int(alpha*100)}%':round(chi2_2,3),
        "Independence": "RejectNull" if LR_ind > chi2_1 else None,
        "alpha": round(alpha * 100, 1),
        "converge_rate": round(converge_rate * 100,1)}

    return result


# --------------------------------------------------------------------------- #
# Unit-level goldens
# --------------------------------------------------------------------------- #
def capture_vardata():
    out = {}
    for key, (tickers, weights) in PORTFOLIOS.items():
        v = VaRData.VarData(tickers, weights)
        out[key] = {
            "weights": v.weights,
            "price_matrix": v.price_matrix,
            "price_return_matrix": v.price_return_matrix,
            "wtd_price_return": v.wtd_price_return,
            "wtd_price": v.wtd_price,
            "log_price_matrix": v.log_price_matrix,
            "log_return_matrix": v.log_return_matrix,
            "wtd_log_return": v.wtd_log_return,
            "index_description": v.index_description,
            "ticker_description": v.ticker_description,
        }
    out["pair"]["volatility_factor_id"] = VaRData.VarData(["G6P0", "G9P0"], [0.5, 0.5]).volatility_factor_id
    out["single"]["volatility_factor_id"] = VaRData.VarData("M0A0").volatility_factor_id
    return out


def capture_garch_fit():
    out = {}
    for key, (tickers, weights) in PORTFOLIOS.items():
        g = GarchSim(tickers, weights, *SAMPLE)
        out[key] = {
            "training_data": g.training_data,
            "training_data_scaled": g.training_data_scaled,
            "residuals": g.residuals,
            "std_residuals": g.std_residuals,
            "conditional_vol": g.conditional_vol,
            "std_returns": g.std_returns,
            "coefficients": g.coefficients,
            "constant_mean": g.constant_mean,
            "yscale": g.yscale,
            "convergence_codes": g.convergence_codes,
        }
    g = GarchSim("M0A0", None, *SAMPLE, mean_model="Constant")
    out["single_constant_mean"] = {"coefficients": g.coefficients, "constant_mean": g.constant_mean,
                                   "residuals": g.residuals}
    return out


def capture_simulations():
    out = {}
    for key, (tickers, weights) in PORTFOLIOS.items():
        g = GarchSim(tickers, weights, *SAMPLE)
        for shock in ["normal", "custom_normal", "bootstrap", "custom_bootstrap"]:
            for sv in [None, START_VALS]:
                kwargs = {}
                if shock in ("bootstrap", "custom_bootstrap"):
                    kwargs = {"bootstrap_start": BOOT[0], "bootstrap_end": BOOT[1]}
                d = g.CustomNDayPriceSimulation(horizon1=63, horizon2=252 + 10 - 1, ndays=10, shock_type=shock,
                                                nsims=1000, start_vals=sv, **kwargs)
                paths = np.ascontiguousarray(g.sim_paths)
                out[(key, shock, sv is not None)] = {"dict": d, "sim_paths_shape": paths.shape,
                                                     "sim_paths_sha256": hashlib.sha256(paths.tobytes()).hexdigest()}
    g = GarchSim("M0A0", None, *SAMPLE)
    out[("single", "custom_normal_10k_252")] = {
        "dict": g.CustomNDayPriceSimulation(horizon1=63, horizon2=252 + 252 - 1, ndays=252,
                                            shock_type="custom_normal", nsims=10000,
                                            start_vals=["2024-01-01", 0.95], VaRp=0.95)}
    g = GarchSim("M0A0", None, *SAMPLE, mean_model="Constant")
    out[("single_constant_mean", "normal")] = {
        "dict": g.CustomNDayPriceSimulation(horizon1=63, horizon2=261, ndays=10, shock_type="normal",
                                            nsims=1000)}
    return out


def capture_bootstrap():
    out = {}
    for key, (tickers, weights) in PORTFOLIOS.items():
        n = NonParametricVar(tickers, weights, *SAMPLE)
        out[key] = n.CircularBlockBootstrap(n_liquidation=10, nboot=500, VaRp=0.99)
    return out


def capture_periods():
    sp, fp = create_periods("2024-12-31", periods=3, forecast_mos=12, forecast_lag=3, sample_yrs=9)
    sp2, fp2 = create_periods("2025-06-30", periods=5, forecast_mos=6, forecast_lag=1, sample_yrs=5)
    single = VaRData.VarData("M0A0").log_return_matrix[SAMPLE[0]:SAMPLE[1]].copy()
    multi_v = VaRData.VarData(["G1P0", "G2P0", "CMOS"], [0.5, 0.3, 0.2])
    multi = multi_v.log_return_matrix[SAMPLE[0]:SAMPLE[1]].copy()
    return {
        "periods_a": (sp, fp),
        "periods_b": (sp2, fp2),
        "stress_single": get_stress_range_single(single, *SAMPLE, ndays=126, alpha=0.99),
        "stress_multi": get_stress_range_multi(multi, *SAMPLE, weights=multi_v.weights, ndays=126, alpha=0.99),
    }


def capture_backtest():
    out = {}
    cases = {
        "single_log": dict(tickers="M0A0", n_liquidation=10, VaR=3.0, start_date="2020-01-01",
                           end_date="2022-09-30"),
        "single_pct_fwd": dict(tickers="M0A0", n_liquidation=10, VaR=2.5, start_date="2022-01-01",
                               end_date="2023-12-31", VaR_is_log=False, forward_shift=True),
        "pair_pct": dict(tickers=["G6P0", "G9P0"], weights=[0.5, 0.5], n_liquidation=5, VaR=1.0,
                         start_date="2022-01-01", end_date="2024-12-31", VaR_is_log=False),
        "three_log_fwd": dict(tickers=["G1P0", "G2P0", "CMOS"], weights=[0.5, 0.3, 0.2], n_liquidation=1,
                              VaR=0.5, start_date="2020-01-01", end_date="2025-06-30", forward_shift=True),
        "open_dates": dict(tickers="CMOS", n_liquidation=10, VaR=2.0),
    }
    for key, kw in cases.items():
        bt = VarBacktest(**kw)
        out[key] = {
            "var_pct": bt.var_pct,
            "nday_price_return": bt.nday_price_return,
            "nday_log_return": bt.nday_log_return,
            "break_series": bt.break_series,
            "shortfall_series": bt.shortfall_series,
            "duration": bt.duration,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[key]["llt"] = legacy_loglikelihood_test(bt.break_series)
            out[key]["llt_95"] = legacy_loglikelihood_test(bt.break_series, converge_rate=0.05, alpha=0.95)
    # edge cases for the test statistic
    idx = pd.date_range("2024-01-01", periods=30)
    edge = {
        "none": pd.Series(0, index=idx),
        "all": pd.Series(1, index=idx),
        "first": pd.Series([1] + [0] * 29, index=idx),
        "cluster": pd.Series([0] * 10 + [1, 1, 1] + [0] * 17, index=idx),
        "alt": pd.Series([0, 1] * 15, index=idx),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["llt_edge"] = {k: legacy_loglikelihood_test(s) for k, s in edge.items()}
    return out


# --------------------------------------------------------------------------- #
# Notebook-level goldens
# --------------------------------------------------------------------------- #
def legacy_main_garch(CONFIGS):
    """volatility_factor_main.ipynb cell 5 (file writes removed)."""
    horizon1 = CONFIGS["garch_settings"]["horizon1"]
    horizon2 = CONFIGS["garch_settings"]["horizon2"]
    nsims = CONFIGS["garch_settings"]["nsims"]
    last_sample_date = CONFIGS["garch_settings"]["last_sample_date"]
    number_of_periods = CONFIGS["garch_settings"]["number_of_periods"]
    sample_yrs = CONFIGS["garch_settings"]["sample_years"]
    forecast_mos = CONFIGS["garch_settings"]["forecast_months"]
    forecast_lag = CONFIGS["garch_settings"]["forecast_lag"]
    start_point_quantiles = CONFIGS["garch_settings"]["start_point_quantiles"]
    var_type = CONFIGS["garch_settings"]["var_type"]

    volatility_factor_id = []
    tickers = []
    weights = []
    ndays = []
    for idx, vals in CONFIGS["var_settings"].items():
        volatility_factor_id.append(idx)
        tickers.append(vals["tickers"])
        weights.append(vals["weights"])
        ndays.append(vals["ndays"])
    weights = [None if x == "None" else x for x in weights]

    sample_range, forecast_range =  create_periods(last_sample_date = last_sample_date, periods=number_of_periods, forecast_mos=forecast_mos,
                                        forecast_lag=forecast_lag, sample_yrs = sample_yrs)

    garch_var_dicts = []
    sample_qa_dicts = []
    for i in range(len(volatility_factor_id)):
        for nday in ndays[i]:
            for idx in range(len(forecast_range)):
                for start_point_quantile in start_point_quantiles:
                    horizon2adj = horizon2 + nday - 1
                    sample_start = sample_range[idx][0]
                    sample_end = sample_range[idx][1]
                    forecast_start = forecast_range[idx][0]
                    forecast_end = forecast_range[idx][1]
                    garch = GarchSim(tickers = tickers[i], weights = weights[i], sample_start = sample_start, sample_end = sample_end)

                    first_price_date = garch.price_matrix.first_valid_index() - pd.offsets.Day(4)
                    last_price_date = garch.price_matrix.last_valid_index() + pd.offsets.Day(4)
                    if first_price_date > pd.to_datetime(sample_start) or last_price_date < pd.to_datetime(sample_end):
                        sample_qa =  {
                            "Volatility_Factor_ID": volatility_factor_id[i],
                            "Tickers": garch.ticker_description,
                            "SampleStart": sample_start,
                            "SampleEnd" : sample_end,
                            "FirstPriceDate": first_price_date.strftime("%Y-%m-%d"),
                            "LastPriceDate": last_price_date.strftime("%Y-%m-%d")}
                        sample_qa_dicts.append(sample_qa)
                        continue

                    if "long_term" in var_type:
                        start_point_date = (pd.to_datetime(sample_end) - pd.offsets.MonthBegin(12)).strftime("%Y-%m-%d")
                        var = garch.CustomNDayPriceSimulation(horizon1 = horizon1, horizon2 = horizon2adj, ndays = nday,
                                                            nsims = nsims, shock_type = "custom_normal", start_vals= [start_point_date, start_point_quantile])
                        var.update({
                            "Volatility_Factor_ID": volatility_factor_id[i],
                            "ForecastStart": forecast_start,
                            "ForecastEnd" : forecast_end,
                            "Method": f"long_term_var_{round(start_point_quantile * 100)}"})
                        garch_var_dicts.append(var)

                    if "normal" in var_type:
                        start_point_date = (pd.to_datetime(sample_end) - pd.offsets.MonthBegin(12)).strftime("%Y-%m-%d")
                        nvar = garch.CustomNDayPriceSimulation(horizon1 = horizon1, horizon2 = horizon2adj, ndays = nday,
                                                            nsims = nsims, shock_type = "normal", start_vals= [start_point_date, start_point_quantile])
                        nvar.update({
                            "Volatility_Factor_ID": volatility_factor_id[i],
                            "ForecastStart": forecast_start,
                            "ForecastEnd" : forecast_end,
                            "Method": f"normal_var_{round(start_point_quantile * 100)}"})
                        garch_var_dicts.append(nvar)

                    if "short_term" in var_type:
                        start_point_date = (pd.to_datetime(sample_end) - pd.offsets.MonthBegin(12)).strftime("%Y-%m-%d")
                        bootstrap_start = (pd.to_datetime(sample_end) - pd.offsets.MonthBegin(6)).strftime("%Y-%m-%d")
                        bootstrap_end = sample_end
                        stvar = garch.CustomNDayPriceSimulation(horizon1 = horizon1, horizon2 = horizon2adj, ndays = nday, nsims = nsims,
                                        shock_type = "custom_bootstrap", bootstrap_start = bootstrap_start, bootstrap_end = bootstrap_end,
                                    start_vals=[start_point_date, start_point_quantile])
                        stvar.update({
                            "Volatility_Factor_ID": volatility_factor_id[i],
                            "ForecastStart": forecast_start,
                            "ForecastEnd" : forecast_end,
                            "Method": f"short_term_var_{round(start_point_quantile * 100)}"})
                        garch_var_dicts.append(stvar)

                    if "stress_var" in var_type:
                        wts = garch.weights
                        start_point_date = (pd.to_datetime(sample_end) - pd.offsets.MonthBegin(12)).strftime("%Y-%m-%d")
                        returns = garch.log_return_matrix[sample_start: sample_end].copy()
                        single_index = garch.single_index
                        if single_index is True:
                            stress_periods = get_stress_range_single(start=sample_start, end=sample_end, ndays = 126, returns = returns, alpha=0.99)
                        else:
                           stress_periods = get_stress_range_multi(start=sample_start, end=sample_end, ndays = 126, returns = returns, weights = wts, alpha=0.99)
                        bootstrap_start = stress_periods[0]
                        bootstrap_end = stress_periods[1]
                        stressvar = garch.CustomNDayPriceSimulation(horizon1 = horizon1, horizon2 = horizon2adj, ndays = nday, nsims = nsims,
                                        shock_type = "custom_bootstrap", bootstrap_start = bootstrap_start, bootstrap_end = bootstrap_end,
                                    start_vals=[start_point_date, start_point_quantile])
                        stressvar.update({
                            "Volatility_Factor_ID": volatility_factor_id[i],
                            "ForecastStart": forecast_start,
                            "ForecastEnd" : forecast_end,
                            "Method": f"stress_var_{round(start_point_quantile * 100)}"})
                        garch_var_dicts.append(stressvar)

    df_wide = pd.DataFrame.from_dict(garch_var_dicts)
    df = df_wide.copy()
    date_cols = ["SampleStart", "SampleEnd","ForecastStart", "ForecastEnd", "BootstrapStart", "BootstrapEnd", "StartingValueDate"]
    id_cols = [x for x in df.columns if x not in ["LogVaR_50", "LogVaR_75", "LogVaR_95", "Log_VaR_99", "VaR_50", "VaR_75", "VaR_95", "VaR_99"]]
    value_cols = ["VaR_50", "VaR_75","VaR_95", "VaR_99"]
    df[date_cols] = df[date_cols].apply(pd.to_datetime)
    df = df.melt(id_vars = id_cols, value_vars = value_cols, var_name= "VaRQ", value_name= "VaR")
    df["VaRQ"] = df["VaRQ"].str[-2:] + "Q"
    return {"wide": df_wide, "long": df, "qa": pd.DataFrame.from_dict(sample_qa_dicts)}


def legacy_main_bootstrap(CONFIGS, nboot):
    """volatility_factor_main.ipynb cell 7 (file write removed; nboot parameterised)."""
    last_sample_date = CONFIGS["garch_settings"]["last_sample_date"]
    number_of_periods = CONFIGS["garch_settings"]["number_of_periods"]
    sample_yrs = CONFIGS["garch_settings"]["sample_years"]
    forecast_mos = CONFIGS["garch_settings"]["forecast_months"]
    forecast_lag = CONFIGS["garch_settings"]["forecast_lag"]
    volatility_factor_id, tickers, weights, ndays = [], [], [], []
    for idx, vals in CONFIGS["var_settings"].items():
        volatility_factor_id.append(idx)
        tickers.append(vals["tickers"])
        weights.append(vals["weights"])
        ndays.append(vals["ndays"])
    weights = [None if x == "None" else x for x in weights]
    sample_range, forecast_range =  create_periods(last_sample_date = last_sample_date, periods=number_of_periods, forecast_mos=forecast_mos,
                                        forecast_lag=forecast_lag, sample_yrs = sample_yrs)
    sample_range = sample_range[0]
    boot_var_dicts = []
    for i in range(len(volatility_factor_id)):
        for nday in ndays[i]:
            a = NonParametricVar(tickers[i], weights=weights[i], sample_start = sample_range[0], sample_end = sample_range[1])
            vol = a.CircularBlockBootstrap(n_liquidation = nday, nboot = nboot, VaRp = 0.99)
            vol.update({"Volatility_Factor_ID": volatility_factor_id[i]})
            boot_var_dicts.append(vol)
    return pd.DataFrame.from_dict(boot_var_dicts)


def legacy_backtest_notebook(wide: pd.DataFrame, tmp_xlsx: Path):
    """backtesting_function_with_independence.ipynb cell 5 (UPDATE dates set for fixture data)."""
    wide.to_excel(tmp_xlsx, index=False)
    data = pd.read_excel(tmp_xlsx, dtype={"Weights": str})
    data_end = FIXTURE_DATA["Date"].max()

    df = pd.melt(data,
                  id_vars = ["Volatility_Factor_ID", "Tickers", "Weights", "SampleStart", "SampleEnd", "BootstrapStart", "BootstrapEnd", "ForecastStart",
                            "ForecastEnd", "StartingValueDate", "ShockType", "VaRp", "Method", "Horizon1", "Horizon2", "NDays", "StartingValueQ", "ConvergencyCode",
                            "Simulations", "MeanModel", "VolatilityModel"],
                  value_vars = ["VaR_50", "VaR_75", "VaR_95", "VaR_99"],
                  var_name = "VaRQ",
                  value_name = "VaR")

    tickers = df["Tickers"].str.split(";").to_list()
    weightsTEMP = df["Weights"].str.split(";").to_list()
    weights = [[float(weight) for weight in x] for x in weightsTEMP]
    forecast_start = pd.to_datetime(df["ForecastStart"]).dt.strftime("%Y-%m-%d").to_list()
    forecast_end = pd.to_datetime(df["ForecastEnd"]).dt.strftime("%Y-%m-%d").to_list()
    method = df["Method"].to_list()
    n_liquidation = df["NDays"].to_list()
    varq = (df["VaRQ"].str[-2:] + "Q").to_list()
    var = df["VaR"].to_list()

    bt_frames = []
    ind_dicts = []
    for i in range(len(tickers)):
        bt = VarBacktest(tickers = tickers[i], n_liquidation = n_liquidation[i], weights = weights[i],
                    start_date = forecast_start[i], end_date = forecast_end[i], VaR = var[i], VaR_is_log = False, forward_shift=True)
        df = pd.DataFrame(data = None, index = bt.price_matrix[forecast_start[i]:forecast_end[i]].index)
        if df.shape[0] == 0:
            df = pd.DataFrame(data = None, index = pd.date_range(start = forecast_start[i], end = forecast_end[i]))
        elif df.index.max() == data_end:  # UPDATE (was "2024-12-31")
                df = pd.concat([df, pd.DataFrame(data = None, index = pd.date_range(start = data_end + pd.Timedelta(days=1), end = forecast_end[i]))])  # UPDATE

        warnings.filterwarnings('ignore')
        ind_test = legacy_loglikelihood_test(bt.break_series)
        ind_test.update({
            "Volatility_Factor_ID": bt.volatility_factor_id,
            "ForecastStart": pd.to_datetime(bt.start_date).strftime("%Y-%m-%d"),
            "ForecastEnd" : pd.to_datetime(bt.end_date).strftime("%Y-%m-%d")})
        ind_dicts.append(ind_test)
        warnings.filterwarnings('default')

        df["Date"] = df.index
        df["Volatility_Factor_ID"] = bt.volatility_factor_id
        df["ForecastStart"] = pd.to_datetime(bt.start_date).strftime("%Y-%m-%d")
        df["ForecastEnd"] = pd.to_datetime(bt.end_date).strftime("%Y-%m-%d")
        df["Method"] = method[i]
        df["NDays"] = bt.n_liquidation
        df["VaRQ"] = varq[i]
        df["VaR_Pct"] = bt.var_pct
        df["Price"] = bt.wtd_price
        df["Pct_Change_Actual"] = bt.nday_price_return
        df["Breaks"] = bt.break_series
        df["ShortfallPct"] = bt.shortfall_series
        df["RejectNullIndependence"] = np.where(ind_test.get("Independence") == "RejectNull", 1, 0)
        bt_frames.append(df)

    detail = pd.concat(bt_frames).reset_index(drop=True)
    return {"detail": detail, "independence": pd.DataFrame.from_dict(ind_dicts)}


def legacy_monitoring(CONFIGS):
    """monitoring_main.ipynb cells 2-17 (display/file writes removed)."""
    def get_lists():
        volatility_factor_id, tickers, weights, ndays = [], [], [], []
        for idx, vals in CONFIGS["var_settings"].items():
            volatility_factor_id.append(idx)
            tickers.append(vals["tickers"])
            weights.append(None if vals["weights"] == "None" else vals["weights"])
            ndays.append(vals["ndays"])
        return {"ids": volatility_factor_id, "tickers": tickers, "weights": weights, "ndays": ndays}

    def F_monitor(index, period_full, period_start, period_new, alpha=0.99):
        df = VaRData.VarData(index).wtd_log_return
        full = df.loc[period_full[0]:period_full[1]]
        start = df.loc[period_start[0]:period_start[1]]
        new  = df.loc[period_new[0]:period_new[1]]
        return {
            "Q-vs-Full":     new.values.var(ddof = 1) / full.values.var(ddof = 1),
            "Q-vs-Full_CV":  scipy.stats.f.ppf(q = alpha, dfn = len(new) - 1, dfd = len(full) - 1),
            "Q-vs-1Y":    new.values.var(ddof = 1) / start.values.var(ddof = 1),
            "Q-vs-1Y_CV": scipy.stats.f.ppf(q = alpha, dfn = len(new) - 1, dfd = len(start) - 1),
            "Q-vs-Max":      new.values.var(ddof = 1) / full.rolling(len(new)).var().dropna().quantile(1).values[0],
            "Q-vs-Max_CV":   scipy.stats.f.ppf(q = alpha, dfn = len(new)-1, dfd = len(new) - 1)
        }

    PERIOD_FULL   = ['2016-01-01', '2024-12-31']
    PERIOD_START  = ['2024-01-01', '2024-12-31']
    PERIOD_NEW_1  = ['2025-01-01', '2025-03-31']
    PERIOD_NEW_2  = ['2025-04-01', '2025-06-30']
    PERIOD_NEW_3  = ['2025-07-01', '2025-09-30']
    PERIOD_NEW_4  = ['2025-10-01', '2025-12-31']
    PERIOD_NEW_5  = ['2026-01-01', '2026-03-31']
    out = {}

    # ---- F-test (cell 7) ----
    quarters = [1,2,3]
    _lists = get_lists()
    tickers, volatility_factor_id = _lists.get("tickers"), _lists.get("ids")
    frames = []
    for qtr in quarters:
        results = []
        for idx in range(len(tickers)):
            temp = {"ID": volatility_factor_id[idx]}
            period_new = {1: PERIOD_NEW_1, 2: PERIOD_NEW_2, 3: PERIOD_NEW_3, 4: PERIOD_NEW_4, 5: PERIOD_NEW_5}[qtr]
            results.append({**temp, **F_monitor(index = tickers[idx], period_full = PERIOD_FULL, period_start = PERIOD_START, period_new = period_new)})
        df = pd.DataFrame.from_dict(results)
        df.columns = [str(qtr) + col if col != "ID" else col for col in df.columns]
        frames.append(df.set_index("ID"))
    out["fstats"] = pd.concat(frames, axis=1)

    # ---- 1-day exceptions (cell 11), nsims reduced to the test config's value ----
    nsims = CONFIGS["garch_settings"]["nsims"]
    _lists = get_lists()
    tickers, volatility_factor_id, weights = _lists.get("tickers"), _lists.get("ids"), _lists.get("weights")
    r1, r2, r3, r4 = [], [], [], []
    nday_vars = []
    for i in range(len(tickers)):
        v = GarchSim(tickers = tickers[i], weights = weights[i], sample_start = PERIOD_FULL[0], sample_end = PERIOD_FULL[1])
        var = v.CustomNDayPriceSimulation(horizon1 = 63, horizon2 = 252, ndays = 1, nsims = nsims,
                                        shock_type = "custom_normal", start_vals=[PERIOD_START[0], 0.50])
        nday_var = var["VaR_95"]
        # NOTE: legacy passes no weights here, so the backtest uses equal weights even when the
        # VaR was computed with configured weights. Kept verbatim; only "three_way" is affected.
        bts = [VarBacktest(tickers = tickers[i], n_liquidation = 1, VaR = nday_var, start_date=p[0], end_date=p[1], VaR_is_log = False, forward_shift = False)
               for p in (PERIOD_NEW_1, PERIOD_NEW_2, PERIOD_NEW_3, PERIOD_NEW_4)]
        r1.append(bts[0].break_series.sum())
        r2.append(bts[1].break_series.sum())
        r3.append(bts[2].break_series.sum())
        r4.append(bts[3].break_series.sum())
        nday_vars.append(nday_var)
    out["one_day_exceptions"] = pd.DataFrame({
        "ID":volatility_factor_id,
        "1-Day_VaR": nday_vars,
        "Exception_Q1": r1,
        "Exception_Q2": r2,
        "Exception_Q3": r3,
        "Exception_Q4": r4,
        "Total_Exceptions": np.array(r1) + np.array(r2) + np.array(r3) + np.array(r4)
        })

    # ---- N-day exceptions (cell 14) ----
    quarter = 3
    ExceptionAdjustment = {
        0:0,1:0, 2:1, 3:2, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:3, 11:4, 12:4, 13:4, 14:5, 15:5, 16:5,
        17:5, 18:6, 19:6, 20:6, 21:7, 22:7, 23:8, 24:8, 25:8, 26:9, 27:9, 28:1, 29:1, 30:1, 31:11,
        32:11, 33:12, 34:12, 35:13, 36:36, 37:14, 38:14, 39:15, 40:15
        }
    _lists = get_lists()
    tickers, volatility_factor_id, weights, ndays = _lists.get("tickers"), _lists.get("ids"), _lists.get("weights"), _lists.get("ndays")
    frames = []
    for i in range(len(tickers)):
        for y in range(len(ndays[i])):
            v = GarchSim(tickers = tickers[i], weights = weights[i], sample_start = PERIOD_FULL[0], sample_end = PERIOD_FULL[1])
            var = v.CustomNDayPriceSimulation(horizon1 = 63, horizon2 = 252 + ndays[i][y] - 1, ndays = ndays[i][y], nsims = nsims,
                                            shock_type = "custom_normal", start_vals=[PERIOD_START[0], 0.50])
            nday_var = var["VaR_95"]
            end_date = {3: PERIOD_NEW_3[1], 4: PERIOD_NEW_4[1], 5: PERIOD_NEW_5[1]}[quarter]
            bt = VarBacktest(tickers = tickers[i], n_liquidation = ndays[i][y], VaR = nday_var, start_date=PERIOD_NEW_2[0], end_date=end_date, VaR_is_log = False, forward_shift = False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ind = legacy_loglikelihood_test(bt.break_series)
            TotalExceptions = bt.break_series.sum()
            IsDependent = True if ind.get("Independence") == "RejectNull" else False
            if IsDependent == False:
                Adjustment = 0
            elif TotalExceptions > 40:
                Adjustment = 15
            else:
                Adjustment = ExceptionAdjustment.get(TotalExceptions)
            frames.append(pd.DataFrame({
                "Volatility_Factor_ID": volatility_factor_id[i],
                "Tickers": ", ".join(tickers[i]) if len(tickers[i]) > 1 else tickers[i],
                "Weights": ", ".join([str(x) for x in weights[i]]) if weights[i] != None else '1',
                "NDays": ndays[i][y],
                "NDayVaR": nday_var,
                "ForecastStart": bt.start_date,
                "ForecastEnd": bt.end_date,
                "TotalExceptions": TotalExceptions,
                "IndependenceTest": ind.get("Independence"),
                "IndependenceAdjustment": Adjustment,
                "TotalExceptionsAdjusted": TotalExceptions - Adjustment
            }, index = [0]))
    out["nday_exceptions"] = pd.concat(frames)

    # ---- Shortfall (cell 17) ----
    quarter = 3
    ticker = "G0BA"
    weights = None
    ndays = 10
    v = GarchSim(tickers = ticker, weights = weights, sample_start = PERIOD_FULL[0], sample_end = PERIOD_FULL[1])
    var = v.CustomNDayPriceSimulation(horizon1 = 63, horizon2 = 252 + ndays - 1, ndays = ndays, nsims = nsims,
                                    shock_type = "custom_normal", start_vals=["2022-01-01", 0.50])
    end_date = {3: PERIOD_NEW_3[1], 4: PERIOD_NEW_4[1]}[quarter]
    bt = VarBacktest(tickers = ticker, n_liquidation = ndays, VaR = var["VaR_95"], start_date=PERIOD_NEW_2[0], end_date=end_date, VaR_is_log = False, forward_shift = False)
    out["shortfall"] = bt.shortfall_series.to_frame(name = ticker)
    out["shortfall_var"] = var
    return out


def check_fixture_matches_full_data():
    """Sanity check: truncating the data to the fixture does not change results."""
    VaRData.DATA = FULL_DATA
    a = GarchSim(["G1P0", "G2P0", "CMOS"], [0.5, 0.3, 0.2], *SAMPLE).CustomNDayPriceSimulation(
        horizon1=63, horizon2=261, ndays=10, shock_type="custom_bootstrap", nsims=1000,
        start_vals=START_VALS, bootstrap_start=BOOT[0], bootstrap_end=BOOT[1])
    VaRData.DATA = FIXTURE_DATA
    b = GarchSim(["G1P0", "G2P0", "CMOS"], [0.5, 0.3, 0.2], *SAMPLE).CustomNDayPriceSimulation(
        horizon1=63, horizon2=261, ndays=10, shock_type="custom_bootstrap", nsims=1000,
        start_vals=START_VALS, bootstrap_start=BOOT[0], bootstrap_end=BOOT[1])
    assert a == b, (a, b)
    print("fixture == full data: OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=FIXTURES / "golden")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = {
        "vardata": capture_vardata,
        "garch_fit": capture_garch_fit,
        "simulations": capture_simulations,
        "bootstrap": capture_bootstrap,
        "periods": capture_periods,
        "backtest": capture_backtest,
        "monitoring": lambda: legacy_monitoring(TEST_CONFIGS),
    }
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        print(f"capturing {name} ...", flush=True)
        pd.to_pickle(fn(), args.out / f"{name}.pkl.gz")

    if not args.only or "pipeline" in args.only:
        print("capturing pipeline ...", flush=True)
        main_out = legacy_main_garch(TEST_CONFIGS)
        main_out["bootstrap"] = legacy_main_bootstrap(TEST_CONFIGS, nboot=500)
        bt_out = legacy_backtest_notebook(main_out["wide"], args.out / "_tmp_results.xlsx")
        (args.out / "_tmp_results.xlsx").unlink()
        pd.to_pickle({**main_out, "backtest": bt_out}, args.out / "pipeline.pkl.gz")

    if not args.only:
        check_fixture_matches_full_data()
    print("done")


if __name__ == "__main__":
    main()
