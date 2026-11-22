"""Exact comparisons with outputs captured from the pre-refactor code.

The goldens in ``tests/fixtures/golden`` were produced by ``capture_legacy_goldens.py``
against commit ``d0d285f``. Every comparison is exact (no tolerance).
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
import pytest

from volatility_factor import (
    GarchSim,
    MonitoringPeriods,
    NonParametricVar,
    VarBacktest,
    VarData,
    create_periods,
    get_stress_range_multi,
    get_stress_range_single,
    log_likelihood_test,
    monitoring,
    run_circular_bootstrap,
    run_garch_var,
)

pytestmark = pytest.mark.regression

PORTFOLIOS = {
    "single": (["M0A0"], None),
    "pair": (["G6P0", "G9P0"], [0.5, 0.5]),
    "three": (["G1P0", "G2P0", "CMOS"], [0.5, 0.3, 0.2]),
}
SAMPLE = ("2016-01-01", "2024-12-31")
START_VALS = ["2024-01-01", 0.5]
BOOT = ("2024-07-01", "2024-12-31")


def assert_same(actual, expected, path="value"):
    """Recursive exact equality for frames, series, arrays, dicts and scalars."""
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, obj=path)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_exact=True, obj=path)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
        assert np.asarray(actual).shape == expected.shape, path
    elif isinstance(expected, dict):
        assert list(actual) == list(expected), f"{path}: keys {list(actual)} != {list(expected)}"
        for k in expected:
            assert_same(actual[k], expected[k], f"{path}[{k!r}]")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual), path
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# --------------------------------------------------------------------------- #
def test_vardata(golden):
    g = golden("vardata")
    for key, (tickers, weights) in PORTFOLIOS.items():
        v = VarData(tickers, weights)
        for attr, expected in g[key].items():
            if attr == "volatility_factor_id":
                continue
            assert_same(getattr(v, attr), expected, f"{key}.{attr}")
    assert VarData(["G6P0", "G9P0"], [0.5, 0.5]).volatility_factor_id == g["pair"]["volatility_factor_id"]
    with pytest.warns(UserWarning):
        assert VarData("M0A0").volatility_factor_id == g["single"]["volatility_factor_id"]


def test_garch_fit(golden):
    g = golden("garch_fit")
    for key, (tickers, weights) in PORTFOLIOS.items():
        model = GarchSim(tickers, weights, *SAMPLE)
        for attr, expected in g[key].items():
            assert_same(getattr(model, attr), expected, f"{key}.{attr}")
    model = GarchSim("M0A0", None, *SAMPLE, mean_model="Constant")
    for attr, expected in g["single_constant_mean"].items():
        assert_same(getattr(model, attr), expected, f"constant.{attr}")


def test_simulations(golden):
    g = golden("simulations")
    for key, (tickers, weights) in PORTFOLIOS.items():
        model = GarchSim(tickers, weights, *SAMPLE)
        for shock in ["normal", "custom_normal", "bootstrap", "custom_bootstrap"]:
            for sv in [None, START_VALS]:
                kwargs = {}
                if shock in ("bootstrap", "custom_bootstrap"):
                    kwargs = {"bootstrap_start": BOOT[0], "bootstrap_end": BOOT[1]}
                res = model.simulate(horizon1=63, horizon2=261, ndays=10, shock_type=shock,
                                     nsims=1000, start_vals=sv, **kwargs)
                expected = g[(key, shock, sv is not None)]
                assert_same(res.to_dict(), expected["dict"], f"{key}/{shock}/{sv}")
                paths = np.ascontiguousarray(res.sim_paths)
                assert paths.shape == expected["sim_paths_shape"]
                assert hashlib.sha256(paths.tobytes()).hexdigest() == expected["sim_paths_sha256"]

    res = GarchSim("M0A0", None, *SAMPLE).simulate(
        horizon1=63, horizon2=503, ndays=252, shock_type="custom_normal", nsims=10000,
        start_vals=["2024-01-01", 0.95], var_p=0.95)
    assert_same(res.to_dict(), g[("single", "custom_normal_10k_252")]["dict"])
    res = GarchSim("M0A0", None, *SAMPLE, mean_model="Constant").simulate(
        horizon1=63, horizon2=261, ndays=10, shock_type="normal", nsims=1000)
    assert_same(res.to_dict(), g[("single_constant_mean", "normal")]["dict"])


def test_legacy_alias_matches(golden):
    expected = golden("simulations")[("pair", "custom_normal", True)]["dict"]
    model = GarchSim(["G6P0", "G9P0"], [0.5, 0.5], *SAMPLE)
    with pytest.warns(DeprecationWarning):
        assert_same(model.CustomNDayPriceSimulation(63, 261, 10, "custom_normal", 1000,
                                                    start_vals=START_VALS), expected)


def test_bootstrap(golden):
    g = golden("bootstrap")
    for key, (tickers, weights) in PORTFOLIOS.items():
        res = NonParametricVar(tickers, weights, *SAMPLE).circular_block_bootstrap(10, nboot=500)
        assert_same(res, g[key], key)


def test_periods(golden):
    g = golden("periods")
    assert_same(list(create_periods("2024-12-31", 3, 12, 3, 9)), list(g["periods_a"]))
    assert_same(list(create_periods("2025-06-30", 5, 6, 1, 5)), list(g["periods_b"]))
    single = VarData("M0A0").log_return_matrix[SAMPLE[0]:SAMPLE[1]].copy()
    assert get_stress_range_single(single, *SAMPLE, ndays=126) == g["stress_single"]
    multi = VarData(["G1P0", "G2P0", "CMOS"], [0.5, 0.3, 0.2])
    returns = multi.log_return_matrix[SAMPLE[0]:SAMPLE[1]].copy()
    assert get_stress_range_multi(returns, *SAMPLE, weights=multi.weights, ndays=126) == g["stress_multi"]


def test_backtest(golden):
    g = golden("backtest")
    cases = {
        "single_log": dict(tickers="M0A0", n_liquidation=10, var=3.0, start_date="2020-01-01",
                           end_date="2022-09-30"),
        "single_pct_fwd": dict(tickers="M0A0", n_liquidation=10, var=2.5, start_date="2022-01-01",
                               end_date="2023-12-31", var_is_log=False, forward_shift=True),
        "pair_pct": dict(tickers=["G6P0", "G9P0"], weights=[0.5, 0.5], n_liquidation=5, var=1.0,
                         start_date="2022-01-01", end_date="2024-12-31", var_is_log=False),
        "three_log_fwd": dict(tickers=["G1P0", "G2P0", "CMOS"], weights=[0.5, 0.3, 0.2],
                              n_liquidation=1, var=0.5, start_date="2020-01-01",
                              end_date="2025-06-30", forward_shift=True),
        "open_dates": dict(tickers="CMOS", n_liquidation=10, var=2.0),
    }
    for key, kw in cases.items():
        bt = VarBacktest(**kw)
        expected = g[key]
        for attr in ["var_pct", "nday_price_return", "nday_log_return", "break_series",
                     "shortfall_series", "duration"]:
            assert_same(getattr(bt, attr), expected[attr], f"{key}.{attr}")
        assert_same(bt.independence_test(), expected["llt"], f"{key}.llt")
        assert_same(bt.independence_test(0.05, 0.95), expected["llt_95"], f"{key}.llt_95")

    idx = pd.date_range("2024-01-01", periods=30)
    edge = {
        "none": pd.Series(0, index=idx),
        "all": pd.Series(1, index=idx),
        "first": pd.Series([1] + [0] * 29, index=idx),
        "cluster": pd.Series([0] * 10 + [1, 1, 1] + [0] * 17, index=idx),
        "alt": pd.Series([0, 1] * 15, index=idx),
    }
    for key, series in edge.items():
        assert_same(log_likelihood_test(series), g["llt_edge"][key], f"edge.{key}")


def test_legacy_backtest_keywords_still_work(golden):
    with pytest.warns(DeprecationWarning):
        bt = VarBacktest(tickers="M0A0", n_liquidation=10, VaR=3.0, start_date="2020-01-01",
                         end_date="2022-09-30")
    assert_same(bt.break_series, golden("backtest")["single_log"]["break_series"])


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def garch_run(config, pricing):
    return run_garch_var(config, pricing, verbose=False)


def test_pipeline_garch(golden, garch_run):
    g = golden("pipeline")
    assert_same(garch_run.results, g["wide"], "wide")
    assert_same(garch_run.long, g["long"], "long")
    assert_same(garch_run.qa, g["qa"], "qa")


def test_pipeline_bootstrap(golden, config, pricing):
    g = golden("pipeline")
    assert_same(run_circular_bootstrap(config, pricing, nboot=500).results, g["bootstrap"])


def test_pipeline_backtest(golden, garch_run):
    g = golden("pipeline")["backtest"]
    run = garch_run.backtest()
    assert_same(run.detail, g["detail"], "detail")
    legacy_cols = list(g["independence"].columns)
    assert_same(run.independence[legacy_cols], g["independence"], "independence")


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def periods():
    return MonitoringPeriods.from_sample_end("2024-12-31", sample_years=9)


def test_monitoring_periods(periods):
    assert periods.full == ("2016-01-01", "2024-12-31")
    assert periods.start == ("2024-01-01", "2024-12-31")
    assert periods.quarters == (
        ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
        ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
        ("2026-01-01", "2026-03-31"),
    )


def test_monitoring_ftest(golden, periods):
    expected = golden("monitoring")["fstats"]
    # Original behaviour: equal weights.
    assert_same(monitoring.f_test_report(periods, use_config_weights=False), expected)
    # Default: configured weights. Only the unequal-weight factor changes.
    weighted = monitoring.f_test_report(periods)
    equal_weight_ids = [i for i in expected.index if i != "three_way"]
    assert_same(weighted.loc[equal_weight_ids], expected.loc[equal_weight_ids])
    assert not weighted.loc["three_way"].equals(expected.loc["three_way"])


def test_monitoring_exceptions(golden, periods, config):
    g = golden("monitoring")
    nsims = config.garch.nsims
    assert_same(
        monitoring.one_day_exceptions(periods, nsims=nsims, use_config_weights=False),
        g["one_day_exceptions"],
    )
    assert_same(
        monitoring.nday_exceptions(periods, quarter=3, nsims=nsims, use_config_weights=False),
        g["nday_exceptions"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shortfall = monitoring.shortfall_report("G0BA", periods, quarter=3, nsims=nsims)
    assert_same(shortfall, g["shortfall"])
