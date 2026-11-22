"""Comparisons with the reference VaR, backtesting and monitoring fixtures."""

from __future__ import annotations

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
PATH_SAMPLE = np.ix_([0, 1, 19, 101, 250, 499, 750, 999], [0, 62, 63, 67, 68, 161, 322, 323])
FLOAT_TOLERANCE = 1e-12


def assert_same(actual, expected, path="value", *, approximate=False):
    """Compare recursively, allowing small roundoff for unrounded calculations."""
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=not approximate, obj=path,
            rtol=FLOAT_TOLERANCE, atol=FLOAT_TOLERANCE,
        )
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(
            actual, expected, check_exact=not approximate, obj=path,
            rtol=FLOAT_TOLERANCE, atol=FLOAT_TOLERANCE,
        )
    elif isinstance(expected, np.ndarray):
        assert np.asarray(actual).shape == expected.shape, path
        if approximate and np.issubdtype(expected.dtype, np.inexact):
            np.testing.assert_allclose(
                actual, expected, rtol=FLOAT_TOLERANCE, atol=FLOAT_TOLERANCE, err_msg=path,
            )
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected), f"{path}: keys {list(actual)} != {list(expected)}"
        for k in expected:
            assert_same(actual[k], expected[k], f"{path}[{k!r}]", approximate=approximate)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, e, f"{path}[{i}]", approximate=approximate)
    elif approximate and isinstance(expected, (float, np.floating)):
        assert actual == pytest.approx(expected, rel=FLOAT_TOLERANCE, abs=FLOAT_TOLERANCE, nan_ok=True), path
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual), path
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# --------------------------------------------------------------------------- #
def test_vardata(golden):
    """Verify price inputs, calculated returns, descriptions and factor IDs."""
    g = golden("vardata")
    for key, (tickers, weights) in PORTFOLIOS.items():
        v = VarData(tickers, weights)
        for attr, expected in g[key].items():
            if attr == "volatility_factor_id":
                continue
            assert_same(getattr(v, attr), expected, f"{key}.{attr}", approximate=attr != "price_matrix")
    assert VarData(["G6P0", "G9P0"], [0.5, 0.5]).volatility_factor_id == g["pair"]["volatility_factor_id"]
    with pytest.warns(UserWarning):
        assert VarData("M0A0").volatility_factor_id == g["single"]["volatility_factor_id"]


def test_garch_fit(golden):
    """Verify fitted frames, coefficients, scales and convergence codes."""
    g = golden("garch_fit")
    for key, (tickers, weights) in PORTFOLIOS.items():
        model = GarchSim(tickers, weights, *SAMPLE)
        for attr, expected in g[key].items():
            assert_same(getattr(model, attr), expected, f"{key}.{attr}", approximate=True)
    model = GarchSim("M0A0", None, *SAMPLE, mean_model="Constant")
    for attr, expected in g["single_constant_mean"].items():
        assert_same(getattr(model, attr), expected, f"constant.{attr}", approximate=True)


def test_simulations(golden):
    """Verify rounded VaR and sample paths for every shock type and starting rule."""
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
                assert res.sim_paths.shape == expected["sim_paths_shape"]
                assert_same(res.sim_paths[PATH_SAMPLE], expected["path_sample"],
                            f"{key}/{shock}/{sv}/paths", approximate=True)

    res = GarchSim("M0A0", None, *SAMPLE).simulate(
        horizon1=63, horizon2=503, ndays=252, shock_type="custom_normal", nsims=10000,
        start_vals=["2024-01-01", 0.95], var_p=0.95)
    assert_same(res.to_dict(), g[("single", "custom_normal_10k_252")]["dict"])
    res = GarchSim("M0A0", None, *SAMPLE, mean_model="Constant").simulate(
        horizon1=63, horizon2=261, ndays=10, shock_type="normal", nsims=1000)
    assert_same(res.to_dict(), g[("single_constant_mean", "normal")]["dict"])


def test_bootstrap(golden):
    """Circular block bootstrap results match exactly."""
    g = golden("bootstrap")
    for key, (tickers, weights) in PORTFOLIOS.items():
        res = NonParametricVar(tickers, weights, *SAMPLE).circular_block_bootstrap(10, nboot=500)
        assert_same(res, g[key], key)


def test_periods(golden):
    """Sample/forecast windows and stress windows match exactly."""
    g = golden("periods")
    assert_same(list(create_periods("2024-12-31", 3, 12, 3, 9)), list(g["periods_a"]))
    assert_same(list(create_periods("2025-06-30", 5, 6, 1, 5)), list(g["periods_b"]))
    single = VarData("M0A0").log_return_matrix[SAMPLE[0]:SAMPLE[1]].copy()
    assert get_stress_range_single(single, *SAMPLE, ndays=126) == g["stress_single"]
    multi = VarData(["G1P0", "G2P0", "CMOS"], [0.5, 0.3, 0.2])
    returns = multi.log_return_matrix[SAMPLE[0]:SAMPLE[1]].copy()
    assert get_stress_range_multi(returns, *SAMPLE, weights=multi.weights, ndays=126) == g["stress_multi"]


def test_backtest(golden):
    """Backtest return, exception and shortfall series match exactly."""
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
            approximate = attr in {"nday_price_return", "nday_log_return", "shortfall_series"}
            assert_same(getattr(bt, attr), expected[attr], f"{key}.{attr}", approximate=approximate)


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def garch_run(config, pricing):
    """One full GARCH run on the test config, shared by the pipeline tests."""
    return run_garch_var(config, pricing, verbose=False)


def test_pipeline_garch(golden, garch_run):
    """The GARCH run's wide results, long format and QA match exactly."""
    g = golden("pipeline")
    assert_same(garch_run.results, g["wide"], "wide")
    assert_same(garch_run.long, g["long"], "long")
    assert_same(garch_run.qa, g["qa"], "qa")


def test_pipeline_bootstrap(golden, config, pricing):
    """The configured bootstrap run matches exactly."""
    g = golden("pipeline")
    assert_same(run_circular_bootstrap(config, pricing, nboot=500).results, g["bootstrap"])


def test_pipeline_backtest(golden, garch_run):
    """Verify return details and the observation counts behind the statistical tests."""
    g = golden("pipeline")["backtest"]
    run = garch_run.backtest()
    calculated = ["Price", "Pct_Change_Actual", "ShortfallPct"]
    detail = run.detail.drop(columns="RejectNullIndependence")
    assert_same(detail.drop(columns=calculated), g["detail"].drop(columns=calculated), "detail")
    assert_same(detail[calculated], g["detail"][calculated], "calculated values", approximate=True)
    keys = ["Volatility_Factor_ID", "ForecastStart", "ForecastEnd", "Method", "NDays", "VaRQ"]
    counts = run.detail.groupby(keys)["Breaks"].agg(["count", "sum"])
    tests = run.independence.set_index(keys).sort_index()
    assert_same(tests["Observations"].to_numpy(), counts["count"].to_numpy())
    assert_same(tests["Exceptions"].to_numpy(), counts["sum"].to_numpy())
    assert tests.loc[tests["Observations"].eq(0), ["LR_uc", "Independence"]].isna().all().all()


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def periods():
    """Monitoring windows for a 2024-12-31 sample end."""
    return MonitoringPeriods.from_sample_end("2024-12-31", sample_years=9)


def test_monitoring_periods(periods):
    """Monitoring windows are the expected full sample, last year and quarters."""
    assert periods.full == ("2016-01-01", "2024-12-31")
    assert periods.start == ("2024-01-01", "2024-12-31")
    assert periods.quarters == (
        ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
        ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
        ("2026-01-01", "2026-03-31"),
    )


def test_monitoring_ftest(golden, periods):
    """F-test report matches; configured weights change only the unequal-weight factor."""
    expected = golden("monitoring")["fstats"]
    assert_same(monitoring.f_test_report(periods, use_config_weights=False), expected, approximate=True)
    weighted = monitoring.f_test_report(periods)
    equal_weight_ids = [i for i in expected.index if i != "three_way"]
    assert_same(weighted.loc[equal_weight_ids], expected.loc[equal_weight_ids], approximate=True)
    assert not weighted.loc["three_way"].equals(expected.loc["three_way"])


def test_monitoring_exceptions(golden, periods, config):
    """One-day and n-day exception reports and shortfalls match exactly."""
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
