"""Fast tests for the public API: data/config resolution, chaining, validation, plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import volatility_factor as vf
from volatility_factor import (
    Config,
    GarchSim,
    MonitoringPeriods,
    PricingData,
    VarBacktest,
    VarData,
    get_data,
    log_likelihood_test,
    run_backtests,
    set_data,
    use_data,
)
from volatility_factor.data import DATA_ENV_VAR

FIXTURE_CSV = "tests/fixtures/pricing_sample.csv"
SAMPLE = ("2016-01-01", "2024-12-31")


# --------------------------------------------------------------------- config
def test_config_parsing(config):
    assert list(config.factors) == ["G1P0", "G6P0[50];G9P0[50]", "agy_cmo", "three_way"]
    f = config.factor("three_way")
    assert f.tickers == ("G1P0", "G2P0", "CMOS") and f.weights == (0.5, 0.3, 0.2)
    assert config.factor("G1P0").weights is None  # YAML "None" string
    assert config.garch.var_type == ("long_term", "normal", "short_term", "stress_var")
    assert config.pricing_data_path.name == "pricing_sample.csv"
    assert config.find_factor_id(["CMOS"]) == "agy_cmo"
    assert config.find_factor_id(["XXXX"]) is None


def test_config_chaining_is_immutable(config):
    smaller = config.with_garch(nsims=10).with_factors("agy_cmo")
    assert smaller.garch.nsims == 10 and list(smaller.factors) == ["agy_cmo"]
    assert config.garch.nsims == 2000 and len(config.factors) == 4
    with pytest.raises(KeyError, match="Known ids"):
        config.factor("nope")


def test_config_periods(config):
    samples, forecasts = config.periods()
    assert samples[0] == ["2016-01-01", "2024-12-31"]
    assert forecasts[0] == ["2025-04-01", "2026-03-31"]


# ----------------------------------------------------------------------- data
def test_data_resolution_order(pricing, config, monkeypatch, tmp_path):
    small = pricing.filter(tickers=["CMOS"])
    assert get_data() is pricing  # set by the autouse fixture
    with use_data(small):
        assert get_data() is small
        assert VarData("CMOS").data is small
    assert get_data() is pricing

    set_data(None)
    path = tmp_path / "prices.csv"
    small.frame.to_csv(path, index=False)
    monkeypatch.setenv(DATA_ENV_VAR, str(path))
    assert get_data().tickers == ["CMOS"]
    monkeypatch.delenv(DATA_ENV_VAR)
    # falls back to config.data_settings.pricing_data
    assert len(get_data(config=config)) == len(pricing)


def test_data_argument_forms(pricing):
    frame = pricing.filter(tickers=["M0A0"]).frame
    a = VarData("M0A0", data=frame).price_matrix
    b = VarData("M0A0", data=FIXTURE_CSV).price_matrix
    c = VarData("M0A0", data=pricing).price_matrix
    pd.testing.assert_frame_equal(a, c)
    pd.testing.assert_frame_equal(b, c)


def test_missing_ticker_message(pricing):
    with pytest.raises(KeyError, match="NOPE"):
        _ = VarData("NOPE").price_matrix


def test_from_frame_handles_text_prices():
    df = pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"], "Ticker": ["X", "X"], "Source": "s",
        "Description": "d", "Price": ["1,000.5", "1,001.5"],
    })
    pdata = PricingData.from_frame(df)
    assert pdata.frame["Price"].tolist() == [1000.5, 1001.5]
    assert pdata.frame["Date"].dtype.kind == "M"


# -------------------------------------------------------------------- VarData
def test_weights_rules():
    assert VarData("M0A0").weights.tolist() == 1
    assert VarData("M0A0", weights=2).weights.tolist() == 1  # ignored for one ticker
    assert VarData(["G6P0", "G9P0"]).weights.tolist() == [0.5, 0.5]
    assert VarData(["G1P0", "G2P0", "CMOS"], [0.7, 0.2, 0.1]).weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="add to one"):
        VarData(["G6P0", "G9P0"], [1, 2])
    with pytest.raises(ValueError, match="Expected 2 weights"):
        VarData(["G6P0", "G9P0"], [1.0])
    with pytest.raises(ValueError, match="Valid types"):
        VarData({"G6P0"})


def test_from_id_and_descriptions():
    v = VarData.from_id("three_way")
    assert v.tickers == ["G1P0", "G2P0", "CMOS"]
    assert v.weight_description == "0.5;0.3;0.2"
    assert v.volatility_factor_id == "three_way"
    assert VarData.from_id("agy_cmo").weight_description == "1"


def test_changing_tickers_clears_cache():
    v = VarData("M0A0")
    first = v.price_matrix
    v.tickers = "CMOS"
    v.weights = None
    assert list(v.price_matrix.columns) == ["CMOS"] and first.columns.tolist() == ["M0A0"]


# ---------------------------------------------------------------- chaining
def test_chaining_garch_shares_data_and_cache():
    base = VarData.from_id("G6P0[50];G9P0[50]")
    _ = base.price_matrix
    garch = base.garch(*SAMPLE)
    assert isinstance(garch, GarchSim)
    assert garch.data is base.data
    assert garch.__dict__["price_matrix"] is base.__dict__["price_matrix"]
    assert garch.volatility_factor_id == "G6P0[50];G9P0[50]"


def test_garch_fit_exposes_arch_result():
    garch = GarchSim.from_id("agy_cmo", sample_start=SAMPLE[0], sample_end=SAMPLE[1]).fit()
    fit = garch.fits["CMOS"]
    assert fit.aic == fit.result.aic  # delegated attribute
    assert "omega" in fit.params.index
    assert fit.convergence_code == garch.convergence_codes[0]
    assert fit.attempts[-1] == (fit.yscale, fit.convergence_code)
    assert {"AIC", "BIC", "omega", "yscale"} <= set(garch.summary().columns)
    assert "summary" in dir(fit)


def test_changing_sample_refits():
    garch = GarchSim("CMOS", None, *SAMPLE).fit()
    first = garch.fits["CMOS"]
    garch.with_sample("2015-01-01", "2023-12-31")
    assert garch.fits["CMOS"] is not first
    assert garch.training_data.index.min() >= pd.Timestamp("2015-01-01")


def test_simulate_result_object():
    res = GarchSim("CMOS", None, *SAMPLE).simulate(63, 261, 10, "normal", nsims=200)
    assert res["VaR_99"] == res.var(0.99) == res.to_dict()["VaR_99"]
    assert res.sim_paths.shape == (200, 324)
    assert list(res)[:3] == ["Tickers", "Weights", "SampleStart"]
    tagged = res.with_metadata(Method="custom", Volatility_Factor_ID="agy_cmo")
    assert tagged["Method"] == "custom" and res["Method"] == "GarchSim_normal"
    assert list(tagged)[-1] == "Volatility_Factor_ID"


def test_simulate_validation():
    garch = GarchSim("CMOS", None, *SAMPLE)
    with pytest.raises(ValueError, match="shock_type"):
        garch.simulate(63, 261, 10, "bogus", nsims=10)
    with pytest.raises(ValueError, match="bootstrap_start"):
        garch.simulate(63, 261, 10, "bootstrap", nsims=10)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        GarchSim("CMOS", None, "2016/01/01", SAMPLE[1])


def test_backtest_chaining_resets_results():
    bt = VarData("CMOS").backtest(10, var=2.0, start_date="2022-01-01", end_date="2022-12-31")
    breaks_2 = bt.exceptions
    assert bt.with_var(0.5).exceptions > breaks_2
    assert bt.with_window("2023-01-01", "2023-12-31").nday_price_return.index.min().year == 2023
    with pytest.raises(ValueError, match="var is required"):
        _ = VarBacktest("CMOS", 10).var_pct


def test_log_likelihood_accepts_lists():
    assert log_likelihood_test([0, 0, 1, 0])["Exceptions"] == 1


def test_run_backtests_pads_incomplete_window(pricing):
    last = pricing.date_range("CMOS")[1]
    results = pd.DataFrame([{
        "Volatility_Factor_ID": "agy_cmo", "Tickers": "CMOS", "Weights": "1",
        "ForecastStart": "2025-04-01", "ForecastEnd": "2026-03-31", "Method": "m", "NDays": 10,
        "VaR_50": 1.0, "VaR_75": 1.5, "VaR_95": 2.0, "VaR_99": 3.0,
    }])
    run = run_backtests(results)
    detail = run.detail[run.detail["VaRQ"] == "99Q"]
    assert detail["Date"].max() == pd.Timestamp("2026-03-31")
    assert detail.loc[detail["Date"] > last, "Price"].isna().all()
    assert len(run.independence) == 4 and set(run.independence["VaRQ"]) == {"50Q", "75Q", "95Q", "99Q"}
    no_pad = run_backtests(results, pad_incomplete=False).detail
    assert no_pad["Date"].max() == last


def test_monitoring_periods_from_config(config):
    p = MonitoringPeriods.from_config(config, n_quarters=2)
    assert p.full == ("2016-01-01", "2024-12-31") and len(p.quarters) == 2


# ---------------------------------------------------------------------- plots
def test_plots_return_figures():
    bt = VarData.from_id("G6P0[50];G9P0[50]").backtest(
        10, var=1.0, start_date="2022-01-01", end_date="2023-12-31", var_is_log=False
    )
    figs = [
        bt.backtest_plot(),
        bt.price_change_plot(),
        bt.price_change_plot(weighted=False),
        bt.price_plot(),
        bt.price_plot(weighted=True),
        VarData("CMOS").plot_rolling_volatility(
            averages={"2024 - Avg": ("2024-01-01", None), "Pre2022 Avg": (None, "2021-12-31")}
        ),
    ]
    garch = GarchSim("CMOS", None, *SAMPLE)
    figs.append(garch.plot_fit())
    figs.append(garch.simulate(63, 261, 10, "normal", nsims=200).plot_distribution())
    for fig in figs:
        assert isinstance(fig, go.Figure) and len(fig.data) >= 1
    backtest = figs[0]
    assert {t.name.split(" (")[0] for t in backtest.data} >= {"Positive return", "Break", "VaR"}
    assert "Backtesting Results" in backtest.layout.title.text
    assert len(figs[2].data) == 2  # one line per index


def test_public_api_exports():
    for name in vf.__all__:
        assert hasattr(vf, name), name
    assert isinstance(Config.from_dict({"var_settings": {}}).factors, dict)
    assert np.isfinite(VarData("CMOS").rolling_volatility().iloc[-1])
