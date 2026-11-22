"""Fast tests for the public API: data/config resolution, chaining, validation, plots."""

from __future__ import annotations

import math

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
    """Factors, weights, settings and paths are read from the config."""
    assert list(config.factors) == ["G1P0", "G6P0[50];G9P0[50]", "agy_cmo", "three_way"]
    f = config.factor("three_way")
    assert f.tickers == ("G1P0", "G2P0", "CMOS") and f.weights == (0.5, 0.3, 0.2)
    assert config.factor("G1P0").weights is None  # YAML "None" string
    assert config.garch.start_point_quantiles == (0.50, 0.95)
    assert config.pricing_data_path.name == "pricing_sample.csv"
    assert config.find_factor_id(["CMOS"]) == "agy_cmo"
    assert config.find_factor_id(["XXXX"]) is None


def test_config_chaining_is_immutable(config):
    """with_garch/with_factors return copies and leave the original alone."""
    smaller = config.with_garch(nsims=10).with_factors("agy_cmo")
    assert smaller.garch.nsims == 10 and list(smaller.factors) == ["agy_cmo"]
    assert config.garch.nsims == 2000 and len(config.factors) == 4
    with pytest.raises(KeyError, match="Known ids"):
        config.factor("nope")


def test_config_periods(config):
    """Config.periods() returns the expected latest windows."""
    samples, forecasts = config.periods()
    assert samples[0] == ["2016-01-01", "2024-12-31"]
    assert forecasts[0] == ["2025-04-01", "2026-03-31"]


# ----------------------------------------------------------------------- data
def test_data_resolution_order(pricing, config, monkeypatch, tmp_path):
    """Data comes from use_data, then the environment variable, then the config."""
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
    """A DataFrame, a path and a PricingData give the same prices."""
    frame = pricing.filter(tickers=["M0A0"]).frame
    a = VarData("M0A0", data=frame).price_matrix
    b = VarData("M0A0", data=FIXTURE_CSV).price_matrix
    c = VarData("M0A0", data=pricing).price_matrix
    pd.testing.assert_frame_equal(a, c)
    pd.testing.assert_frame_equal(b, c)


def test_missing_ticker_message(pricing):
    """An unknown ticker raises a KeyError naming it."""
    with pytest.raises(KeyError, match="NOPE"):
        _ = VarData("NOPE").price_matrix


def test_from_frame_handles_text_prices():
    """Text prices with thousands separators are parsed."""
    df = pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"], "Ticker": ["X", "X"], "Source": "s",
        "Description": "d", "Price": ["1,000.5", "1,001.5"],
    })
    pdata = PricingData.from_frame(df)
    assert pdata.frame["Price"].tolist() == [1000.5, 1001.5]
    assert pdata.frame["Date"].dtype.kind == "M"


# -------------------------------------------------------------------- VarData
def test_weights_rules():
    """Weight defaults and validation errors."""
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
    """from_id sets tickers, weights and id; descriptions are formatted."""
    v = VarData.from_id("three_way")
    assert v.tickers == ["G1P0", "G2P0", "CMOS"]
    assert v.weight_description == "0.5;0.3;0.2"
    assert v.volatility_factor_id == "three_way"
    assert VarData.from_id("agy_cmo").weight_description == "1"


def test_changing_tickers_clears_cache():
    """Changing tickers recomputes the price matrix."""
    v = VarData("M0A0")
    first = v.price_matrix
    v.tickers = "CMOS"
    v.weights = None
    assert list(v.price_matrix.columns) == ["CMOS"] and first.columns.tolist() == ["M0A0"]


# ---------------------------------------------------------------- chaining
def test_chaining_garch_shares_data_and_cache():
    """VarData.garch() reuses the data and computed matrices."""
    base = VarData.from_id("G6P0[50];G9P0[50]")
    _ = base.price_matrix
    garch = base.garch(*SAMPLE)
    assert isinstance(garch, GarchSim)
    assert garch.data is base.data
    assert garch.__dict__["price_matrix"] is base.__dict__["price_matrix"]
    assert garch.volatility_factor_id == "G6P0[50];G9P0[50]"


def test_garch_fit_exposes_arch_result():
    """GarchFit passes attributes through to the arch result."""
    garch = GarchSim.from_id("agy_cmo", sample_start=SAMPLE[0], sample_end=SAMPLE[1]).fit()
    fit = garch.fits["CMOS"]
    assert fit.aic == fit.result.aic  # delegated attribute
    assert "omega" in fit.params.index
    assert fit.convergence_code == garch.convergence_codes[0]
    assert fit.attempts[-1] == (fit.yscale, fit.convergence_code)
    assert {"AIC", "BIC", "omega", "yscale"} <= set(garch.summary().columns)
    assert "summary" in dir(fit)


def test_changing_sample_refits():
    """Changing the sample window refits the models."""
    garch = GarchSim("CMOS", None, *SAMPLE).fit()
    first = garch.fits["CMOS"]
    garch.with_sample("2015-01-01", "2023-12-31")
    assert garch.fits["CMOS"] is not first
    assert garch.training_data.index.min() >= pd.Timestamp("2015-01-01")


def test_simulate_result_object():
    """VaRResult acts like a dict and keeps the simulated paths."""
    res = GarchSim("CMOS", None, *SAMPLE).simulate(63, 261, 10, "normal", nsims=200)
    assert res["VaR_99"] == res.var(0.99) == res.to_dict()["VaR_99"]
    assert res.sim_paths.shape == (200, 324)
    assert list(res)[:3] == ["Tickers", "Weights", "SampleStart"]
    tagged = res.with_metadata(Method="custom", Volatility_Factor_ID="agy_cmo")
    assert tagged["Method"] == "custom" and res["Method"] == "GarchSim_normal"
    assert list(tagged)[-1] == "Volatility_Factor_ID"


def test_simulate_validation():
    """Bad shock types, missing bootstrap windows and bad dates raise."""
    garch = GarchSim("CMOS", None, *SAMPLE)
    with pytest.raises(ValueError, match="shock_type"):
        garch.simulate(63, 261, 10, "bogus", nsims=10)
    with pytest.raises(ValueError, match="bootstrap_start"):
        garch.simulate(63, 261, 10, "bootstrap", nsims=10)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        GarchSim("CMOS", None, "2016/01/01", SAMPLE[1])


def test_backtest_chaining_resets_results():
    """with_var/with_window clear cached results."""
    bt = VarData("CMOS").backtest(10, var=2.0, start_date="2022-01-01", end_date="2022-12-31")
    breaks_2 = bt.exceptions
    assert bt.with_var(0.5).exceptions > breaks_2
    assert bt.with_window("2023-01-01", "2023-12-31").nday_price_return.index.min().year == 2023
    with pytest.raises(ValueError, match="var is required"):
        _ = VarBacktest("CMOS", 10).var_pct


def test_log_likelihood_transition_statistics():
    """Alternating outcomes have four transitions of each kind and no self-transitions."""
    result = log_likelihood_test([1, 0] * 4 + [1], converge_rate=0.25)
    fitted = 5 * math.log(5 / 9) + 4 * math.log(4 / 9)
    null = 5 * math.log(0.25) + 4 * math.log(0.75)
    assert result["Observations"] == 9
    assert result["Exceptions"] == 5
    assert result["LR_uc"] == round(2 * (fitted - null), 3)
    assert result["LR_ind"] == round(16 * math.log(2), 3)
    assert result["LR_cc"] == round(-8 * math.log(0.25 * 0.75), 3)
    assert result["Independence"] == "RejectNull"


@pytest.mark.parametrize("breaks", [[], [0] * 30, [1] * 30])
def test_log_likelihood_with_no_variation(breaks):
    """Coverage can use constant outcomes; a transition test requires both states."""
    result = log_likelihood_test(breaks)
    assert result["Observations"] == len(breaks)
    assert result["Exceptions"] == sum(breaks)
    assert math.isfinite(result["LR_uc"]) == bool(breaks)
    assert math.isnan(result["LR_ind"])
    assert math.isnan(result["LR_cc"])
    assert result["Independence"] is None


def test_run_backtests_pads_incomplete_window(pricing):
    """Forecast windows past the last price are padded (or not)."""
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
    assert run.independence.set_index("VaRQ")["converge_rate"].to_dict() == {
        "50Q": 50.0, "75Q": 25.0, "95Q": 5.0, "99Q": 1.0,
    }
    assert (run.independence["Observations"] == detail["Breaks"].notna().sum()).all()
    no_pad = run_backtests(results, pad_incomplete=False).detail
    assert no_pad["Date"].max() == last


def test_run_backtests_with_no_observations():
    """A forecast after the available prices has no statistics or independence decision."""
    results = pd.DataFrame([{
        "Volatility_Factor_ID": "agy_cmo", "Tickers": "CMOS", "Weights": "1",
        "ForecastStart": "2030-01-01", "ForecastEnd": "2030-01-05", "Method": "m", "NDays": 10,
        "VaR_50": 1.0, "VaR_75": 1.5, "VaR_95": 2.0, "VaR_99": 3.0,
    }])
    run = run_backtests(results, converge_rate=0.1)
    assert (run.independence["Observations"] == 0).all()
    assert (run.independence["converge_rate"] == 10.0).all()
    assert run.independence[["LR_uc", "LR_ind", "LR_cc", "Independence"]].isna().all().all()
    assert len(run.detail) == 20
    assert run.detail["RejectNullIndependence"].isna().all()
    assert run_backtests(results, pad_incomplete=False).detail.empty


def test_garch_save_without_results(config, pricing, tmp_path):
    """An entirely skipped run publishes QA and keeps the last successful results."""
    cfg = config.with_factors("agy_cmo")
    recent = pricing.filter(tickers=["CMOS"], start="2024-12-26", end="2024-12-31")
    run = vf.run_garch_var(cfg, data=recent, verbose=False)
    assert run.results.empty and not run.qa.empty
    assert run.long.empty
    assert run.backtest().detail.empty
    path = tmp_path / "batch"
    output = [path.with_suffix(".xlsx"), path.with_suffix(".parquet")]
    for file in output:
        file.write_bytes(b"previous results")
    with pytest.raises(ValueError, match="No VaR results to publish"):
        run.save(path)
    assert all(file.read_bytes() == b"previous results" for file in output)
    pd.testing.assert_frame_equal(pd.read_excel(tmp_path / "QA1_batch.xlsx"), run.qa)


def test_monitoring_periods_from_config(config):
    """MonitoringPeriods.from_config uses the config's sample end."""
    p = MonitoringPeriods.from_config(config, n_quarters=2)
    assert p.full == ("2016-01-01", "2024-12-31") and len(p.quarters) == 2


def test_drawdown_report():
    """Check weighted losses, quarter boundaries, and factors without observations."""
    dates = pd.bdate_range("2024-12-17", "2025-01-03").difference(
        pd.DatetimeIndex(["2024-12-25", "2025-01-01"])
    )
    prices = pd.concat([
        pd.DataFrame({
            "Date": dates, "Ticker": ticker, "Source": "fixture", "Description": ticker,
            "Price": [100] * (len(dates) - 2) + final_prices,
        })
        for ticker, final_prices in {"A": [80, 90], "B": [90, 95], "C": [105, 110]}.items()
    ])
    cfg = Config.from_dict({"var_settings": {
        "gain": {"tickers": ["C"], "ndays": [10]},
        "portfolio": {"tickers": ["A", "B"], "weights": [0.75, 0.25], "ndays": [10, 252]},
        "annual": {"tickers": ["B"], "ndays": [252]},
    }})
    periods = MonitoringPeriods.from_sample_end("2024-12-31", n_quarters=2)
    result = vf.monitoring.drawdown_report(periods, quarters=[2, 1], config=cfg, data=prices)
    assert list(result["Quarter"]) == [1, 1, 2, 2]
    assert set(result["Volatility_Factor_ID"]) == {"portfolio", "gain"}

    loss, gain = result.iloc[0], result.iloc[1]
    expected_log_return = 0.75 * math.log(0.8) + 0.25 * math.log(0.9)
    expected_drawdown = 100 * (1 - math.exp(expected_log_return))
    assert loss["Volatility_Factor_ID"] == "portfolio"
    assert loss["Tickers"] == "A; B" and loss["Weights"] == "0.75; 0.25"
    assert loss["Observations"] == 2 and loss["NDays"] == 10
    assert loss["Drawdown Start"] == "2024-12-17"
    assert loss["Drawdown End"] == "2025-01-02"
    assert loss["Worst 10-Day Log Return"] == pytest.approx(expected_log_return)
    assert loss["Realized 10-Day Log Loss"] == pytest.approx(-expected_log_return)
    assert loss["Worst 10-Day Arithmetic Drawdown"] == pytest.approx(expected_drawdown)
    assert gain["Worst 10-Day Log Return"] == pytest.approx(math.log(1.05))
    assert gain["Worst 10-Day Arithmetic Drawdown"] == gain["Realized 10-Day Log Loss"] == 0
    assert gain["Weights"] is None

    unavailable = result[result["Quarter"] == 2]
    assert (unavailable["Observations"] == 0).all()
    assert unavailable.loc[:, "Worst 10-Day Log Return":].isna().all().all()
    empty = vf.monitoring.drawdown_report(
        periods, quarters=[1], config=cfg.with_factors("annual"), data=prices,
    )
    assert empty.empty and list(empty) == list(result)
    with pytest.raises(ValueError, match="Quarter 0"):
        vf.monitoring.drawdown_report(periods, quarters=[0], config=cfg, data=prices)


# ---------------------------------------------------------------------- plots
def test_plots_return_figures():
    """Every chart function returns a Plotly figure with the expected traces."""
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
        assert fig.layout.title.font.color == "#003359"
        assert fig.layout.plot_bgcolor == "#FFFFFF"
    backtest = figs[0]
    assert {t.name.split(" (")[0] for t in backtest.data} >= {"Non-negative return", "Break", "VaR"}
    assert "Backtesting Results" in backtest.layout.title.text
    assert [t.line.color for t in figs[2].data] == ["#003359", "#058AA8"]
    assert [t.line.color for t in figs[5].data] == ["#003359", "#058AA8", "#AAAD00"]


def test_backtest_plot_loss_colors():
    """Highlight breaks in red and keep an exact-threshold loss in the negative series."""
    prices = pd.DataFrame({
        "Date": pd.bdate_range("2025-01-06", periods=4),
        "Ticker": "A", "Source": "fixture", "Description": "Index A", "Price": [100, 75, 90, 45],
    })
    bt = VarBacktest("A", 1, var=25.0, var_is_log=False, data=prices, volatility_factor_id="A")
    positive, negative, breaks, _ = bt.backtest_plot().data
    assert [t.marker.color for t in (positive, negative, breaks)] == ["#93B1CC", "#DA9694", "#D62728"]
    assert list(negative.y) == [-0.25]
    assert list(breaks.y) == [-0.5]
    assert breaks.marker.symbol == "x"


def test_public_api_exports():
    """Every name in __all__ exists."""
    for name in vf.__all__:
        assert hasattr(vf, name), name
    assert isinstance(Config.from_dict({"var_settings": {}}).factors, dict)
    assert np.isfinite(VarData("CMOS").rolling_volatility().iloc[-1])
