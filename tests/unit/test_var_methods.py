from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatility_factor.methods.filtered_historical import FilteredHistoricalSimulationVaR
from volatility_factor.methods.historical import CircularBlockBootstrapVaR, HistoricalSimulationVaR
from volatility_factor.backtesting import VaRBacktest


def sample_pricing_data(days: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=days, freq="B")
    rng = np.random.RandomState(42)
    returns_a = rng.normal(loc=0.0002, scale=0.006, size=days)
    returns_b = rng.normal(loc=0.0001, scale=0.004, size=days)
    prices_a = 100 * np.exp(np.cumsum(returns_a))
    prices_b = 200 * np.exp(np.cumsum(returns_b))
    rows = []
    for date, price_a, price_b in zip(dates, prices_a, prices_b):
        rows.append(
            {
                "Date": date,
                "Ticker": "AAA",
                "Source": "TEST",
                "Description": "AAA Index",
                "Price": price_a,
            }
        )
        rows.append(
            {
                "Date": date,
                "Ticker": "BBB",
                "Source": "TEST",
                "Description": "BBB Index",
                "Price": price_b,
            }
        )
    return pd.DataFrame(rows)


def base_kwargs():
    return {
        "tickers": ["AAA", "BBB"],
        "weights": [0.4, 0.6],
        "pricing_data": sample_pricing_data(),
        "sample_start": "2023-02-01",
        "sample_end": "2023-12-15",
        "volatility_factor_id": "TEST",
    }


def test_historical_simulation_returns_long_records():
    result = HistoricalSimulationVaR(**base_kwargs()).run(n_liquidation=10)
    records = result.to_records()

    assert result.method == "historical_simulation"
    assert len(records) == 4
    assert {record["confidence_label"] for record in records} == {"50Q", "75Q", "95Q", "99Q"}
    assert all(np.isfinite(record["var_value"]) for record in records)


def test_circular_block_bootstrap_is_deterministic_for_seed():
    kwargs = base_kwargs()
    first = CircularBlockBootstrapVaR(**kwargs, random_seed=7).run(
        n_liquidation=5,
        nboot=100,
        confidence_levels=[0.95],
    )
    second = CircularBlockBootstrapVaR(**kwargs, random_seed=7).run(
        n_liquidation=5,
        nboot=100,
        confidence_levels=[0.95],
    )

    assert first.var_by_confidence[0.95] == second.var_by_confidence[0.95]


def test_filtered_historical_simulation_returns_requested_confidence_level():
    result = FilteredHistoricalSimulationVaR(**base_kwargs(), random_seed=11).run(
        n_liquidation=10,
        confidence_levels=[0.99],
        n_sims=250,
    )

    assert set(result.var_by_confidence) == {0.99}
    assert result.diagnostics["standardized_residuals"] > 0


def test_backtest_summary_counts_observations_and_exceptions():
    bt = VaRBacktest(
        tickers=["AAA", "BBB"],
        weights=[0.4, 0.6],
        pricing_data=sample_pricing_data(),
        n_liquidation=5,
        start_date="2023-02-01",
        end_date="2023-12-15",
        var_value=1.0,
        var_is_log=False,
        confidence_level=0.99,
        volatility_factor_id="TEST",
    )
    summary = bt.summary()

    assert summary.observations == bt.exception_series.shape[0]
    assert 0 <= summary.exception_rate <= 1
    assert summary.expected_exception_rate == pytest.approx(0.01)


def test_scipy_backed_methods_smoke():
    pytest.importorskip("scipy")

    from volatility_factor.methods.cornish_fisher import CornishFisherVaR
    from volatility_factor.methods.evt import EVTPeaksOverThresholdVaR
    from volatility_factor.methods.ewma import EWMAVaR
    from volatility_factor.methods.parametric import ParametricVaR

    kwargs = base_kwargs()
    assert ParametricVaR(**kwargs).run(n_liquidation=10, confidence_levels=[0.95]).var_by_confidence
    assert EWMAVaR(**kwargs).run(n_liquidation=10, confidence_levels=[0.95]).var_by_confidence
    assert CornishFisherVaR(**kwargs).run(
        n_liquidation=10,
        confidence_levels=[0.95],
    ).var_by_confidence
    assert EVTPeaksOverThresholdVaR(**kwargs).run(
        n_liquidation=10,
        confidence_levels=[0.99],
        min_exceedances=5,
    ).var_by_confidence
