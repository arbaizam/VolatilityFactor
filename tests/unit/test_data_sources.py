from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatility_factor.data_sources import DataFramePricingDataSource, normalize_pricing_data
from volatility_factor.portfolio import PortfolioPricingData


def sample_pricing_data(days: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    rows = []
    for idx, date in enumerate(dates):
        rows.append(
            {
                "Date": date,
                "Ticker": "AAA",
                "Source": "TEST",
                "Description": "AAA Index",
                "Price": 100 * (1.001 ** idx),
            }
        )
        rows.append(
            {
                "Date": date,
                "Ticker": "BBB",
                "Source": "TEST",
                "Description": "BBB Index",
                "Price": 200 * (0.999 ** idx),
            }
        )
    return pd.DataFrame(rows)


def test_normalize_pricing_data_filters_and_types():
    data = sample_pricing_data()
    data["Price"] = data["Price"].astype(object)
    data.loc[0, "Price"] = "100,000.25"

    normalized = normalize_pricing_data(data, tickers=["AAA"], start="2024-01-03")

    assert set(normalized["Ticker"]) == {"AAA"}
    assert normalized["Date"].min() >= pd.Timestamp("2024-01-03")
    assert np.issubdtype(normalized["Price"].dtype, np.number)


def test_dataframe_pricing_source_returns_filtered_copy():
    source = DataFramePricingDataSource(sample_pricing_data())

    filtered = source.load_pricing_data(tickers=["BBB"], end="2024-01-10")

    assert set(filtered["Ticker"]) == {"BBB"}
    assert filtered["Date"].max() <= pd.Timestamp("2024-01-10")


def test_portfolio_pricing_data_weighted_returns():
    portfolio = PortfolioPricingData(
        tickers=["AAA", "BBB"],
        weights=[0.25, 0.75],
        pricing_data=sample_pricing_data(),
        volatility_factor_id="TEST",
    )

    assert portfolio.single_index is False
    assert list(portfolio.price_matrix.columns) == ["AAA", "BBB"]
    assert portfolio.wtd_log_return.shape[0] == portfolio.log_return_matrix.shape[0]


def test_portfolio_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="Weights must add to one"):
        PortfolioPricingData(
            tickers=["AAA", "BBB"],
            weights=[0.25, 0.25],
            pricing_data=sample_pricing_data(),
        )
