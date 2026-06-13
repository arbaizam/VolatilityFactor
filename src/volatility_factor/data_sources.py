"""Pricing data source adapters.

The model code consumes pandas data frames, while Databricks is used at the
input and output boundaries. This keeps the math deterministic and easy to
regression test locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

PRICE_COLUMNS = ("Date", "Ticker", "Source", "Description", "Price")
DELTA_PRICING_TABLE = "prod_silver.volatility_factor.pricing_data"


class PricingDataSource(Protocol):
    """Interface for objects that load normalized pricing data."""

    def load_pricing_data(
        self,
        tickers: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return pricing rows with columns ``Date, Ticker, Source, Description, Price``."""


def normalize_pricing_data(
    data: pd.DataFrame,
    tickers: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Validate, type, filter, and sort pricing data."""

    missing = [column for column in PRICE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Pricing data is missing required columns: {missing}")

    df = data.loc[:, PRICE_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Price"].dtype == object:
        df["Price"] = df["Price"].astype(str).str.replace(",", "", regex=False)
    df["Price"] = pd.to_numeric(df["Price"], errors="raise")
    df["Ticker"] = df["Ticker"].astype(str)
    df["Source"] = df["Source"].astype(str)
    df["Description"] = df["Description"].astype(str)

    if tickers is not None:
        ticker_set = {str(ticker) for ticker in tickers}
        df = df[df["Ticker"].isin(ticker_set)]
    if start is not None:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["Date"] <= pd.to_datetime(end)]

    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


class DataFramePricingDataSource:
    """Pricing source backed by an in-memory pandas DataFrame."""

    def __init__(self, data: pd.DataFrame):
        self._data = normalize_pricing_data(data)

    def load_pricing_data(
        self,
        tickers: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return normalize_pricing_data(self._data, tickers=tickers, start=start, end=end)


class CsvPricingDataSource:
    """Pricing source backed by the legacy CSV file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_pricing_data(
        self,
        tickers: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        data = pd.read_csv(self.path)
        return normalize_pricing_data(data, tickers=tickers, start=start, end=end)


class SparkDeltaPricingDataSource:
    """Pricing source backed by a Unity Catalog Delta table."""

    def __init__(self, table_name: str = DELTA_PRICING_TABLE, spark: object | None = None):
        self.table_name = table_name
        self.spark = spark

    def _get_spark(self):
        if self.spark is not None:
            return self.spark
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise RuntimeError("pyspark is required to use SparkDeltaPricingDataSource") from exc
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

    def load_pricing_data(
        self,
        tickers: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        from pyspark.sql import functions as F

        spark = self._get_spark()
        sdf = spark.table(self.table_name).select(*PRICE_COLUMNS)

        if tickers is not None:
            sdf = sdf.where(sdf["Ticker"].isin([str(ticker) for ticker in tickers]))
        if start is not None:
            start_literal = F.to_date(F.lit(pd.to_datetime(start).date().isoformat()))
            sdf = sdf.where(F.col("Date") >= start_literal)
        if end is not None:
            end_literal = F.to_date(F.lit(pd.to_datetime(end).date().isoformat()))
            sdf = sdf.where(F.col("Date") <= end_literal)

        return normalize_pricing_data(sdf.toPandas(), tickers=tickers, start=start, end=end)
