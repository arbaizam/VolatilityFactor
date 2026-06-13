"""Regression check against the current legacy VaR output parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from volatility_factor.config import load_yaml_config
from volatility_factor.data_sources import CsvPricingDataSource
from volatility_factor.orchestration import run_configured_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare package GARCH output to legacy output.")
    parser.add_argument("--config", default="configs/var_config_volfactor.yml")
    parser.add_argument("--csv-pricing-data", default="data/pricing_data_combined.csv")
    parser.add_argument(
        "--baseline",
        default="output/combined_vars_2024-12-31.xlsx.parquet",
        help="Legacy long-form parquet produced by volatility_factor_main.ipynb.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser


def _comparison_frame(new_results: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    new = new_results.rename(
        columns={
            "volatility_factor_id": "Volatility_Factor_ID",
            "method": "Method",
            "sample_start": "SampleStart",
            "sample_end": "SampleEnd",
            "forecast_start": "ForecastStart",
            "forecast_end": "ForecastEnd",
            "liquidation_days": "NDays",
            "confidence_label": "VaRQ",
            "var_value": "new_VaR",
        }
    )
    key_cols = [
        "Volatility_Factor_ID",
        "Method",
        "SampleStart",
        "SampleEnd",
        "ForecastStart",
        "ForecastEnd",
        "NDays",
        "VaRQ",
    ]
    new = new[key_cols + ["new_VaR"]].copy()
    old = baseline[key_cols + ["VaR"]].rename(columns={"VaR": "legacy_VaR"}).copy()
    for column in ["SampleStart", "SampleEnd", "ForecastStart", "ForecastEnd"]:
        new[column] = pd.to_datetime(new[column])
        old[column] = pd.to_datetime(old[column])
    comparison = old.merge(new, on=key_cols, how="outer", indicator=True)
    comparison["abs_diff"] = (comparison["legacy_VaR"] - comparison["new_VaR"]).abs()
    return comparison


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    pricing_source = CsvPricingDataSource(args.csv_pricing_data)
    outputs = run_configured_models(
        config=config,
        pricing_source=pricing_source,
        additional_methods=[],
        raise_on_error=True,
    )
    baseline = pd.read_parquet(args.baseline)
    comparison = _comparison_frame(outputs.results, baseline)

    unmatched = comparison[comparison["_merge"] != "both"]
    exceeded = comparison[comparison["abs_diff"] > args.tolerance]
    print(f"Compared rows: {len(comparison)}")
    print(f"Unmatched rows: {len(unmatched)}")
    print(f"Rows over tolerance: {len(exceeded)}")
    if not unmatched.empty:
        print(unmatched.head(20).to_string(index=False))
    if not exceeded.empty:
        print(exceeded.sort_values("abs_diff", ascending=False).head(20).to_string(index=False))
    if not unmatched.empty or not exceeded.empty:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

