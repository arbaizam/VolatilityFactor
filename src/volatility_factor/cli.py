"""Command-line entrypoint for configured VaR model runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from volatility_factor.config import load_yaml_config
from volatility_factor.data_sources import CsvPricingDataSource, SparkDeltaPricingDataSource
from volatility_factor.orchestration import run_configured_models
from volatility_factor.outputs import (
    PRICING_QA_TABLE,
    VAR_DIAGNOSTICS_TABLE,
    VAR_MODEL_RUNS_TABLE,
    VAR_RESULTS_TABLE,
    write_pandas_to_delta,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run volatility factor VaR models.")
    parser.add_argument(
        "--config",
        default="configs/var_config_volfactor.yml",
        help="Path to YAML model configuration.",
    )
    parser.add_argument(
        "--csv-pricing-data",
        default=None,
        help="Optional local CSV pricing data path. If omitted, reads the Delta pricing table.",
    )
    parser.add_argument(
        "--pricing-table",
        default="prod_silver.volatility_factor.pricing_data",
        help="Unity Catalog pricing table used when --csv-pricing-data is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional local directory for CSV copies of model output tables.",
    )
    parser.add_argument(
        "--write-delta",
        action="store_true",
        help="Write outputs to prod_silver.volatility_factor Delta tables through Spark.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record method failures in diagnostics instead of failing the run.",
    )
    return parser


def _write_local_outputs(output_dir: str | Path, outputs) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    outputs.model_runs.to_csv(path / "var_model_runs.csv", index=False)
    outputs.results.to_csv(path / "var_results.csv", index=False)
    outputs.diagnostics.to_csv(path / "var_diagnostics.csv", index=False)
    outputs.pricing_qa.to_csv(path / "pricing_qa.csv", index=False)


def _write_delta_outputs(outputs) -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    write_pandas_to_delta(spark, outputs.model_runs, VAR_MODEL_RUNS_TABLE)
    write_pandas_to_delta(spark, outputs.results, VAR_RESULTS_TABLE)
    write_pandas_to_delta(spark, outputs.diagnostics, VAR_DIAGNOSTICS_TABLE)
    write_pandas_to_delta(spark, outputs.pricing_qa, PRICING_QA_TABLE)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    if args.csv_pricing_data:
        pricing_source = CsvPricingDataSource(args.csv_pricing_data)
    else:
        pricing_source = SparkDeltaPricingDataSource(args.pricing_table)

    outputs = run_configured_models(
        config=config,
        pricing_source=pricing_source,
        raise_on_error=not args.continue_on_error,
    )
    if args.output_dir:
        _write_local_outputs(args.output_dir, outputs)
    if args.write_delta:
        _write_delta_outputs(outputs)

    print(f"Completed volatility factor run_id={outputs.run_id}")
    print(f"Result rows: {len(outputs.results)}")
    print(f"Pricing QA rows: {len(outputs.pricing_qa)}")
    print(f"Diagnostic rows: {len(outputs.diagnostics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

