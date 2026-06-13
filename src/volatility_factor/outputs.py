"""Output helpers for Delta table persistence."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

VAR_MODEL_RUNS_TABLE = "prod_silver.volatility_factor.var_model_runs"
VAR_RESULTS_TABLE = "prod_silver.volatility_factor.var_results"
VAR_DIAGNOSTICS_TABLE = "prod_silver.volatility_factor.var_diagnostics"
VAR_BACKTESTS_TABLE = "prod_silver.volatility_factor.var_backtests"
PRICING_QA_TABLE = "prod_silver.volatility_factor.pricing_qa"


def write_pandas_to_delta(
    spark,
    data: pd.DataFrame,
    table_name: str,
    mode: str = "append",
    options: Mapping[str, str] | None = None,
) -> None:
    """Write a pandas DataFrame to a managed Delta table through Spark."""

    if data.empty:
        return

    writer = spark.createDataFrame(data).write.format("delta").mode(mode)
    for key, value in (options or {"mergeSchema": "true"}).items():
        writer = writer.option(key, value)
    writer.saveAsTable(table_name)

