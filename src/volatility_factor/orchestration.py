"""Config-driven model orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import traceback
from typing import Any, Sequence
from uuid import uuid4

import pandas as pd

from volatility_factor.config import FactorConfig, iter_factor_configs
from volatility_factor.data_sources import PricingDataSource
from volatility_factor.methods.cornish_fisher import CornishFisherVaR
from volatility_factor.methods.evt import EVTPeaksOverThresholdVaR
from volatility_factor.methods.ewma import EWMAVaR
from volatility_factor.methods.filtered_historical import FilteredHistoricalSimulationVaR
from volatility_factor.methods.garch import GarchSimulation
from volatility_factor.methods.historical import CircularBlockBootstrapVaR, HistoricalSimulationVaR
from volatility_factor.methods.parametric import ParametricVaR
from volatility_factor.periods import create_periods, get_stress_range_multi, get_stress_range_single
from volatility_factor.results import legacy_garch_dict_to_records

ADDITIONAL_METHODS = (
    "historical_simulation",
    "parametric_variance_covariance",
    "ewma",
    "cornish_fisher",
    "filtered_historical_simulation",
    "evt_peaks_over_threshold",
    "circular_block_bootstrap",
)


@dataclass
class RunOutputs:
    """DataFrames produced by one configured model run."""

    run_id: str
    results: pd.DataFrame
    diagnostics: pd.DataFrame
    pricing_qa: pd.DataFrame
    model_runs: pd.DataFrame


def _config_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return config.get(section, {}).get(key, default)


def _base_model_kwargs(
    factor: FactorConfig,
    pricing_data: pd.DataFrame,
    sample_start: str,
    sample_end: str,
) -> dict[str, Any]:
    return {
        "tickers": factor.tickers,
        "weights": factor.weights,
        "pricing_data": pricing_data,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "volatility_factor_id": factor.volatility_factor_id,
    }


def _validate_sample_coverage(
    model: GarchSimulation,
    factor: FactorConfig,
    sample_start: str,
    sample_end: str,
) -> dict[str, Any] | None:
    first_price_date = model.price_matrix.first_valid_index() - pd.offsets.Day(4)
    last_price_date = model.price_matrix.last_valid_index() + pd.offsets.Day(4)
    if first_price_date > pd.to_datetime(sample_start) or last_price_date < pd.to_datetime(sample_end):
        return {
            "volatility_factor_id": factor.volatility_factor_id,
            "tickers": model.ticker_description,
            "sample_start": sample_start,
            "sample_end": sample_end,
            "first_price_date": first_price_date.strftime("%Y-%m-%d"),
            "last_price_date": last_price_date.strftime("%Y-%m-%d"),
            "qa_check": "sample_coverage",
            "status": "failed",
        }
    return None


def _stamp_records(
    records: list[dict[str, Any]],
    run_id: str,
    as_of_date: str | None,
    forecast_start: str | None,
    forecast_end: str | None,
) -> list[dict[str, Any]]:
    for record in records:
        record["run_id"] = run_id
        record["as_of_date"] = as_of_date
        record["forecast_start"] = forecast_start
        record["forecast_end"] = forecast_end
    return records


def _run_garch_variants(
    factor: FactorConfig,
    pricing_data: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    forecast_start: str,
    forecast_end: str,
    nday: int,
    config: dict[str, Any],
    start_point_quantile: float,
    run_id: str,
    as_of_date: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    garch_settings = config["garch_settings"]
    var_type = garch_settings.get("var_type", [])
    horizon1 = int(garch_settings["horizon1"])
    horizon2 = int(garch_settings["horizon2"]) + nday - 1
    nsims = int(garch_settings["nsims"])

    garch = GarchSimulation(**_base_model_kwargs(factor, pricing_data, sample_start, sample_end))
    records: list[dict[str, Any]] = []
    start_point_date = (pd.to_datetime(sample_end) - pd.offsets.MonthBegin(12)).strftime("%Y-%m-%d")
    start_vals = [start_point_date, start_point_quantile]

    if "long_term" in var_type:
        result = garch.CustomNDayPriceSimulation(
            horizon1=horizon1,
            horizon2=horizon2,
            ndays=nday,
            nsims=nsims,
            shock_type="custom_normal",
            start_vals=start_vals,
        )
        result["Method"] = f"long_term_var_{round(start_point_quantile * 100)}"
        records.extend(
            legacy_garch_dict_to_records(
                result,
                volatility_factor_id=factor.volatility_factor_id,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                run_id=run_id,
                as_of_date=as_of_date,
            )
        )

    if "normal" in var_type:
        result = garch.CustomNDayPriceSimulation(
            horizon1=horizon1,
            horizon2=horizon2,
            ndays=nday,
            nsims=nsims,
            shock_type="normal",
            start_vals=start_vals,
        )
        result["Method"] = f"normal_var_{round(start_point_quantile * 100)}"
        records.extend(
            legacy_garch_dict_to_records(
                result,
                volatility_factor_id=factor.volatility_factor_id,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                run_id=run_id,
                as_of_date=as_of_date,
            )
        )

    if "short_term" in var_type:
        bootstrap_start = (pd.to_datetime(sample_end) - pd.offsets.MonthBegin(6)).strftime("%Y-%m-%d")
        result = garch.CustomNDayPriceSimulation(
            horizon1=horizon1,
            horizon2=horizon2,
            ndays=nday,
            nsims=nsims,
            shock_type="custom_bootstrap",
            bootstrap_start=bootstrap_start,
            bootstrap_end=sample_end,
            start_vals=start_vals,
        )
        result["Method"] = f"short_term_var_{round(start_point_quantile * 100)}"
        records.extend(
            legacy_garch_dict_to_records(
                result,
                volatility_factor_id=factor.volatility_factor_id,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                run_id=run_id,
                as_of_date=as_of_date,
            )
        )

    if "stress_var" in var_type:
        returns = garch.log_return_matrix[sample_start:sample_end].copy()
        if garch.single_index:
            stress_periods = get_stress_range_single(
                start=sample_start,
                end=sample_end,
                ndays=126,
                returns=returns,
                alpha=0.99,
            )
        else:
            stress_periods = get_stress_range_multi(
                start=sample_start,
                end=sample_end,
                ndays=126,
                returns=returns,
                weights=garch.weights,
                alpha=0.99,
            )
        result = garch.CustomNDayPriceSimulation(
            horizon1=horizon1,
            horizon2=horizon2,
            ndays=nday,
            nsims=nsims,
            shock_type="custom_bootstrap",
            bootstrap_start=stress_periods[0],
            bootstrap_end=stress_periods[1],
            start_vals=start_vals,
        )
        result["Method"] = f"stress_var_{round(start_point_quantile * 100)}"
        records.extend(
            legacy_garch_dict_to_records(
                result,
                volatility_factor_id=factor.volatility_factor_id,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                run_id=run_id,
                as_of_date=as_of_date,
            )
        )

    return records, []


def _run_additional_method(
    method_name: str,
    factor: FactorConfig,
    pricing_data: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    nday: int,
):
    kwargs = _base_model_kwargs(factor, pricing_data, sample_start, sample_end)
    if method_name == "historical_simulation":
        return HistoricalSimulationVaR(**kwargs).run(n_liquidation=nday)
    if method_name == "parametric_variance_covariance":
        return ParametricVaR(**kwargs).run(n_liquidation=nday)
    if method_name == "ewma":
        return EWMAVaR(**kwargs).run(n_liquidation=nday)
    if method_name == "cornish_fisher":
        return CornishFisherVaR(**kwargs).run(n_liquidation=nday)
    if method_name == "filtered_historical_simulation":
        return FilteredHistoricalSimulationVaR(**kwargs).run(n_liquidation=nday)
    if method_name == "evt_peaks_over_threshold":
        return EVTPeaksOverThresholdVaR(**kwargs).run(n_liquidation=nday)
    if method_name == "circular_block_bootstrap":
        return CircularBlockBootstrapVaR(**kwargs).run(n_liquidation=nday, nboot=10000)
    raise ValueError(f"Unknown additional VaR method: {method_name}")


def run_configured_models(
    config: dict[str, Any],
    pricing_source: PricingDataSource,
    additional_methods: Sequence[str] = ADDITIONAL_METHODS,
    run_id: str | None = None,
    as_of_date: str | None = None,
    raise_on_error: bool = True,
) -> RunOutputs:
    """Run configured legacy GARCH variants and the additional VaR methods."""

    run_id = run_id or str(uuid4())
    run_timestamp = datetime.now(timezone.utc)
    as_of_date = as_of_date or _config_value(config, "garch_settings", "last_sample_date")
    sample_range, forecast_range = create_periods(
        last_sample_date=_config_value(config, "garch_settings", "last_sample_date"),
        periods=int(_config_value(config, "garch_settings", "number_of_periods")),
        forecast_mos=int(_config_value(config, "garch_settings", "forecast_months")),
        forecast_lag=int(_config_value(config, "garch_settings", "forecast_lag")),
        sample_yrs=int(_config_value(config, "garch_settings", "sample_years")),
    )
    start_point_quantiles = _config_value(config, "garch_settings", "start_point_quantiles", [0.50])

    result_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    pricing_qa: list[dict[str, Any]] = []

    for factor in iter_factor_configs(config):
        pricing_data = pricing_source.load_pricing_data(tickers=factor.tickers)
        for nday in factor.ndays:
            for idx, forecast_period in enumerate(forecast_range):
                sample_start, sample_end = sample_range[idx]
                forecast_start, forecast_end = forecast_period
                coverage_model = GarchSimulation(
                    **_base_model_kwargs(factor, pricing_data, sample_start, sample_end)
                )
                qa_failure = _validate_sample_coverage(
                    coverage_model,
                    factor,
                    sample_start,
                    sample_end,
                )
                if qa_failure is not None:
                    pricing_qa.append(qa_failure)
                    continue

                for start_point_quantile in start_point_quantiles:
                    garch_records, qa_records = _run_garch_variants(
                        factor=factor,
                        pricing_data=pricing_data,
                        sample_start=sample_start,
                        sample_end=sample_end,
                        forecast_start=forecast_start,
                        forecast_end=forecast_end,
                        nday=nday,
                        config=config,
                        start_point_quantile=float(start_point_quantile),
                        run_id=run_id,
                        as_of_date=as_of_date,
                    )
                    result_records.extend(garch_records)
                    pricing_qa.extend(qa_records)

                for method_name in additional_methods:
                    try:
                        method_result = _run_additional_method(
                            method_name=method_name,
                            factor=factor,
                            pricing_data=pricing_data,
                            sample_start=sample_start,
                            sample_end=sample_end,
                            nday=nday,
                        )
                        records = method_result.to_records()
                        result_records.extend(
                            _stamp_records(records, run_id, as_of_date, forecast_start, forecast_end)
                        )
                    except Exception as exc:
                        if raise_on_error:
                            raise
                        diagnostics.append(
                            {
                                "run_id": run_id,
                                "run_timestamp": run_timestamp,
                                "as_of_date": as_of_date,
                                "volatility_factor_id": factor.volatility_factor_id,
                                "method": method_name,
                                "sample_start": sample_start,
                                "sample_end": sample_end,
                                "liquidation_days": nday,
                                "status": "failed",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )

    model_runs = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "as_of_date": as_of_date,
                "status": "completed",
                "configured_factors": len(list(iter_factor_configs(config))),
                "additional_methods": ",".join(additional_methods),
            }
        ]
    )
    return RunOutputs(
        run_id=run_id,
        results=pd.DataFrame.from_records(result_records),
        diagnostics=pd.DataFrame.from_records(diagnostics),
        pricing_qa=pd.DataFrame.from_records(pricing_qa),
        model_runs=model_runs,
    )
