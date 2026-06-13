"""Shared result contracts for model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Mapping

import numpy as np

CONFIDENCE_LEVELS = (0.50, 0.75, 0.95, 0.99)


def confidence_label(confidence_level: float) -> str:
    return f"{round(confidence_level * 100)}Q"


def arithmetic_var_from_log_quantile(log_quantile: float) -> float:
    """Convert a left-tail log-return quantile to positive arithmetic VaR percent."""

    return float(-(np.exp(log_quantile) - 1.0) * 100.0)


def log_var_from_log_quantile(log_quantile: float) -> float:
    """Convert a left-tail log-return quantile to positive log VaR percent."""

    return float(-log_quantile * 100.0)


@dataclass(frozen=True)
class MethodResult:
    """Long-form serializable result from one VaR method run."""

    method: str
    volatility_factor_id: str
    tickers: str
    weights: str
    sample_start: str
    sample_end: str
    liquidation_days: int
    var_by_confidence: Mapping[float, float]
    log_var_by_confidence: Mapping[float, float] | None = None
    method_variant: str | None = None
    forecast_start: str | None = None
    forecast_end: str | None = None
    run_id: str | None = None
    as_of_date: str | None = None
    run_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parameters: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        log_vars = self.log_var_by_confidence or {}
        for confidence_level, var_value in sorted(self.var_by_confidence.items()):
            records.append(
                {
                    "run_id": self.run_id,
                    "run_timestamp": self.run_timestamp,
                    "as_of_date": self.as_of_date,
                    "volatility_factor_id": self.volatility_factor_id,
                    "tickers": self.tickers,
                    "weights": self.weights,
                    "method": self.method,
                    "method_variant": self.method_variant,
                    "sample_start": self.sample_start,
                    "sample_end": self.sample_end,
                    "forecast_start": self.forecast_start,
                    "forecast_end": self.forecast_end,
                    "liquidation_days": self.liquidation_days,
                    "confidence_level": confidence_level,
                    "confidence_label": confidence_label(confidence_level),
                    "var_value": float(var_value),
                    "log_var_value": (
                        float(log_vars[confidence_level])
                        if confidence_level in log_vars
                        else None
                    ),
                    "parameters_json": json.dumps(self.parameters, default=str, sort_keys=True),
                    "diagnostics_json": json.dumps(self.diagnostics, default=str, sort_keys=True),
                }
            )
        return records


def legacy_garch_dict_to_records(
    result: Mapping[str, Any],
    volatility_factor_id: str,
    forecast_start: str | None = None,
    forecast_end: str | None = None,
    run_id: str | None = None,
    as_of_date: str | None = None,
) -> list[dict[str, Any]]:
    """Convert the legacy GARCH wide dictionary to the shared long-form schema."""

    var_by_confidence = {
        level: float(result[f"VaR_{round(level * 100)}"]) for level in CONFIDENCE_LEVELS
    }
    log_var_by_confidence = {
        level: float(result[f"LogVaR_{round(level * 100)}"]) for level in CONFIDENCE_LEVELS
    }
    method_result = MethodResult(
        run_id=run_id,
        as_of_date=as_of_date,
        volatility_factor_id=volatility_factor_id,
        tickers=str(result["Tickers"]),
        weights=str(result["Weights"]),
        method=str(result["Method"]),
        method_variant=str(result["ShockType"]),
        sample_start=str(result["SampleStart"]),
        sample_end=str(result["SampleEnd"]),
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        liquidation_days=int(result["NDays"]),
        var_by_confidence=var_by_confidence,
        log_var_by_confidence=log_var_by_confidence,
        parameters={
            "VaRp": result.get("VaRp"),
            "Horizon1": result.get("Horizon1"),
            "Horizon2": result.get("Horizon2"),
            "Simulations": result.get("Simulations"),
            "MeanModel": result.get("MeanModel"),
            "VolatilityModel": result.get("VolatilityModel"),
            "BootstrapStart": result.get("BootstrapStart"),
            "BootstrapEnd": result.get("BootstrapEnd"),
            "StartingValueQ": result.get("StartingValueQ"),
            "StartingValueDate": result.get("StartingValueDate"),
        },
        diagnostics={"ConvergencyCode": result.get("ConvergencyCode")},
    )
    return method_result.to_records()

