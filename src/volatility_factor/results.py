"""Result containers returned by the VaR models."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go

CONFIDENCE_LEVELS = (0.50, 0.75, 0.95, 0.99)
VAR_COLUMNS = ["VaR_50", "VaR_75", "VaR_95", "VaR_99"]
LOG_VAR_COLUMNS = ["LogVaR_50", "LogVaR_75", "LogVaR_95", "LogVaR_99"]
DATE_COLUMNS = [
    "SampleStart",
    "SampleEnd",
    "ForecastStart",
    "ForecastEnd",
    "BootstrapStart",
    "BootstrapEnd",
    "StartingValueDate",
]


@dataclass(frozen=True)
class VaRResult(Mapping):
    """Output of :meth:`GarchSim.simulate`.

    Behaves like a read-only dict of the results (``result["VaR_99"]``, ``dict(result)``)
    and also keeps the simulation arrays.

    Attributes:
        record: the result dict (settings first, then the VaR columns).
        sim_paths: ``(nsims, horizon)`` simulated one-day log returns (portfolio level).
        nday_returns: rolling ``ndays`` sums of the forecast part of each path
            (rows = rolling windows, columns = simulations).
        var_by_path: VaR quantile of each path, in log-percent.
    """

    record: dict[str, Any]
    sim_paths: np.ndarray = field(repr=False)
    nday_returns: pd.DataFrame = field(repr=False)
    var_by_path: pd.Series = field(repr=False)

    # Mapping interface ---------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        """Value from the result dict."""
        return self.record[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the result dict's keys."""
        return iter(self.record)

    def __len__(self) -> int:
        """Number of keys in the result dict."""
        return len(self.record)

    # Conversions ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The result dict, one key per output column (a copy)."""
        return dict(self.record)

    def to_frame(self) -> pd.DataFrame:
        """One-row DataFrame of :meth:`to_dict`."""
        return pd.DataFrame([self.record])

    def with_metadata(self, **fields: Any) -> VaRResult:
        """Return a copy with ``fields`` merged into the record (``dict.update`` order)."""
        record = dict(self.record)
        record.update(fields)
        return replace(self, record=record)

    def var(self, level: float = 0.99) -> float:
        """Arithmetic VaR (percent) at ``level`` (one of 0.50, 0.75, 0.95, 0.99)."""
        return self.record[f"VaR_{round(level * 100)}"]

    def log_var(self, level: float = 0.99) -> float:
        """Log VaR (percent) at ``level``."""
        return self.record[f"LogVaR_{round(level * 100)}"]

    def plot_distribution(self, **kwargs: Any) -> go.Figure:
        """See :func:`volatility_factor.plotting.var_distribution_plot`."""
        from volatility_factor.plotting import var_distribution_plot

        return var_distribution_plot(self, **kwargs)

    def __repr__(self) -> str:
        """Tickers, method, liquidation days and the 50th and 99th percentile VaR."""
        keys = ("Tickers", "Method", "NDays", "VaR_50", "VaR_99")
        body = ", ".join(f"{k}={self.record.get(k)!r}" for k in keys)
        return f"VaRResult({body})"


def to_long(results: pd.DataFrame | list[Mapping[str, Any]]) -> pd.DataFrame:
    """Melt wide GARCH results to one row per VaR quantile (the saved ``.parquet`` layout).

    ``LogVaR_99`` is kept as an id column; the other ``LogVaR_*`` columns are dropped.
    """
    df = pd.DataFrame.from_records(results) if not isinstance(results, pd.DataFrame) else results
    if df.columns.empty:
        return pd.DataFrame(columns=["VaRQ", "VaR"])
    df = df.copy()
    excluded = [*LOG_VAR_COLUMNS[:-1], *VAR_COLUMNS]
    id_cols = [c for c in df.columns if c not in excluded]
    date_cols = [c for c in DATE_COLUMNS if c in df.columns]
    df[date_cols] = df[date_cols].apply(pd.to_datetime)
    df = df.melt(id_vars=id_cols, value_vars=VAR_COLUMNS, var_name="VaRQ", value_name="VaR")
    df["VaRQ"] = df["VaRQ"].str[-2:] + "Q"
    return df
