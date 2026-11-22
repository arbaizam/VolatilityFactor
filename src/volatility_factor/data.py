"""Pricing data and the :class:`VarData` base class.

Pricing data is never read at import time. Every model takes an optional ``data=``
argument (a :class:`PricingData`, a path to a CSV/parquet file, or a DataFrame). When it
is omitted, the active data is resolved lazily, in order:

1. :func:`set_data` / :func:`use_data`
2. the ``VOLFACTOR_DATA`` environment variable
3. ``data_settings.pricing_data`` in the active config file

Files loaded by path are cached, so many models share a single read.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

from volatility_factor._utils import clear_cached_properties
from volatility_factor.config import Config, get_config

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from volatility_factor.backtesting import VarBacktest
    from volatility_factor.garch import GarchSim
    from volatility_factor.nonparametric import NonParametricVar

DATA_ENV_VAR = "VOLFACTOR_DATA"
PRICE_COLUMNS = ("Date", "Ticker", "Source", "Description", "Price")

DataLike = Union["PricingData", str, os.PathLike, pd.DataFrame]  # noqa: UP007 (forward ref)
ConfigLike = Config | str | os.PathLike | dict

__all__ = [
    "PRICE_COLUMNS",
    "PricingData",
    "VarData",
    "get_data",
    "set_data",
    "use_data",
]


class PricingData:
    """Long-format index prices: ``Date, Ticker, Source, Description, Price``.

    Build one with :meth:`from_csv`, :meth:`from_parquet`, :meth:`from_frame` or
    :meth:`from_delta` (Databricks). Rows are sorted by ``Ticker, Date``.
    """

    def __init__(self, frame: pd.DataFrame, *, source: str | None = None):
        missing = [c for c in PRICE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Pricing data is missing required columns: {missing}")
        self._frame = frame
        self.source = source

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_csv(cls, path: str | os.PathLike, **read_csv_kwargs: Any) -> PricingData:
        """Read the standard pricing CSV (prices may contain thousands separators)."""
        kwargs = {
            "parse_dates": ["Date"],
            "dtype": {"Price": np.float64},
            "thousands": ",",
            **read_csv_kwargs,
        }
        frame = pd.read_csv(path, **kwargs).sort_values(by=["Ticker", "Date"])
        return cls(frame, source=str(path))

    @classmethod
    def from_parquet(cls, path: str | os.PathLike) -> PricingData:
        return cls.from_frame(pd.read_parquet(path), source=str(path))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, source: str | None = "DataFrame") -> PricingData:
        """Normalise an in-memory DataFrame (dates parsed, prices as float64)."""
        df = frame.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        if df["Price"].dtype == object:
            df["Price"] = df["Price"].astype(str).str.replace(",", "", regex=False)
        df["Price"] = df["Price"].astype(np.float64)
        return cls(df.sort_values(by=["Ticker", "Date"]), source=source)

    @classmethod
    def from_delta(cls, table_name: str, spark: Any = None) -> PricingData:
        """Read a Delta / Unity Catalog table through Spark (Databricks)."""
        if spark is None:
            try:
                from pyspark.sql import SparkSession
            except ImportError as exc:  # pragma: no cover - depends on runtime
                raise RuntimeError("pyspark is required for PricingData.from_delta") from exc
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        frame = spark.table(table_name).select(*PRICE_COLUMNS).toPandas()
        return cls.from_frame(frame, source=table_name)

    @classmethod
    def load(cls, path: str | os.PathLike) -> PricingData:
        """Load by file extension (``.csv`` or ``.parquet``)."""
        suffix = Path(path).suffix.lower()
        if suffix == ".parquet":
            return cls.from_parquet(path)
        if suffix in {".csv", ".txt"}:
            return cls.from_csv(path)
        raise ValueError(f"Unsupported pricing data file type: {path}")

    # ------------------------------------------------------------------ access
    @property
    def frame(self) -> pd.DataFrame:
        """The underlying DataFrame (treat as read-only)."""
        return self._frame

    @cached_property
    def tickers(self) -> list[str]:
        return sorted(self._frame["Ticker"].unique())

    def rows(self, tickers: Sequence[str]) -> pd.DataFrame:
        """Rows for ``tickers`` (a copy with a fresh RangeIndex)."""
        missing = [t for t in tickers if t not in set(self.tickers)]
        if missing:
            raise KeyError(f"Tickers not found in pricing data ({self.source}): {missing}")
        return self._frame[self._frame["Ticker"].isin(tickers)].copy().reset_index(drop=True)

    def filter(
        self,
        tickers: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> PricingData:
        """Return a new :class:`PricingData` restricted to tickers and/or dates."""
        df = self._frame
        if tickers is not None:
            df = df[df["Ticker"].isin(list(tickers))]
        if start is not None:
            df = df[df["Date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["Date"] <= pd.Timestamp(end)]
        return PricingData(df, source=self.source)

    def date_range(self, ticker: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
        df = self._frame if ticker is None else self._frame[self._frame["Ticker"] == ticker]
        return df["Date"].min(), df["Date"].max()

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return f"PricingData(rows={len(self)}, tickers={len(self.tickers)}, source={self.source!r})"


# --------------------------------------------------------------------------- #
# Active (default) pricing data
# --------------------------------------------------------------------------- #
_active_data: PricingData | None = None
_path_cache: dict[Path, PricingData] = {}


def _load_cached(path: str | os.PathLike) -> PricingData:
    key = Path(path).resolve()
    if key not in _path_cache:
        _path_cache[key] = PricingData.load(key)
    return _path_cache[key]


def _coerce_data(data: DataLike) -> PricingData:
    if isinstance(data, PricingData):
        return data
    if isinstance(data, pd.DataFrame):
        return PricingData.from_frame(data)
    return _load_cached(data)


def get_data(data: DataLike | None = None, config: ConfigLike | None = None) -> PricingData:
    """Resolve pricing data (see the module docstring for the lookup order)."""
    if data is not None:
        return _coerce_data(data)
    if _active_data is not None:
        return _active_data
    env = os.environ.get(DATA_ENV_VAR)
    if env:
        return _load_cached(env)
    try:
        cfg = get_config(config)
    except FileNotFoundError:
        cfg = None
    if cfg is not None and cfg.pricing_data_path is not None:
        return _load_cached(cfg.pricing_data_path)
    raise FileNotFoundError(
        "No pricing data available. Pass data=..., call set_data(...), set "
        f"${DATA_ENV_VAR}, or add data_settings.pricing_data to the config file."
    )


def set_data(data: DataLike | None) -> PricingData | None:
    """Set the active default pricing data (``None`` resets to automatic lookup)."""
    global _active_data
    _active_data = None if data is None else _coerce_data(data)
    return _active_data


@contextmanager
def use_data(data: DataLike) -> Iterator[PricingData]:
    """Temporarily make ``data`` the active default pricing data."""
    global _active_data
    previous = _active_data
    _active_data = _coerce_data(data)
    try:
        yield _active_data
    finally:
        _active_data = previous


# --------------------------------------------------------------------------- #
# VarData
# --------------------------------------------------------------------------- #
class VarData:
    """Historical prices and returns for one index or a weighted portfolio of indices.

    Args:
        tickers: a ticker string, or a list/tuple of tickers.
        weights: portfolio weights (must sum to one). ``None`` means equal weights. Ignored
            for a single ticker, whose weight is always 1.
        data: pricing data to use (see module docstring). Resolved once, at construction.
        config: config used to look up ``volatility_factor_id`` and default data.
        volatility_factor_id: the factor id, if known. Otherwise it is looked up from
            ``tickers`` in the config.

    Build from a configured factor with :meth:`from_id`, then chain into a model::

        var = (VarData.from_id("agy_cmo")
                 .garch("2016-01-01", "2024-12-31")
                 .simulate(horizon1=63, horizon2=261, ndays=10, nsims=10_000,
                           shock_type="custom_normal", start_vals=("2024-01-01", 0.5)))
    """

    def __init__(
        self,
        tickers: str | Sequence[str],
        weights: Sequence[float] | float | None = None,
        *,
        data: DataLike | None = None,
        config: ConfigLike | None = None,
        volatility_factor_id: str | None = None,
    ):
        self._config_arg = config
        self._volatility_factor_id = volatility_factor_id
        self.data = get_data(data, config)
        self.tickers = tickers
        self.weights = weights

    @classmethod
    def from_id(
        cls,
        volatility_factor_id: str,
        *,
        data: DataLike | None = None,
        config: ConfigLike | None = None,
        **kwargs: Any,
    ):
        """Build from a ``var_settings`` entry (tickers and weights come from the config).

        Extra keyword arguments go to the class constructor, e.g.
        ``GarchSim.from_id("agy_cmo", sample_start=..., sample_end=...)``.
        """
        cfg = get_config(config)
        factor = cfg.factor(volatility_factor_id)
        return cls(
            tickers=factor.ticker_list,
            weights=factor.weight_list,
            data=data,
            config=cfg,
            volatility_factor_id=factor.volatility_factor_id,
            **kwargs,
        )

    # ------------------------------------------------------------ tickers/weights
    @property
    def tickers(self) -> list[str]:
        return self._tickers

    @tickers.setter
    def tickers(self, value: str | Sequence[str]) -> None:
        if isinstance(value, tuple):
            tickers = list(value)
        elif isinstance(value, str):
            tickers = [value]
        elif isinstance(value, list):
            tickers = value
        else:
            raise ValueError(
                f"{value} is of object type {type(value)}.  Valid types include: str, tuple, or list"
            )
        self._tickers = tickers
        self._weights = None  # invalidate; re-set via the weights setter
        clear_cached_properties(self)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, value: Sequence[float] | float | None) -> None:
        n = len(self.tickers)
        if n == 1:
            weights = np.array(1)
        elif value is None:
            weights = np.array([1 / n] * n)
        else:
            weights = np.array(value)
            if weights.shape != (n,):
                raise ValueError(f"Expected {n} weights for {self.tickers}; got {value}")
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError(f"Weights must add to one.  Weights totaled {sum(weights)}")
        self._weights = weights
        clear_cached_properties(self)

    # ------------------------------------------------------------------ chaining
    def _model_kwargs(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "config": self._config_arg,
            "volatility_factor_id": self._volatility_factor_id,
        }

    def _share_cache_with(self, other: VarData) -> VarData:
        """Copy already-computed price/return matrices to ``other`` (same portfolio)."""
        for name, attr in vars(VarData).items():
            if isinstance(attr, cached_property) and name in self.__dict__:
                other.__dict__[name] = self.__dict__[name]
        return other

    def garch(
        self,
        sample_start: str,
        sample_end: str,
        mean_model: str = "Zero",
        vol_model: str = "Garch",
    ) -> GarchSim:
        """A :class:`~volatility_factor.garch.GarchSim` for this portfolio."""
        from volatility_factor.garch import GarchSim

        model = GarchSim(self.tickers, self.weights.tolist(), sample_start, sample_end,
                         mean_model=mean_model, vol_model=vol_model, **self._model_kwargs())
        return self._share_cache_with(model)

    def nonparametric(self, sample_start: str, sample_end: str) -> NonParametricVar:
        """A :class:`~volatility_factor.nonparametric.NonParametricVar` for this portfolio."""
        from volatility_factor.nonparametric import NonParametricVar

        model = NonParametricVar(self.tickers, self.weights.tolist(), sample_start, sample_end,
                                 **self._model_kwargs())
        return self._share_cache_with(model)

    def backtest(
        self,
        n_liquidation: int,
        var: float | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        var_is_log: bool = True,
        forward_shift: bool = False,
    ) -> VarBacktest:
        """A :class:`~volatility_factor.backtesting.VarBacktest` for this portfolio."""
        from volatility_factor.backtesting import VarBacktest

        model = VarBacktest(self.tickers, n_liquidation, weights=self.weights.tolist(),
                            start_date=start_date, end_date=end_date, var=var,
                            var_is_log=var_is_log, forward_shift=forward_shift,
                            **self._model_kwargs())
        return self._share_cache_with(model)

    # --------------------------------------------------------- price matrices
    @cached_property
    def single_index(self) -> bool:
        return len(self.tickers) == 1

    @cached_property
    def price_matrix(self) -> pd.DataFrame:
        df = self.data.rows(self.tickers)
        return pd.DataFrame(
            df.pivot(index="Date", values="Price", columns="Ticker").rename_axis(None, axis=1)
        )[self.tickers]

    @cached_property
    def price_return_matrix(self) -> pd.DataFrame:
        """One-day percentage change."""
        return (self.price_matrix.diff() / self.price_matrix.shift(1)).dropna()

    @cached_property
    def wtd_price_return(self) -> pd.DataFrame:
        """Weighted one-day percentage change."""
        return pd.DataFrame(
            data=np.dot(self.price_return_matrix, self.weights),
            columns=["Wtd_Price_Return"],
            index=self.price_return_matrix.index,
        )

    @cached_property
    def wtd_price(self) -> pd.DataFrame:
        """Weighted price."""
        return pd.DataFrame(
            data=np.dot(self.price_matrix, self.weights),
            columns=["Wtd_Price"],
            index=self.price_matrix.index,
        )

    @cached_property
    def log_price_matrix(self) -> pd.DataFrame:
        return np.log(self.price_matrix)

    @cached_property
    def log_return_matrix(self) -> pd.DataFrame:
        """One-day log return."""
        return self.log_price_matrix.diff().dropna()

    @cached_property
    def wtd_log_return(self) -> pd.DataFrame:
        """Weighted one-day log return."""
        return pd.DataFrame(
            data=np.dot(self.log_return_matrix, self.weights),
            columns=["Wtd_Log_Return"],
            index=self.log_return_matrix.index,
        )

    # ------------------------------------------------------------ descriptions
    @cached_property
    def index_description(self) -> str:
        """Semicolon-delimited vendor description of each index (used in chart titles)."""
        df = self.data.rows(self.tickers).groupby("Ticker")["Description"].first()
        vals = [f"{df.iloc[i]} ({df.index[i]})" for i in range(df.shape[0])]
        # Same string formatting as the original implementation.
        return str(vals).replace("['", "").replace("']", "").replace(",", ";")

    @cached_property
    def description(self) -> str:
        """Readable ``"Description (TICKER); ..."`` in ticker order (used in chart titles)."""
        df = self.data.rows(self.tickers).groupby("Ticker")["Description"].first()
        return "; ".join(f"{df.loc[t]} ({t})" for t in self.tickers)

    @cached_property
    def ticker_description(self) -> str:
        """Semicolon-delimited tickers."""
        return ";".join(str(t) for t in self.tickers)

    @cached_property
    def weight_description(self) -> str:
        """Weights as written to results: ``"1"`` or ``"0.5;0.5"``."""
        if self.single_index:
            return str(self.weights)
        return ";".join(str(x) for x in self.weights)

    @cached_property
    def volatility_factor_id(self) -> str:
        """The factor id: as given, else looked up from tickers in the config."""
        if self._volatility_factor_id is not None:
            return self._volatility_factor_id
        try:
            found = get_config(self._config_arg).find_factor_id(self.tickers)
        except FileNotFoundError:
            found = None
        if found is None:
            warnings.warn(f"Unknown Volatility Factor ID for {self.tickers}", stacklevel=2)
            return "Unknown"
        return found

    # ------------------------------------------------------------------ analysis
    def rolling_volatility(
        self, window: int = 30, periods_per_year: int = 252
    ) -> pd.Series:
        """Rolling annualised volatility of the weighted log return."""
        returns = self.wtd_log_return["Wtd_Log_Return"]
        vol = returns.rolling(window=window).std() * np.sqrt(periods_per_year)
        return vol.dropna().rename("AnnualizedVol")

    def plot_rolling_volatility(self, **kwargs: Any) -> go.Figure:
        """See :func:`volatility_factor.plotting.rolling_volatility_plot`."""
        from volatility_factor.plotting import rolling_volatility_plot

        return rolling_volatility_plot(self, **kwargs)

    def plot_price(self, **kwargs: Any) -> go.Figure:
        """See :func:`volatility_factor.plotting.price_plot`."""
        from volatility_factor.plotting import price_plot

        return price_plot(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tickers={self.tickers}, weights={self.weights.tolist()})"
        )
