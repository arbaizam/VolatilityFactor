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


class PricingData:
    """Long-format index prices: ``Date, Ticker, Source, Description, Price``.

    Build one with :meth:`from_csv`, :meth:`from_parquet` or :meth:`from_frame`.
    Rows are sorted by ``Ticker, Date``.
    """

    def __init__(self, frame: pd.DataFrame, *, source: str | None = None):
        """Wrap an already-normalised price DataFrame.

        Use the ``from_*`` constructors for files and raw frames.

        Args:
            frame: rows with ``Date, Ticker, Source, Description, Price``.
            source: where the data came from (shown in messages).

        Raises:
            ValueError: if a required column is missing.
        """
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
        return cls.from_frame(pd.read_csv(path, **kwargs), source=str(path))

    @classmethod
    def from_parquet(cls, path: str | os.PathLike) -> PricingData:
        """Read a parquet file with the standard pricing columns."""
        return cls.from_frame(pd.read_parquet(path), source=str(path))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, source: str | None = "DataFrame") -> PricingData:
        """Normalise an in-memory DataFrame (dates parsed, prices as float64)."""
        df = frame.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df["Price"].dtype == object:
            df["Price"] = df["Price"].astype("string").str.replace(",", "", regex=False)
        df["Price"] = df["Price"].astype(np.float64)
        return cls(df.sort_values(by=["Ticker", "Date"]), source=source)

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
        """Sorted list of tickers in the data."""
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
        """First and last date, for one ticker or for the whole data set.

        Args:
            ticker: ticker to restrict to; ``None`` for all rows.

        Returns:
            ``(first_date, last_date)``.
        """
        df = self._frame if ticker is None else self._frame[self._frame["Ticker"] == ticker]
        return df["Date"].min(), df["Date"].max()

    # --------------------------------------------------------- observation QA
    def _window(
        self, start: str | None, end: str | None, priced_only: bool = False
    ) -> pd.DataFrame:
        """Rows between ``start`` and ``end`` (inclusive; ``None`` means open-ended).

        Args:
            start, end: date bounds (``YYYY-MM-DD``).
            priced_only: keep only dated observations with finite, positive prices.
                Otherwise, undated rows are kept for reporting.
        """
        df = self._frame
        if priced_only:
            df = df[_valid_observations(df)]
        if start is not None:
            df = df[df["Date"].isna() | df["Date"].ge(pd.Timestamp(start))]
        if end is not None:
            df = df[df["Date"].isna() | df["Date"].le(pd.Timestamp(end))]
        return df

    @staticmethod
    def _vendor_calendars(
        window: pd.DataFrame, min_vendor_share: float = 0.5
    ) -> dict[str, pd.DatetimeIndex]:
        """Expected pricing dates per vendor (``Source``) within ``window``.

        A date is expected for a vendor when at least ``min_vendor_share`` of that vendor's
        indices that are live on the date (between their first and last price) are priced on
        it. This ignores dates only a few indices are priced on, such as weekend month-end
        marks.
        """
        calendars = {}
        for source, group in window.groupby("Source"):
            priced = group.drop_duplicates(["Ticker", "Date"]).groupby("Date")["Ticker"].size()
            spans = group.groupby("Ticker")["Date"].agg(["min", "max"])
            dates = priced.index.values
            started = np.searchsorted(np.sort(spans["min"].values), dates, side="right")
            ended = np.searchsorted(np.sort(spans["max"].values), dates, side="left")
            live = np.maximum(started - ended, 1)
            keep = priced.values >= min_vendor_share * live
            calendars[source] = pd.DatetimeIndex(dates[keep])
        return calendars

    def _resolve_tickers(self, tickers: str | Sequence[str] | None) -> list[str]:
        """Normalise a ticker argument and check every ticker exists."""
        if tickers is None:
            return list(self.tickers)
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        missing = [t for t in tickers if t not in set(self.tickers)]
        if missing:
            raise KeyError(f"Tickers not found in pricing data ({self.source}): {missing}")
        return tickers

    def coverage_report(
        self,
        tickers: str | Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        max_gap_days: int = 5,
        stale_days: int = 5,
        min_vendor_share: float = 0.5,
    ) -> pd.DataFrame:
        """Summarise price observations by index to spot incomplete pricing.

        An index's *expected* dates are its vendor's (``Source``) pricing dates between the
        index's own first and last observation. A vendor pricing date is one on which at
        least ``min_vendor_share`` of that vendor's live indices are priced, so vendor
        holidays and sporadic dates (e.g. weekend month-end marks) are not counted as missing.

        Args:
            tickers: tickers to report on; ``None`` for every ticker in the data.
            start, end: restrict the check to this date window (``YYYY-MM-DD``).
            max_gap_days: calendar days between consecutive observations above which a gap
                is flagged.
            stale_days: number of consecutive observations with an unchanged price at or
                above which the run is flagged as stale.
            min_vendor_share: share of a vendor's live indices that must be priced on a
                date for it to count as an expected date.

        Returns:
            One row per ticker with ``Ticker, Description, Source, FirstDate, LastDate,
            Observations, ExpectedObservations, MissingObservations, CoveragePct,
            LargestGapDays, LargestGapStart, LargestGapEnd, GapsOverThreshold,
            LongestStaleRun, DuplicateDates, InvalidDates, MissingPrices, NonFinitePrices,
            NonPositivePrices, MissingSources,
            WeekendObservations, StartsLate, EndsEarly, Status, Issues``.

            ``Status`` is ``"Review"`` when any issue is found: missing vendor dates, gaps
            longer than ``max_gap_days`` that span vendor pricing dates, stale runs,
            duplicate or invalid dates, missing sources, missing or invalid prices, or prices
            ending before the vendor's last pricing date. ``LargestGapDays``, ``WeekendObservations`` and
            ``StartsLate`` (usually the index's inception) are informational.

        Example::

            vf.get_data().coverage_report(start="2016-01-01").query("Status == 'Review'")
        """
        window = self._window(start, end)
        calendars = self._vendor_calendars(window[_valid_observations(window)], min_vendor_share)
        rows = []
        for ticker in self._resolve_tickers(tickers):
            group = window[window["Ticker"] == ticker].sort_values("Date")
            rows.append(_coverage_row(ticker, group, calendars, max_gap_days, stale_days))
        return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)

    def gaps(
        self,
        ticker: str,
        min_days: int = 5,
        start: str | None = None,
        end: str | None = None,
        min_vendor_share: float = 0.5,
    ) -> pd.DataFrame:
        """Every gap between consecutive observations of one index longer than ``min_days``.

        Args:
            ticker: the index to inspect.
            min_days: minimum gap, in calendar days, to report.
            start, end: restrict to this date window (``YYYY-MM-DD``).
            min_vendor_share: see :meth:`coverage_report`.

        Returns:
            One row per gap: ``LastPriceDate`` (before the gap), ``NextPriceDate``,
            ``CalendarDays`` and ``MissingVendorDates`` (dates the vendor priced other indices
            but not this one).
        """
        self._resolve_tickers(ticker)
        window = self._window(start, end, priced_only=True)
        group = window[window["Ticker"] == ticker]
        dates = pd.DatetimeIndex(np.sort(group["Date"].unique()))
        columns = ["LastPriceDate", "NextPriceDate", "CalendarDays", "MissingVendorDates"]
        if len(dates) < 2:
            return pd.DataFrame(columns=columns)
        calendar = self._vendor_calendars(window, min_vendor_share).get(
            group["Source"].iloc[0], dates
        )
        days = np.diff(dates.values).astype("timedelta64[D]").astype(int)
        rows = []
        for i in np.flatnonzero(days > min_days):
            before, after = dates[i], dates[i + 1]
            missing = int(((calendar > before) & (calendar < after)).sum())
            rows.append([before, after, int(days[i]), missing])
        return pd.DataFrame(rows, columns=columns)

    def observation_counts(
        self,
        freq: str = "ME",
        tickers: str | Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Priced observations per index per period (rows = period end, columns = tickers).

        Args:
            freq: pandas period frequency, e.g. ``"ME"`` (month), ``"QE"`` (quarter) or
                ``"YE"`` (year).
            tickers: tickers to include; ``None`` for all.
            start, end: restrict to this date window (``YYYY-MM-DD``).
            normalize: return coverage (observations / vendor dates in the period) instead
                of counts. Periods without a vendor calendar and periods before an
                index's first observation are ``NaN``.

        Returns:
            DataFrame of counts (or coverage fractions).
        """
        tickers = self._resolve_tickers(tickers)
        window = self._window(start, end, priced_only=True)
        subset = window[window["Ticker"].isin(tickers)].drop_duplicates(["Ticker", "Date"])
        periods = pd.Series(1, index=pd.DatetimeIndex(window["Date"].unique())).resample(freq).sum()
        counts = (
            subset.groupby(["Ticker", pd.Grouper(key="Date", freq=freq)]).size()
            .unstack("Ticker").reindex(index=periods.index, columns=tickers).fillna(0).astype(int)
        )
        counts.index.name = "Date"
        if not normalize:
            return counts

        calendars = self._vendor_calendars(window)
        source_of = subset.groupby("Ticker")["Source"].first()
        first_of = subset.groupby("Ticker")["Date"].min()
        coverage = pd.DataFrame(np.nan, index=counts.index, columns=tickers)
        for ticker in tickers:
            calendar = calendars.get(source_of.get(ticker))
            if calendar is None:
                continue
            # Start at each index's first valid observation.
            calendar = calendar[calendar >= first_of[ticker]]
            expected = pd.Series(1, index=calendar).resample(freq).sum().reindex(counts.index)
            # Count only vendor pricing dates, so off-calendar prints can't push coverage
            # above 100%.
            own = subset.loc[subset["Ticker"] == ticker, "Date"]
            on_calendar = own[own.isin(calendar)]
            observed = pd.Series(1, index=pd.DatetimeIndex(on_calendar)).resample(freq).sum()
            share = observed.reindex(counts.index).fillna(0) / expected
            # Period labels are period ends, so the period holding the first price qualifies.
            coverage[ticker] = share.where(counts.index >= first_of[ticker])
        return coverage

    def plot_observation_counts(self, **kwargs: Any) -> go.Figure:
        """See :func:`volatility_factor.plotting.observation_heatmap`."""
        from volatility_factor.plotting import observation_heatmap

        return observation_heatmap(self, **kwargs)

    def __len__(self) -> int:
        """Number of price rows."""
        return len(self._frame)

    def __repr__(self) -> str:
        """Short summary: row count, ticker count and source."""
        return f"PricingData(rows={len(self)}, tickers={len(self.tickers)}, source={self.source!r})"


COVERAGE_COLUMNS = [
    "Ticker", "Description", "Source", "FirstDate", "LastDate", "Observations",
    "ExpectedObservations", "MissingObservations", "CoveragePct", "LargestGapDays",
    "LargestGapStart", "LargestGapEnd", "GapsOverThreshold", "LongestStaleRun",
    "DuplicateDates", "InvalidDates", "MissingPrices", "NonFinitePrices", "NonPositivePrices",
    "MissingSources", "WeekendObservations",
    "StartsLate", "EndsEarly", "Status", "Issues",
]  # fmt: skip


def _valid_observations(frame: pd.DataFrame) -> pd.Series:
    """Identify dated observations with finite, positive prices."""
    price = frame["Price"]
    return frame["Date"].notna() & np.isfinite(price) & price.gt(0)


def _longest_run(values: np.ndarray) -> int:
    """Length of the longest run of consecutive equal, non-missing values."""
    if len(values) == 0:
        return 0
    longest = current = 1
    for prev, cur in zip(values[:-1], values[1:], strict=True):
        current = current + 1 if (cur == prev and not np.isnan(cur)) else 1
        longest = max(longest, current)
    return longest


def _coverage_row(
    ticker: str,
    group: pd.DataFrame,
    calendars: dict[str, pd.DatetimeIndex],
    max_gap_days: int,
    stale_days: int,
) -> dict[str, Any]:
    """Build one :meth:`PricingData.coverage_report` row for one ticker's rows."""
    prices = group["Price"]
    dated = group["Date"].dropna()
    row = {
        "Ticker": ticker,
        "Description": group["Description"].iloc[0] if len(group) else None,
        "Source": group["Source"].iloc[0] if len(group) else None,
        "FirstDate": pd.NaT,
        "LastDate": pd.NaT,
        "Observations": 0,
        "ExpectedObservations": 0,
        "MissingObservations": 0,
        "CoveragePct": np.nan,
        "LargestGapDays": np.nan,
        "LargestGapStart": pd.NaT,
        "LargestGapEnd": pd.NaT,
        "GapsOverThreshold": 0,
        "LongestStaleRun": 0,
        "DuplicateDates": int(dated.duplicated().sum()),
        "InvalidDates": int(group["Date"].isna().sum()),
        "MissingPrices": int(prices.isna().sum()),
        "NonFinitePrices": int(np.isinf(prices).sum()),
        "NonPositivePrices": int((np.isfinite(prices) & prices.le(0)).sum()),
        "MissingSources": int(group["Source"].isna().sum()),
        "WeekendObservations": 0,
        "StartsLate": False,
        "EndsEarly": False,
    }
    issues = [
        f"{row[column]} {label}"
        for column, label in (
            ("DuplicateDates", "duplicate dates"),
            ("InvalidDates", "rows without a valid date"),
            ("MissingPrices", "rows without a price"),
            ("NonFinitePrices", "non-finite prices"),
            ("NonPositivePrices", "non-positive prices"),
            ("MissingSources", "rows without a source"),
        )
        if row[column]
    ]

    priced = group[_valid_observations(group)]
    if priced.empty:
        issues.insert(0, "no valid observations in window")
        return dict(row, Status="Review", Issues="; ".join(issues))

    dates = pd.DatetimeIndex(priced["Date"].unique()).sort_values()
    first, last = dates[0], dates[-1]
    calendar = calendars.get(row["Source"], dates)
    expected = calendar[(calendar >= first) & (calendar <= last)]
    missing = len(expected.difference(dates))

    days = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    if len(days):
        i = int(np.argmax(days))
        largest, gap_start, gap_end = int(days[i]), dates[i], dates[i + 1]
    else:
        largest, gap_start, gap_end = 0, pd.NaT, pd.NaT
    # Count long gaps only when the vendor priced other indices during them, so vendor-wide
    # holidays and outages don't flag every index.
    gaps_over = 0
    for j in np.flatnonzero(days > max_gap_days):
        if ((calendar > dates[j]) & (calendar < dates[j + 1])).any():
            gaps_over += 1

    prices = priced.drop_duplicates("Date")["Price"].to_numpy(dtype=float)
    stale = _longest_run(prices)
    weekend = int((dates.dayofweek >= 5).sum())
    starts_late = bool(len(calendar) and first > calendar[0])
    ends_early = bool(len(calendar) and last < calendar[-1])

    if missing:
        issues.append(f"{missing} missing vendor dates")
    if gaps_over:
        issues.append(f"{gaps_over} gaps over {max_gap_days} days with vendor prices "
                      f"(largest gap {largest} days)")
    if stale >= stale_days:
        issues.append(f"{stale} consecutive unchanged prices")
    if ends_early:
        issues.append(f"ends {last:%Y-%m-%d}, vendor data runs to {calendar[-1]:%Y-%m-%d}")

    coverage = (len(expected) - missing) / len(expected) * 100 if len(expected) else np.nan
    row.update({
        "FirstDate": first,
        "LastDate": last,
        "Observations": len(dates),
        "ExpectedObservations": len(expected),
        "MissingObservations": missing,
        "CoveragePct": round(coverage, 2),
        "LargestGapDays": largest,
        "LargestGapStart": gap_start,
        "LargestGapEnd": gap_end,
        "GapsOverThreshold": gaps_over,
        "LongestStaleRun": stale,
        "WeekendObservations": weekend,
        "StartsLate": starts_late,
        "EndsEarly": ends_early,
    })
    return dict(row, Status="Review" if issues else "OK", Issues="; ".join(issues))


# --------------------------------------------------------------------------- #
# Active (default) pricing data
# --------------------------------------------------------------------------- #
_active_data: PricingData | None = None
_path_cache: dict[Path, PricingData] = {}


def _load_cached(path: str | os.PathLike) -> PricingData:
    """Load a pricing file once per resolved path and reuse it."""
    key = Path(path).resolve()
    if key not in _path_cache:
        _path_cache[key] = PricingData.load(key)
    return _path_cache[key]


def _coerce_data(data: DataLike) -> PricingData:
    """Turn a :class:`PricingData`, DataFrame or path into :class:`PricingData`."""
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
        """Resolve the pricing data and set tickers and weights; see the class docstring."""
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
        """Tickers as a list."""
        return self._tickers

    @tickers.setter
    def tickers(self, value: str | Sequence[str]) -> None:
        """Set the tickers (a string, list or tuple) and clear cached values.

        Weights must be set again afterwards.

        Raises:
            ValueError: for any other type.
        """
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
        """Portfolio weights (a zero-dimensional ``array(1)`` for a single index)."""
        return self._weights

    @weights.setter
    def weights(self, value: Sequence[float] | float | None) -> None:
        """Validate and set the weights, then clear cached values.

        A single index always gets weight 1; ``None`` means equal weights.

        Raises:
            ValueError: if the count doesn't match the tickers or the weights don't sum to one.
        """
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
        """Keyword arguments that carry this instance's data and config to a new model."""
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
        """``True`` when there is exactly one ticker."""
        return len(self.tickers) == 1

    @cached_property
    def price_matrix(self) -> pd.DataFrame:
        """Prices, one column per ticker (in ticker order), indexed by date."""
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
        """Natural log of :attr:`price_matrix`."""
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
        """Vendor descriptions as one string, e.g. ``"Desc A (A)'; 'Desc B (B)"``.

        Indices are in sorted ticker order. :attr:`description` is the readable form.
        """
        df = self.data.rows(self.tickers).groupby("Ticker")["Description"].first()
        vals = [f"{df.iloc[i]} ({df.index[i]})" for i in range(df.shape[0])]
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

    # -------------------------------------------------------------- price QA
    def coverage_report(
        self,
        start: str | None = None,
        end: str | None = None,
        max_gap_days: int = 5,
        stale_days: int = 5,
    ) -> pd.DataFrame:
        """:meth:`PricingData.coverage_report` for this portfolio's indices.

        For a portfolio, ``MissingVsPortfolio`` counts dates within the overlap of the
        constituent histories when this constituent lacks a valid price and another has one.
        A portfolio without two consecutive dates of aligned prices is flagged for review.

        Args:
            start, end: restrict the check to this date window (``YYYY-MM-DD``).
            max_gap_days, stale_days: see :meth:`PricingData.coverage_report`.

        Returns:
            One row per constituent.
        """
        report = self.data.coverage_report(
            self.tickers, start=start, end=end, max_gap_days=max_gap_days, stale_days=stale_days
        )
        if self.single_index:
            return report

        window = self.data._window(start, end)
        rows = window.loc[window["Ticker"].isin(self.tickers) & window["Date"].notna()]
        present = (
            rows.assign(Priced=_valid_observations(rows))
            .groupby(["Date", "Ticker"])["Priced"].any().unstack(fill_value=False)
            .reindex(columns=self.tickers, fill_value=False)
        )
        aligned = present.all(axis=1)
        has_returns = bool((aligned & aligned.shift(fill_value=False)).any())

        spans = rows.loc[_valid_observations(rows)].groupby("Ticker")["Date"].agg(["min", "max"])
        report["MissingVsPortfolio"] = 0
        if len(spans) == len(self.tickers):
            first, last = spans["min"].max(), spans["max"].min()
            overlap = present.loc[first:last]
            missing = (~overlap.loc[overlap.any(axis=1)]).sum()
            report["MissingVsPortfolio"] = report["Ticker"].map(missing)

        for i, missing in report["MissingVsPortfolio"].items():
            issues = [report.at[i, "Issues"]]
            if missing:
                issues.append(f"{missing} dates priced by other constituents only")
            if not has_returns:
                issues.append("insufficient aligned prices for portfolio returns")
            report.at[i, "Issues"] = "; ".join(filter(None, issues))
            if report.at[i, "Issues"]:
                report.at[i, "Status"] = "Review"
        return report

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
        """Class name, tickers and weights."""
        return (
            f"{type(self).__name__}(tickers={self.tickers}, weights={self.weights.tolist()})"
        )
