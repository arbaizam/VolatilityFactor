"""Tests for the price-observation QA report (``PricingData.coverage_report`` and friends).

Uses a small synthetic data set with known problems:

* ``A`` (vendor V): complete, plus one weekend price that no other index has.
* ``B`` (vendor V): two business days missing, and an 8-day gap while V priced others.
* ``C`` (vendor V): six unchanged prices in a row; stops two weeks before V's last date.
* ``D`` (vendor V): a duplicated date, a row without a price, and a zero price.
* ``E`` (vendor W): misses a date that no W index is priced on (a W holiday), so it's OK.
* ``F`` (vendor V): complete, but starts mid-January.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from volatility_factor import PricingData, VarData

DAYS = pd.bdate_range("2024-01-01", "2024-02-29")
WEEKEND = pd.Timestamp("2024-01-31") + pd.offsets.Day(3)  # Saturday 2024-02-03
B_MISSING = [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]
B_GAP = pd.bdate_range("2024-02-06", "2024-02-12")[1:-1]  # 2024-02-07 .. 2024-02-09
W_HOLIDAY = pd.Timestamp("2024-01-15")


def _rows(ticker: str, source: str | None, dates, prices=None) -> pd.DataFrame:
    """Price rows for one synthetic ticker (a gentle upward drift unless given)."""
    dates = pd.DatetimeIndex(dates)
    if prices is None:
        prices = 100 + np.arange(len(dates)) * 0.1
    return pd.DataFrame({"Date": dates, "Ticker": ticker, "Source": source,
                         "Description": f"Index {ticker}", "Price": prices})


@pytest.fixture(scope="module")
def synthetic() -> PricingData:
    """The synthetic data set described in the module docstring."""
    a = _rows("A", "V", DAYS.append(pd.DatetimeIndex([WEEKEND])).sort_values())
    b = _rows("B", "V", DAYS.difference(pd.DatetimeIndex(B_MISSING)).difference(B_GAP))
    c_dates = DAYS[DAYS <= "2024-02-15"]
    c_prices = 100 + np.arange(len(c_dates)) * 0.1
    c_prices[10:16] = c_prices[10]  # six identical prices
    c = _rows("C", "V", c_dates, c_prices)
    d = _rows("D", "V", DAYS)
    d.loc[5, "Price"] = np.nan
    d.loc[6, "Price"] = 0.0
    d = pd.concat([d, d.iloc[[20]]])  # duplicated date
    e = _rows("E", "W", DAYS.difference(pd.DatetimeIndex([W_HOLIDAY])))
    f = _rows("F", "V", DAYS[DAYS >= "2024-01-15"])
    return PricingData.from_frame(pd.concat([a, b, c, d, e, f], ignore_index=True))


@pytest.fixture(scope="module")
def report(synthetic) -> pd.DataFrame:
    """Coverage report for the synthetic data, indexed by ticker."""
    return synthetic.coverage_report().set_index("Ticker")


def test_complete_index_is_ok(report):
    """A complete index passes, and its weekend print is informational only."""
    a = report.loc["A"]
    assert a.Status == "OK" and a.Issues == ""
    assert a.MissingObservations == 0 and a.CoveragePct == 100.0
    assert a.WeekendObservations == 1
    assert a.ExpectedObservations == len(DAYS)  # the weekend date is not a vendor date


def test_missing_dates_and_gap(report):
    """Missing vendor dates and a long gap during vendor pricing are flagged."""
    b = report.loc["B"]
    assert b.MissingObservations == len(B_MISSING) + len(B_GAP)
    assert b.LargestGapDays == 6
    assert (b.LargestGapStart, b.LargestGapEnd) == (pd.Timestamp("2024-02-06"),
                                                   pd.Timestamp("2024-02-12"))
    assert b.GapsOverThreshold == 1
    assert b.Status == "Review" and "missing vendor dates" in b.Issues
    assert b.CoveragePct == pytest.approx((len(DAYS) - 5) / len(DAYS) * 100, abs=0.01)


def test_stale_run_and_early_end(report):
    """A run of unchanged prices and an early last date are flagged."""
    c = report.loc["C"]
    assert c.LongestStaleRun == 6
    assert bool(c.EndsEarly) and c.LastDate == pd.Timestamp("2024-02-15")
    assert "6 consecutive unchanged prices" in c.Issues and "ends 2024-02-15" in c.Issues


def test_bad_rows(report):
    """Duplicate dates, rows without a price and non-positive prices are counted."""
    d = report.loc["D"]
    assert (d.DuplicateDates, d.MissingPrices, d.NonPositivePrices) == (1, 1, 1)
    assert d.MissingObservations == 2
    assert d.Status == "Review"


def test_vendor_holiday_not_missing(report):
    """A date no index of the vendor is priced on is not counted as missing."""
    e = report.loc["E"]
    assert e.MissingObservations == 0 and e.Status == "OK"


def test_thresholds_and_window(synthetic):
    """Thresholds and the date window change what is flagged."""
    lenient = synthetic.coverage_report(["C"], stale_days=7).set_index("Ticker").loc["C"]
    assert "unchanged" not in lenient.Issues
    early = synthetic.coverage_report("B", end="2024-01-12").set_index("Ticker").loc["B"]
    assert early.Status == "OK"
    with pytest.raises(KeyError, match="ZZZ"):
        synthetic.coverage_report(["ZZZ"])


def test_gaps(synthetic):
    """gaps() lists each gap with the vendor dates it skips."""
    gaps = synthetic.gaps("B", min_days=3)
    assert list(gaps.columns) == ["LastPriceDate", "NextPriceDate", "CalendarDays",
                                  "MissingVendorDates"]
    row = gaps.loc[gaps["CalendarDays"].idxmax()]
    assert row.CalendarDays == 6 and row.MissingVendorDates == 3
    assert synthetic.gaps("A", min_days=10).empty


def test_observation_counts(synthetic):
    """Monthly counts and coverage, with coverage capped at 100%."""
    counts = synthetic.observation_counts(tickers=["A", "B"])
    assert counts.loc["2024-01-31", "B"] == len(DAYS[DAYS.month == 1]) - 2
    cov = synthetic.observation_counts(tickers=["A", "B"], normalize=True)
    assert cov["A"].max() == 1.0  # the weekend print doesn't push it above 100%
    assert cov.loc["2024-02-29", "B"] == pytest.approx(1 - 3 / len(DAYS[DAYS.month == 2]))
    # An index that starts mid-month is fully covered from its first price.
    late = synthetic.observation_counts(tickers=["F"], normalize=True)["F"]
    assert late.loc["2024-01-31"] == 1.0
    report = synthetic.coverage_report("F").set_index("Ticker").loc["F"]
    assert bool(report.StartsLate) and report.Status == "OK"


def test_portfolio_report(synthetic):
    """VarData adds dates priced by other constituents only."""
    report = VarData(["A", "B"], data=synthetic).coverage_report().set_index("Ticker")
    # B's missing dates, plus A's weekend print (a date only A is priced on).
    assert report.loc["B", "MissingVsPortfolio"] == len(B_MISSING) + len(B_GAP) + 1
    assert report.loc["A", "MissingVsPortfolio"] == 0
    assert "priced by other constituents only" in report.loc["B", "Issues"]
    assert "MissingVsPortfolio" not in VarData("A", data=synthetic).coverage_report()


def test_portfolio_report_with_duplicate_dates(synthetic):
    """Duplicate dates are reported in the requested window."""
    portfolio = VarData(["A", "D"], data=synthetic)
    report = portfolio.coverage_report().set_index("Ticker")
    assert report.loc["D", "DuplicateDates"] == 1
    assert report.loc["D", "Status"] == "Review"
    early = portfolio.coverage_report(end="2024-01-05").set_index("Ticker")
    assert early.loc["D", "DuplicateDates"] == 0
    assert early.loc["D", "Status"] == "OK"


@pytest.mark.parametrize("first_b", ["2024-01-03", "2024-01-04"])
def test_portfolio_report_without_returns(first_b):
    """A portfolio with at most one aligned date cannot produce a return."""
    a = _rows("A", "V", pd.date_range("2024-01-01", periods=3))
    b = _rows("B", "W", pd.date_range(first_b, periods=3))
    data = PricingData.from_frame(pd.concat([a, b]))
    report = VarData(["A", "B"], data=data).coverage_report()
    assert (report["Status"] == "Review").all()
    assert report["Issues"].str.contains("insufficient aligned prices").all()


def test_invalid_observations(tmp_path):
    """Undated observations and infinite prices are excluded from coverage and counts."""
    rows = _rows("A", "V", [pd.Timestamp("2024-01-02"), pd.NaT, *DAYS[2:4]], [100, 101, np.inf, 105])
    data = PricingData.from_frame(rows)
    result = data.coverage_report(start="2024-01-01").iloc[0]
    assert (result.InvalidDates, result.NonFinitePrices, result.Observations) == (1, 1, 2)
    assert result.Status == "Review"
    assert data.observation_counts().loc["2024-01-31", "A"] == 2
    malformed = rows.astype({"Date": "string"}).fillna({"Date": "invalid"})
    path = tmp_path / "prices.csv"
    malformed.to_csv(path, index=False)
    pd.testing.assert_frame_equal(PricingData.from_csv(path).coverage_report(), data.coverage_report())


def test_missing_vendor_calendar():
    """A missing source is flagged and has no normalized observation counts."""
    data = PricingData.from_frame(_rows("A", None, DAYS))
    report = data.coverage_report().iloc[0]
    assert report.MissingSources == len(DAYS) and report.Status == "Review"
    assert data.observation_counts(normalize=True)["A"].isna().all()


def test_heatmap(synthetic):
    """The heatmap has one row per ticker."""
    fig = synthetic.plot_observation_counts(tickers=["A", "B", "C"])
    assert isinstance(fig, go.Figure) and list(fig.data[0].y) == ["A", "B", "C"]
