"""Plotly charts. Every function returns a ``plotly.graph_objects.Figure``.

Figures are returned, not shown, so they chain::

    bt.backtest_plot().update_layout(height=700).write_html("backtest.html")
    bt.backtest_plot().show()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from volatility_factor.backtesting import VarBacktest
    from volatility_factor.data import VarData
    from volatility_factor.garch import GarchSim
    from volatility_factor.results import VaRResult

__all__ = [
    "COLORS",
    "SERIES_COLORS",
    "backtest_plot",
    "garch_fit_plot",
    "price_change_plot",
    "price_plot",
    "rolling_volatility_plot",
    "var_distribution_plot",
]

# Brand colours carried over from the original matplotlib charts.
COLORS = {
    "primary": "#003359",  # single-series lines, VaR threshold
    "positive": "#93B1CC",  # positive returns
    "negative": "#DA9694",  # negative returns that are not breaks
    "break": "#D62728",  # VaR breaks
    "break_line": "rgba(128,128,128,0.25)",
    "average_1": "#D62728",
    "average_2": "#D86DCD",
    "grid": "#E6E6E6",
    "axis": "#BDBDBD",
    "text": "#333333",
}
# Fixed-order categorical palette for multi-ticker charts (colour-blind checked).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")

LINE_WIDTH = 2
MARKER_SIZE = 8
DEFAULT_HEIGHT = 550


def _series_color(i: int) -> str:
    if i >= len(SERIES_COLORS):
        raise ValueError(
            f"More than {len(SERIES_COLORS)} series; plot the weighted series or split the chart."
        )
    return SERIES_COLORS[i]


def _style(
    fig: go.Figure,
    title: str,
    y_title: str,
    y_format: str | None = None,
    x_range: tuple[Any, Any] | None = None,
    hovermode: str = "x unified",
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": _wrap_title(title), "x": 0.0, "xanchor": "left"},
        height=DEFAULT_HEIGHT,
        hovermode=hovermode,
        font={"color": COLORS["text"]},
        legend={"orientation": "h", "yanchor": "top", "y": -0.08, "xanchor": "left", "x": 0.0},
        margin={"l": 70, "r": 30, "t": 90, "b": 80},
    )
    axis = {"showgrid": True, "gridcolor": COLORS["grid"], "gridwidth": 1,
            "linecolor": COLORS["axis"], "linewidth": 1, "zeroline": False}
    fig.update_xaxes(**axis)
    fig.update_yaxes(title_text=y_title, tickformat=y_format, **axis)
    if x_range is not None and any(v is not None for v in x_range):
        fig.update_xaxes(range=[pd.Timestamp(v) if v is not None else None for v in x_range])
    return fig


def _wrap_title(title: str, width: int = 90) -> str:
    """Break a long ``"<description>: <chart name>"`` title onto two lines."""
    if len(title) <= width or ": " not in title:
        return title
    head, _, tail = title.rpartition(": ")
    return f"{tail}<br><sup>{head}</sup>"


def _line(x, y, name: str, color: str, **kwargs: Any) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line={"color": color, "width": LINE_WIDTH}, **kwargs)


# --------------------------------------------------------------------------- #
# Backtesting charts
# --------------------------------------------------------------------------- #
def backtest_plot(
    bt: VarBacktest, title_text: str = "Standard", show_break_lines: bool = True
) -> go.Figure:
    """N-day returns against the VaR threshold.

    Light blue = positive return, light red = negative return within VaR, red ``x`` =
    break (with a faint vertical guide), dashed navy = VaR.
    """
    r = bt.nday_price_return
    q = pd.Series(data=-bt.var_pct, index=r.index)
    breaks = r[r <= q]
    positive = r[r > 0]
    negative = r[(q <= r) & (r <= 0)]

    def dots(s: pd.Series, name: str, color: str, symbol: str = "circle") -> go.Scatter:
        return go.Scatter(
            x=s.index, y=s.values, name=name, mode="markers",
            marker={"color": color, "size": MARKER_SIZE, "symbol": symbol,
                    "line": {"color": "white", "width": 1 if symbol == "circle" else 0}},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra>" + name + "</extra>",
        )

    fig = go.Figure()
    fig.add_trace(dots(positive, "Positive return", COLORS["positive"]))
    fig.add_trace(dots(negative, "Negative return", COLORS["negative"]))
    fig.add_trace(dots(breaks, f"Break ({len(breaks)})", COLORS["break"], symbol="x"))
    fig.add_trace(_line(q.index, q.values, f"VaR ({bt.var_pct:.2%})", COLORS["primary"],
                        hovertemplate="VaR %{y:.2%}<extra></extra>"))
    fig.data[-1].line.dash = "dash"
    if show_break_lines:
        for h in breaks.index:
            fig.add_vline(x=h, line={"color": COLORS["break_line"], "width": 1}, layer="below")

    if title_text == "Standard":
        title_text = f"{bt.description}: {bt.n_liquidation}-Day Backtesting Results"
    return _style(fig, title_text, f"{bt.n_liquidation}-Day Percent Change", ".2%",
                  x_range=(bt.start_date, bt.end_date), hovermode="closest")


def price_change_plot(
    bt: VarBacktest, title_text: str = "Standard", weighted: bool = True
) -> go.Figure:
    """N-day percentage change: the weighted series, or one line per index."""
    fig = go.Figure()
    if weighted:
        r = bt.nday_price_return
        fig.add_trace(_line(r.index, r.values, bt.ticker_description, COLORS["primary"]))
        fig.update_layout(showlegend=False)
    else:
        df = bt.price_matrix.sort_index().copy()
        pct_chg = (df / df.shift(bt.n_liquidation) - 1).dropna()[bt.start_date : bt.end_date]
        for i, col in enumerate(pct_chg.columns):
            fig.add_trace(_line(pct_chg.index, pct_chg[col], col, _series_color(i)))
        fig.update_layout(showlegend=len(pct_chg.columns) > 1)

    if title_text == "Standard":
        title_text = f"{bt.description}: {bt.n_liquidation}-Day Percentage Change"
    fig.update_traces(hovertemplate="%{y:.2%}")
    return _style(fig, title_text, f"{bt.n_liquidation}-Day Percent Change", ".2%")


def price_plot(
    vd: VarData,
    title_text: str = "Standard",
    weighted: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> go.Figure:
    """Index prices: the weighted price, or one line per index."""
    df = vd.price_matrix[start_date:end_date].sort_index().copy()
    fig = go.Figure()
    if weighted:
        ser = pd.Series(data=df.values.dot(vd.weights), index=df.index)
        fig.add_trace(_line(ser.index, ser.values, "Weighted price", COLORS["primary"]))
        fig.update_layout(showlegend=False)
    elif df.shape[1] == 1:
        fig.add_trace(_line(df.index, df.iloc[:, 0], df.columns[0], COLORS["primary"]))
        fig.update_layout(showlegend=False)
    else:
        for i, col in enumerate(df.columns):
            fig.add_trace(_line(df.index, df[col], col, _series_color(i)))

    if title_text == "Standard":
        title_text = f"{vd.description}: Price"
    fig.update_traces(hovertemplate="%{y:,.2f}")
    return _style(fig, title_text, "Price")


# --------------------------------------------------------------------------- #
# Volatility charts
# --------------------------------------------------------------------------- #
def rolling_volatility_plot(
    vd: VarData,
    window: int = 30,
    periods_per_year: int = 252,
    averages: Mapping[str, tuple[str | None, str | None]] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Rolling annualised realised volatility, with optional period averages.

    Args:
        window: rolling window in days.
        periods_per_year: annualisation factor.
        averages: ``{label: (start, end)}``; each draws the mean volatility over that
            slice as a flat line, e.g. ``{"2024 - Avg": ("2024-01-01", None),
            "Pre2022 Avg": (None, "2021-12-31")}``.
        start_date, end_date: visible x-axis range.
        title: chart title (defaults to the index description).
    """
    vol = vd.rolling_volatility(window=window, periods_per_year=periods_per_year)
    fig = go.Figure()
    fig.add_trace(_line(vol.index, vol.values, "Annualized volatility", COLORS["primary"]))
    avg_colors = [COLORS["average_1"], COLORS["average_2"], *SERIES_COLORS]
    for i, (label, (a, b)) in enumerate((averages or {}).items()):
        level = vol.loc[a:b].mean()
        fig.add_trace(_line(vol.index, np.full(len(vol), level), f"{label} ({level:.1%})",
                            avg_colors[i % len(avg_colors)]))
    fig.update_traces(hovertemplate="%{y:.1%}")
    fig.update_layout(showlegend=bool(averages))
    title = title or f"{vd.description}: Annualized Volatility - Rolling {window}-Day"
    return _style(fig, title, "Realized Volatility (Annualized)", ".0%",
                  x_range=(start_date, end_date))


def garch_fit_plot(garch: GarchSim, ticker: str | None = None, annualize: int = 252) -> go.Figure:
    """Standardized residuals and annualised conditional volatility for one constituent.

    Conditional volatility is converted back to return units (divided by ``yscale``)
    before annualising.
    """
    ticker = ticker or garch.tickers[0]
    fit = garch.fits[ticker]
    std_resid = fit.result.std_resid
    cond_vol = fit.result.conditional_volatility / fit.yscale * np.sqrt(annualize)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Standardized residuals",
                                        "Annualized conditional volatility"))
    fig.add_trace(_line(std_resid.index, std_resid.values, "Std. residual", COLORS["primary"],
                        hovertemplate="%{y:.2f}"), row=1, col=1)
    fig.add_trace(_line(cond_vol.index, cond_vol.values, "Conditional vol.", COLORS["primary"],
                        hovertemplate="%{y:.1%}"), row=2, col=1)
    fig = _style(fig, f"GARCH(1,1) fit: {ticker} ({garch.sample_start} to {garch.sample_end})",
                 "")
    fig.update_yaxes(title_text="Std. residual", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", tickformat=".0%", row=2, col=1)
    fig.update_layout(showlegend=False, height=700)
    return fig


def var_distribution_plot(result: VaRResult, bins: int = 60) -> go.Figure:
    """Distribution of per-path n-day VaR, with the reported VaR quantiles marked."""
    per_path = -(np.exp(result.var_by_path / 100) - 1)  # arithmetic, positive = loss
    fig = go.Figure(go.Histogram(
        x=per_path.values, nbinsx=bins, name="Per-path VaR",
        marker={"color": COLORS["positive"], "line": {"color": "white", "width": 2}},
        hovertemplate="VaR %{x:.2%}<br>%{y} paths<extra></extra>",
    ))
    for i, level in enumerate((0.50, 0.75, 0.95, 0.99)):
        value = result.var(level) / 100
        fig.add_vline(x=value, line={"color": COLORS["primary"], "width": LINE_WIDTH,
                                     "dash": "dot"})
        # Stagger labels vertically so neighbouring quantiles don't collide.
        fig.add_annotation(x=value, y=1.0 - 0.07 * (i % 2), xref="x", yref="paper",
                           text=f"VaR {round(level * 100)}: {value:.2%}", showarrow=False,
                           xanchor="left", xshift=4, bgcolor="rgba(255,255,255,0.8)")
    title = (f"{result['Tickers']} {result['NDays']}-day VaR ({result['Method']}, "
             f"{result['Simulations']:,} paths)")
    fig.update_layout(showlegend=False, bargap=0)
    return _style(fig, title, "Paths", hovermode="closest").update_xaxes(
        title_text="Per-path VaR", tickformat=".1%"
    )
