"""
views/chart_renderer.py
-----------------------
Pure rendering component. No business logic.
"""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detectors.pattern_detector import PatternSignal
from core.trade_engine import Position

logger = logging.getLogger(__name__)

# Visual constants
BG_COLOR       = "#0d1117"
PANEL_COLOR    = "#161b22"
GRID_COLOR     = "#21262d"
TEXT_COLOR     = "#c9d1d9"
BULL_COLOR     = "#3fb950"
BEAR_COLOR     = "#f85149"
VOLUME_BULL    = "rgba(63,185,80,0.35)"
VOLUME_BEAR    = "rgba(248,81,73,0.35)"
FONT_FAMILY    = "JetBrains Mono, Menlo, monospace"


class ChartRenderer:

    @classmethod
    def render_chart(
        cls,
        market_data: pd.DataFrame,
        patterns: Optional[list[PatternSignal]] = None,
        trades: Optional[list[Position]] = None,
        symbol: str = "EURUSD",
        timeframe: str = "M1",
        show_volume: bool = True,
        max_candles: int = 100,
    ) -> go.Figure:
        patterns = patterns or []
        trades   = trades   or []

        display_df = market_data.iloc[-max_candles:] if len(market_data) > max_candles else market_data
        if display_df.empty:
            return cls._empty_figure(symbol, timeframe)

        row_heights = [0.75, 0.25] if show_volume and "volume" in display_df.columns else [1.0]
        rows = 2 if len(row_heights) == 2 else 1

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)

        # EMAs
        if len(display_df) >= 21:
            close = display_df["close"]
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            fig.add_trace(go.Scatter(x=display_df.index, y=ema9, name="EMA 9", line=dict(color="#58a6ff", width=1)))
            fig.add_trace(go.Scatter(x=display_df.index, y=ema21, name="EMA 21", line=dict(color="#f0883e", width=1)))

        cls._add_candlesticks(fig, display_df, row=1)

        if patterns:
            cls._add_pattern_annotations(fig, display_df, patterns)   # vertical rectangles and L/S labels

        if trades:
            cls._add_trade_markers(fig, trades, display_df, row=1)

        if rows == 2:
            cls._add_volume(fig, display_df, row=2)

        cls._apply_layout(fig, symbol, timeframe, display_df, rows)
        return fig

    @staticmethod
    def _add_candlesticks(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                name="Price",
                increasing=dict(line=dict(color=BULL_COLOR, width=1), fillcolor=BULL_COLOR),
                decreasing=dict(line=dict(color=BEAR_COLOR, width=1), fillcolor=BEAR_COLOR),
                whiskerwidth=0.5,
            ),
            row=row, col=1,
        )

    @staticmethod
    def _add_volume(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
        colors = [VOLUME_BULL if c >= o else VOLUME_BEAR for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume", showlegend=False), row=row, col=1)

    @staticmethod
    def _add_pattern_annotations(fig: go.Figure, df: pd.DataFrame, patterns: list[PatternSignal]) -> None:
        # Collect L and S positions
        long_x, long_y = [], []
        short_x, short_y = [], []

        for signal in patterns:
            # Vertical shaded rectangle (only for patterns that are not single-candle? but fine)
            if signal.end_time < df.index[0] or signal.start_time > df.index[-1]:
                continue

            fig.add_vrect(
                x0=signal.start_time, x1=signal.end_time,
                fillcolor="rgba(255,220,50,0.10)", line_width=0,
                annotation_text=signal.name, annotation_position="top left",
                annotation=dict(font=dict(color="#f0e68c", size=9, family=FONT_FAMILY)),
            )

            # For strategy signals, prepare L/S markers
            if signal.name in ("STRATEGY_LONG", "STRATEGY_SHORT"):
                # Find the exact row in df (display_df) that matches signal.end_time
                mask = df.index == signal.end_time
                if not mask.any():
                    continue
                row = df[mask].iloc[0]
                direction = signal.metadata.get("direction", "")
                if direction == "LONG":
                    long_x.append(signal.end_time)
                    # Place L above the candle (high + 0.3 * range)
                    y_pos = row["high"] + (row["high"] - row["low"]) * 0.3
                    long_y.append(y_pos)
                elif direction == "SHORT":
                    short_x.append(signal.end_time)
                    y_pos = row["low"] - (row["high"] - row["low"]) * 0.3
                    short_y.append(y_pos)

        # Add L and S text traces (reliable)
        if long_x:
            fig.add_trace(go.Scatter(
                x=long_x, y=long_y,
                mode="text",
                text=["L"] * len(long_x),
                textfont=dict(size=16, color=BULL_COLOR, family=FONT_FAMILY, weight="bold"),
                textposition="middle center",
                showlegend=False,
                hoverinfo="text",
                hovertext=[f"LONG<br>Trend: {p.metadata.get('trend','')}<br>Stoch: {p.metadata.get('stoch_fastk',0):.1f}" 
                           for p in patterns if p.name=="STRATEGY_LONG" and p.end_time in long_x],
            ), row=1, col=1)

        if short_x:
            fig.add_trace(go.Scatter(
                x=short_x, y=short_y,
                mode="text",
                text=["S"] * len(short_x),
                textfont=dict(size=16, color=BEAR_COLOR, family=FONT_FAMILY, weight="bold"),
                textposition="middle center",
                showlegend=False,
                hoverinfo="text",
                hovertext=[f"SHORT<br>Trend: {p.metadata.get('trend','')}<br>Stoch: {p.metadata.get('stoch_fastk',0):.1f}"
                           for p in patterns if p.name=="STRATEGY_SHORT" and p.end_time in short_x],
            ), row=1, col=1)

    @staticmethod
    def _add_trade_markers(fig: go.Figure, trades: list[Position], df: pd.DataFrame, row: int) -> None:
        entries_buy_x, entries_buy_y = [], []
        entries_sell_x, entries_sell_y = [], []
        exits_x, exits_y = [], []

        for pos in trades:
            if pos.direction == "BUY":
                entries_buy_x.append(pos.entry_time)
                entries_buy_y.append(pos.entry_price)
            else:
                entries_sell_x.append(pos.entry_time)
                entries_sell_y.append(pos.entry_price)
            if pos.exit_time and pos.exit_price:
                exits_x.append(pos.exit_time)
                exits_y.append(pos.exit_price)

        if entries_buy_x:
            fig.add_trace(go.Scatter(x=entries_buy_x, y=entries_buy_y, mode="markers+text",
                                     marker=dict(symbol="triangle-up", size=12, color=BULL_COLOR),
                                     text=["Buy"]*len(entries_buy_x), textposition="top center",
                                     textfont=dict(size=9, color=BULL_COLOR), name="Buy Entry", showlegend=False), row=row, col=1)
        if entries_sell_x:
            fig.add_trace(go.Scatter(x=entries_sell_x, y=entries_sell_y, mode="markers+text",
                                     marker=dict(symbol="triangle-down", size=12, color=BEAR_COLOR),
                                     text=["Sell"]*len(entries_sell_x), textposition="bottom center",
                                     textfont=dict(size=9, color=BEAR_COLOR), name="Sell Entry", showlegend=False), row=row, col=1)
        if exits_x:
            fig.add_trace(go.Scatter(x=exits_x, y=exits_y, mode="markers+text",
                                     marker=dict(symbol="x", size=10, color=TEXT_COLOR),
                                     text=["Exit"]*len(exits_x), textposition="top center",
                                     textfont=dict(size=8, color=TEXT_COLOR), name="Exit", showlegend=False), row=row, col=1)

    @staticmethod
    def _apply_layout(fig: go.Figure, symbol: str, timeframe: str, df: pd.DataFrame, rows: int) -> None:
        shared_axis = dict(showgrid=True, gridcolor=GRID_COLOR, gridwidth=1, zeroline=False,
                           color=TEXT_COLOR, tickfont=dict(family=FONT_FAMILY, size=10, color=TEXT_COLOR))
        fig.update_layout(
            title=dict(text=f"{symbol} · {timeframe}", font=dict(family=FONT_FAMILY, size=13, color=TEXT_COLOR), x=0.01),
            paper_bgcolor=BG_COLOR, plot_bgcolor=PANEL_COLOR,
            font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT_COLOR)),
            margin=dict(l=55, r=20, t=40, b=30), hovermode="x unified",
            xaxis=dict(**shared_axis, rangeslider=dict(visible=False), type="date"),
            yaxis=dict(**shared_axis, side="right"),
        )
        if rows == 2:
            fig.update_layout(yaxis2=dict(**shared_axis, side="right"), xaxis2=dict(**shared_axis, rangeslider=dict(visible=False)))

    @staticmethod
    def _empty_figure(symbol: str, timeframe: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=PANEL_COLOR,
                          font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                          annotations=[dict(text="No data loaded", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=TEXT_COLOR))],
                          margin=dict(l=55, r=20, t=40, b=30), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        return fig