"""
views/chart_renderer.py
-----------------------
Pure rendering component. No business logic.

Renders 3 stacked subplots (H1 / M5 / M1) with shared x-axis.
L/S markers appear on M1 row. H1 row shows trend context.
Strategy-agnostic — uses "direction" from signal metadata.
"""

from __future__ import annotations
import logging
from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detectors.signal import PatternSignal
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
H1_UP_COLOR    = "rgba(63,185,80,0.08)"    # subtle green bg for H1 uptrend
H1_DOWN_COLOR  = "rgba(248,81,73,0.08)"    # subtle red bg for H1 downtrend


class ChartRenderer:

    @classmethod
    def render_chart(
        cls,
        market_data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        patterns: Optional[list[PatternSignal]] = None,
        trades: Optional[list[Position]] = None,
        symbol: str = "EURUSD",
        timeframe: str = "M1",
        show_volume: bool = True,
        max_candles: int = 100,
    ) -> go.Figure:
        """
        Render chart.

        Parameters
        ----------
        market_data : DataFrame or dict[str, DataFrame]
            Single DataFrame (legacy) or dict with keys "H1", "M5", "M1".
        patterns : list[PatternSignal], optional
            Signal markers to render (L/S on M1 row).
        trades : list[Position], optional
            Trade entry/exit markers on M1 row.
        """
        patterns = patterns or []
        trades = trades or []

        # --- Normalize input ---
        if isinstance(market_data, dict):
            tf_data = market_data
            h1_df = tf_data.get("H1", pd.DataFrame())
            m5_df = tf_data.get("M5", pd.DataFrame())
            m1_df = tf_data.get("M1", pd.DataFrame())
        else:
            # Legacy: single DataFrame → use as M1 only
            m1_df = market_data
            m5_df = pd.DataFrame()
            h1_df = pd.DataFrame()

        # --- Trim to max_candles ---
        if len(m1_df) > max_candles:
            m1_df = m1_df.iloc[-max_candles:]
        if len(m5_df) > max_candles:
            m5_df = m5_df.iloc[-max_candles:]
        if len(h1_df) > max_candles:
            h1_df = h1_df.iloc[-max_candles:]

        if m1_df.empty:
            return cls._empty_figure(symbol, timeframe)

        # --- Determine layout ---
        has_h1 = not h1_df.empty
        has_m5 = not m5_df.empty
        has_volume = show_volume and "volume" in m1_df.columns

        if has_h1 and has_m5:
            # 3 rows: H1 + M5 + M1
            row_heights = [0.25, 0.30, 0.45] if has_volume else [0.30, 0.35, 0.35]
            rows = 3
            tf_labels = ["H1", "M5", "M1"]
        elif has_m5:
            # 2 rows: M5 + M1
            row_heights = [0.35, 0.65] if has_volume else [0.40, 0.60]
            rows = 2
            tf_labels = ["M5", "M1"]
        else:
            # 1 row: M1 only (legacy mode)
            row_heights = [0.75, 0.25] if has_volume else [1.0]
            rows = 2 if has_volume else 1
            tf_labels = ["M1"]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.015,
            row_heights=row_heights,
        )

        # --- Row mapping ---
        row_map = {tf: i + 1 for i, tf in enumerate(tf_labels)}

        # ==============================================================
        # H1 row (top)
        # ==============================================================
        if "H1" in row_map and not h1_df.empty:
            r = row_map["H1"]
            cls._add_candlesticks(fig, h1_df, row=r, name="H1")

            # H1 EMAs
            if len(h1_df) >= 21:
                close = h1_df["close"]
                ema9 = close.ewm(span=9, adjust=False).mean()
                ema21 = close.ewm(span=21, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=h1_df.index, y=ema9, name="H1 EMA 9",
                    line=dict(color="#58a6ff", width=1),
                    showlegend=False,
                ), row=r, col=1)
                fig.add_trace(go.Scatter(
                    x=h1_df.index, y=ema21, name="H1 EMA 21",
                    line=dict(color="#f0883e", width=1),
                    showlegend=False,
                ), row=r, col=1)

            # H1 trend band (green/red background)
            if len(h1_df) >= 21:
                ema_f = close.ewm(span=9, adjust=False).mean()
                ema_s = close.ewm(span=21, adjust=False).mean()
                for i in range(1, len(h1_df)):
                    if ema_f.iloc[i] > ema_s.iloc[i]:
                        fig.add_vrect(
                            x0=h1_df.index[i - 1], x1=h1_df.index[i],
                            fillcolor=H1_UP_COLOR, line_width=0,
                            layer="below", row=r, col=1,
                        )
                    else:
                        fig.add_vrect(
                            x0=h1_df.index[i - 1], x1=h1_df.index[i],
                            fillcolor=H1_DOWN_COLOR, line_width=0,
                            layer="below", row=r, col=1,
                        )

        # ==============================================================
        # M5 row (middle)
        # ==============================================================
        if "M5" in row_map and not m5_df.empty:
            r = row_map["M5"]
            cls._add_candlesticks(fig, m5_df, row=r, name="M5")

            # M5 stochastic zones (overbought/oversold bands)
            if len(m5_df) >= 8:
                m5_high = m5_df["high"].values.astype("float64")
                m5_low = m5_df["low"].values.astype("float64")
                m5_close = m5_df["close"].values.astype("float64")
                try:
                    import talib
                    fastk, _ = talib.STOCHF(m5_high, m5_low, m5_close, fastk_period=5, fastd_period=3)
                    fig.add_trace(go.Scatter(
                        x=m5_df.index, y=fastk, name="M5 Stoch %K",
                        line=dict(color="#a371f7", width=1),
                        showlegend=False,
                    ), row=r, col=1)
                    # Overbought/oversold reference lines
                    fig.add_hline(y=80, line=dict(color="#f8514966", width=1, dash="dot"), row=r, col=1)
                    fig.add_hline(y=20, line=dict(color="#3fb95066", width=1, dash="dot"), row=r, col=1)
                except Exception:
                    pass  # talib not available

        # ==============================================================
        # M1 row (bottom) — main entry chart
        # ==============================================================
        if "M1" in row_map:
            r = row_map["M1"]
            cls._add_candlesticks(fig, m1_df, row=r, name="M1")

            # M1 EMAs
            if len(m1_df) >= 21:
                close = m1_df["close"]
                ema9 = close.ewm(span=9, adjust=False).mean()
                ema21 = close.ewm(span=21, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=m1_df.index, y=ema9, name="EMA 9",
                    line=dict(color="#58a6ff", width=1),
                ), row=r, col=1)
                fig.add_trace(go.Scatter(
                    x=m1_df.index, y=ema21, name="EMA 21",
                    line=dict(color="#f0883e", width=1),
                ), row=r, col=1)

            # L/S signal markers
            if patterns:
                cls._add_signal_markers(fig, m1_df, patterns, row=r)

            # Trade markers
            if trades:
                cls._add_trade_markers(fig, trades, m1_df, row=r)

            # Volume
            if has_volume:
                # Volume as separate row below M1
                vol_row = r + 1
                if vol_row <= rows:
                    cls._add_volume(fig, m1_df, row=vol_row)
                else:
                    # Inline volume on M1 row (fallback)
                    cls._add_volume(fig, m1_df, row=r)

        cls._apply_layout(fig, symbol, timeframe, rows)
        return fig

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    @staticmethod
    def _add_candlesticks(fig: go.Figure, df: pd.DataFrame, row: int, name: str = "Price") -> None:
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                name=name,
                increasing=dict(line=dict(color=BULL_COLOR, width=1), fillcolor=BULL_COLOR),
                decreasing=dict(line=dict(color=BEAR_COLOR, width=1), fillcolor=BEAR_COLOR),
                whiskerwidth=0.5,
                showlegend=False,
            ),
            row=row, col=1,
        )

    @staticmethod
    def _add_volume(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
        colors = [VOLUME_BULL if c >= o else VOLUME_BEAR for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=colors, name="Volume", showlegend=False,
        ), row=row, col=1)

    @staticmethod
    def _add_signal_markers(
        fig: go.Figure,
        df: pd.DataFrame,
        patterns: list[PatternSignal],
        row: int,
    ) -> None:
        """
        Render L/S markers + TP/SL lines for any signal with "direction" in metadata.
        """
        long_x, long_y, long_meta = [], [], []
        short_x, short_y, short_meta = [], [], []
        tp_sl_lines = []  # (x0, x1, y, color, label)

        for signal in patterns:
            direction = signal.metadata.get("direction", "")
            if direction not in ("LONG", "SHORT"):
                continue

            if signal.end_time < df.index[0] or signal.start_time > df.index[-1]:
                continue

            mask = df.index == signal.end_time
            if not mask.any():
                continue
            candle = df[mask].iloc[0]
            candle_range = candle["high"] - candle["low"]

            # Build hover context
            strategy_name = signal.metadata.get("strategy", signal.name)
            entry = signal.metadata.get("entry_price", 0)
            tp = signal.metadata.get("take_profit", 0)
            sl = signal.metadata.get("stop_loss", 0)
            atr = signal.metadata.get("atr", 0)

            hover_text = (
                f"<b>{direction}</b><br>"
                f"Strategy: {strategy_name}<br>"
                f"Entry: {entry:.5f}<br>"
                f"TP: {tp:.5f}<br>"
                f"SL: {sl:.5f}<br>"
                f"ATR: {atr:.5f}"
            )

            if direction == "LONG":
                long_x.append(signal.end_time)
                long_y.append(candle["high"] + candle_range * 0.3)
                long_meta.append(hover_text)
            elif direction == "SHORT":
                short_x.append(signal.end_time)
                short_y.append(candle["low"] - candle_range * 0.3)
                short_meta.append(hover_text)

            # TP/SL horizontal lines
            if tp and sl:
                # Line extends from signal candle to right edge of visible data
                x_start = signal.end_time
                x_end = df.index[-1]

                tp_sl_lines.append((x_start, x_end, tp, "#3fb950", f"TP {tp:.5f}"))
                tp_sl_lines.append((x_start, x_end, sl, "#f85149", f"SL {sl:.5f}"))

        # Long markers
        if long_x:
            fig.add_trace(go.Scatter(
                x=long_x, y=long_y,
                mode="text",
                text=["L"] * len(long_x),
                textfont=dict(size=16, color=BULL_COLOR, family=FONT_FAMILY, weight="bold"),
                textposition="middle center",
                showlegend=False,
                hoverinfo="text",
                hovertext=long_meta,
            ), row=row, col=1)

        # Short markers
        if short_x:
            fig.add_trace(go.Scatter(
                x=short_x, y=short_y,
                mode="text",
                text=["S"] * len(short_x),
                textfont=dict(size=16, color=BEAR_COLOR, family=FONT_FAMILY, weight="bold"),
                textposition="middle center",
                showlegend=False,
                hoverinfo="text",
                hovertext=short_meta,
            ), row=row, col=1)

        # TP/SL lines
        for x0, x1, y, color, label in tp_sl_lines:
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y, y],
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                showlegend=False,
                hoverinfo="text",
                hovertext=[f"{label}", f"{label}"],
            ), row=row, col=1)

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
    def _apply_layout(fig: go.Figure, symbol: str, timeframe: str, rows: int) -> None:
        shared_axis = dict(
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1, zeroline=False,
            color=TEXT_COLOR,
            tickfont=dict(family=FONT_FAMILY, size=9, color=TEXT_COLOR),
        )

        layout_updates = dict(
            title=dict(
                text=f"{symbol} · {timeframe}",
                font=dict(family=FONT_FAMILY, size=13, color=TEXT_COLOR),
                x=0.01,
            ),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=PANEL_COLOR,
            font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=TEXT_COLOR)),
            margin=dict(l=55, r=20, t=50, b=30),
            hovermode="x unified",
        )

        # Apply axes for each row
        for i in range(1, rows + 1):
            suffix = "" if i == 1 else str(i)
            fig.update_layout(**{
                f"xaxis{suffix}": {
                    **shared_axis,
                    "rangeslider": dict(visible=False),
                    "type": "date",
                },
                f"yaxis{suffix}": {
                    **shared_axis,
                    "side": "right",
                },
            })

        fig.update_layout(**layout_updates)

    @staticmethod
    def _empty_figure(symbol: str, timeframe: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=BG_COLOR, plot_bgcolor=PANEL_COLOR,
            font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
            annotations=[dict(
                text="No data loaded",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=TEXT_COLOR),
            )],
            margin=dict(l=55, r=20, t=40, b=30),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        return fig
