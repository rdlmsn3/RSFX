"""
RSFX Replay Platform — Streamlit App

Unified engine: SignalEngine + TradeEngine (same as CLI and Backtest Web UI).

Usage:
    streamlit run app.py

Architecture
------------
1. Load CSV → MarketDataStore → CandleArrays
2. Create SignalEngine with selected strategies + TradeEngine with config
3. Full backtest run: for each bar, evaluate signals → open trades → update equity
4. Playback loop: advance current_index per rerun, render charts / stats / equity
"""
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.data_loader import HistDataAdapter, get_adapter
from core.market_data_store import MarketDataStore
from core.trade_engine import TradeConfig, TradeEngine, TradeRecord, OpenPosition
from core.signal_engine import SignalEngine
from core.events import BarEvent, SignalEvent
from core.engine import CandleArrays
from detectors.strategies import (
    STRATEGY_REGISTRY,
    CATEGORY_ORDER,
    CATEGORY_DESCRIPTIONS,
    get_strategy_class,
)
from detectors.signal import PatternSignal
from views.chart_renderer import ChartRenderer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forex Replay Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global dark theme */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 8px 12px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.7rem; }
    [data-testid="stMetricValue"] { color: #c9d1d9 !important; font-size: 1.1rem; font-family: monospace; }

    /* Buttons */
    .stButton > button {
        background: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
        transition: background 0.15s;
    }
    .stButton > button:hover { background: #30363d; border-color: #58a6ff; color: #58a6ff; }

    /* Select boxes and inputs */
    .stSelectbox > div > div, .stTextInput > div > div {
        background: #0d1117 !important;
        border-color: #30363d !important;
        color: #c9d1d9 !important;
        font-family: monospace !important;
    }

    /* Slider */
    .stSlider > div { background: transparent; }

    /* Status bar */
    .status-bar {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 6px 14px;
        font-family: monospace;
        font-size: 0.75rem;
        color: #8b949e;
    }

    /* Playback indicator */
    .playing-badge {
        display: inline-block;
        background: #3fb95022;
        color: #3fb950;
        border: 1px solid #3fb95066;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-family: monospace;
    }
    .paused-badge {
        display: inline-block;
        background: #f8514922;
        color: #f85149;
        border: 1px solid #f8514966;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-family: monospace;
    }

    /* Dividers */
    hr { border-color: #21262d; }
    h1, h2, h3 { color: #e6edf3 !important; font-family: monospace; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CSV = str(Path(__file__).parent / "data" / "DAT_ASCII_USDJPY_M1_202605.csv")
TIMEFRAMES = ["M1", "M5", "H1", "D1"]
SPEEDS = {"0.5×": 2.0, "1×": 1.0, "2×": 0.5, "5×": 0.2, "10×": 0.1, "20×": 0.05, "50×": 0.02, "MAX": 0.0}
CHART_LOOKBACK = 100


# ===========================================================================
# Session state bootstrap
# ===========================================================================

def _init_session() -> None:
    """Initialise st.session_state on first load."""
    defaults: dict = {
        "data_loaded": False,
        "csv_path": DEFAULT_CSV,
        "symbol": "EURUSD",
        "timeframe": "M1",
        "speed_label": "1×",
        # Engine components (created after CSV load + backtest)
        "store": None,
        "arrays": None,
        "tf_arrays": {},
        "signal_engine": None,
        "trade_engine": None,
        # Backtest results (populated after full run)
        "signals_timeline": [],   # list of (bar_idx, SignalEvent)
        "trades_completed": [],   # list of TradeRecord (after force-close)
        # Playback state
        "current_index": 0,
        "is_playing": False,
        "max_index": 0,
        # Strategy selection
        "strategy_mode": "Single",
        "active_strategy": None,
        "active_bucket": None,
        "selected_strategies": [],
        # Engine params
        "lookback": 5,
        "threshold": 2,
        "spread_pips": 0.5,
        "min_rr": 1.0,
        "pip_value": 0.01,
        "lot_size": 0.01,
        "balance": 10000.0,
        # Force-close record for the final bar
        "final_close_trade": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ===========================================================================
# Data loading
# ===========================================================================

def _load_data(csv_path: str, symbol: str) -> bool:
    """Load CSV into MarketDataStore. Returns True on success."""
    with st.spinner(f"Loading {csv_path} …"):
        try:
            adapter = get_adapter(csv_path)
            m1_df = adapter.load(csv_path)

            store = MarketDataStore()
            store.load_symbol(symbol, m1_df)

            st.session_state.store = store
            st.session_state.symbol = symbol
            st.session_state.data_loaded = True

            logger.info("Data loaded: %s, %d M1 candles.", symbol, store.length(symbol))
            return True

        except Exception as exc:
            st.error(f"❌ Failed to load data: {exc}")
            logger.exception("Data load error: %s", exc)
            return False


# ===========================================================================
# Full backtest execution
# ===========================================================================

def _run_backtest() -> bool:
    """
    Run full backtest with SignalEngine + TradeEngine.
    Stores all results in session_state.
    Returns True on success.
    """
    store = st.session_state.store
    symbol = st.session_state.symbol
    strategy_names = st.session_state.selected_strategies
    lookback = st.session_state.lookback
    threshold = st.session_state.threshold
    spread_pips = st.session_state.spread_pips
    min_rr = st.session_state.min_rr
    pip_value = st.session_state.pip_value
    lot_size = st.session_state.lot_size
    balance = st.session_state.balance

    if not strategy_names:
        st.warning("Select at least one strategy.")
        return False

    with st.spinner("Running backtest …"):
        try:
            # Build CandleArrays
            m1_df = store.get_data(symbol, "M1")
            arrays = CandleArrays.from_dataframe(m1_df)

            tf_arrays: dict[str, CandleArrays] = {}
            for tf in store.available_timeframes(symbol):
                if tf == "M1":
                    continue
                try:
                    tf_arrays[tf] = CandleArrays.from_dataframe(store.get_data(symbol, tf))
                except Exception:
                    pass

            # Create engines
            config = TradeConfig(
                symbol=symbol,
                pip_value=pip_value,
                lot_size=lot_size,
                initial_balance=balance,
                spread_pips=spread_pips,
                min_rr=min_rr,
            )
            signal_engine = SignalEngine(
                strategy_names=strategy_names,
                lookback=lookback,
                threshold=threshold,
            )
            trade_engine = TradeEngine(config)

            # Pre-compute indicators
            signal_engine.precompute(arrays, tf_arrays)

            # Run backtest loop
            max_start = max(lookback, 100)
            signals_timeline: list[tuple[int, SignalEvent]] = []

            for i in range(max_start, arrays.n):
                signals = signal_engine.evaluate(i, arrays, tf_arrays)
                for sig in signals:
                    trade_engine.open(sig)
                    signals_timeline.append((i, sig))

                bar_event = BarEvent(
                    timestamp=arrays.timestamps[i],
                    open=float(arrays.opens[i]),
                    high=float(arrays.highs[i]),
                    low=float(arrays.lows[i]),
                    close=float(arrays.closes[i]),
                    volume=float(arrays.volumes[i]),
                    symbol=symbol,
                )
                trade_engine.on_bar(bar_event)

            # Force close any open position at end
            final_close_trade = None
            if trade_engine.open_position:
                final_close_trade = trade_engine.force_close(
                    float(arrays.closes[arrays.n - 1]),
                    pd.Timestamp(arrays.timestamps[arrays.n - 1]),
                    "EOD",
                )

            # Store results
            st.session_state.arrays = arrays
            st.session_state.tf_arrays = tf_arrays
            st.session_state.signal_engine = signal_engine
            st.session_state.trade_engine = trade_engine
            st.session_state.signals_timeline = signals_timeline
            st.session_state.trades_completed = list(trade_engine.trades)
            st.session_state.current_index = max_start
            st.session_state.max_index = arrays.n - 1
            st.session_state.is_playing = False
            st.session_state.final_close_trade = final_close_trade

            logger.info(
                "Backtest complete: %d candles, %d signals, %d trades.",
                arrays.n, len(signals_timeline), len(trade_engine.trades),
            )
            return True

        except Exception as exc:
            st.error(f"❌ Backtest failed: {exc}")
            logger.exception("Backtest error: %s", exc)
            return False


# ===========================================================================
# Helpers
# ===========================================================================

def _signal_to_pattern(sig: SignalEvent) -> PatternSignal:
    """Convert a SignalEvent to a PatternSignal for chart rendering."""
    is_confluence = "+" in sig.strategy_name
    conf_count = sig.strategy_name.count("+") + 1
    return PatternSignal(
        name=sig.strategy_name,
        start_time=sig.timestamp,
        end_time=sig.timestamp,
        confidence=sig.confidence,
        metadata={
            "direction": sig.direction,
            "entry_price": sig.entry_price,
            "take_profit": sig.take_profit,
            "stop_loss": sig.stop_loss,
            "strategy": sig.strategy_name,
            "confluence": is_confluence,
            "confluence_count": conf_count,
            "atr": 0.0,
        },
    )


def _get_pip_value(symbol: str) -> float:
    """Return pip value for a symbol (auto-detect JPY pairs)."""
    if "JPY" in symbol.upper():
        return 0.01
    return 0.0001


# ===========================================================================
# Sidebar
# ===========================================================================

def _render_sidebar() -> None:
    """Render all sidebar controls and handle button events."""
    with st.sidebar:
        st.markdown("## 📈 Replay Platform")
        st.markdown("---")

        # ---- Data source --------------------------------------------------
        st.markdown("### Data Source")
        csv_path = st.text_input(
            "CSV File Path",
            value=st.session_state.csv_path,
            help="Absolute or relative path to a HistData.com M1 CSV file.",
        )
        symbol_input = st.text_input("Symbol", value=st.session_state.symbol)

        if st.button("⬆ Load / Reload Data", use_container_width=True):
            st.session_state.csv_path = csv_path
            st.session_state.symbol = symbol_input
            st.session_state.pip_value = _get_pip_value(symbol_input)
            if _load_data(csv_path, symbol_input):
                # Auto-run backtest if strategies are selected
                if st.session_state.selected_strategies:
                    _run_backtest()
            st.rerun()

        st.markdown("---")

        if not st.session_state.data_loaded:
            st.info("Load a CSV file to begin.")
            return

        # ---- Strategy Mode ------------------------------------------------
        st.markdown("### Strategy")
        strategy_mode = st.radio(
            "Mode",
            options=["Single", "Bucket"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        if strategy_mode == "Single":
            # ---- Single Strategy Selection --------------------------------
            available_categories = [
                c for c in CATEGORY_ORDER
                if c not in ("Single TF - Existing", "Two TF - Existing")
            ]

            selected_category = st.selectbox(
                "Category",
                options=available_categories,
                index=0,
                label_visibility="collapsed",
            )

            category_strategies = [
                name for name, info in STRATEGY_REGISTRY.items()
                if info["category"] == selected_category
            ]

            if category_strategies:
                selected_strategy = st.selectbox(
                    "Strategy",
                    options=category_strategies,
                    index=0,
                    label_visibility="collapsed",
                )

                st.caption(STRATEGY_REGISTRY[selected_strategy]["description"])

                # Update selected strategies
                new_names = [selected_strategy]
                if new_names != st.session_state.selected_strategies:
                    st.session_state.selected_strategies = new_names
                    st.session_state.active_strategy = selected_strategy
                    if st.session_state.data_loaded:
                        _run_backtest()
                        st.rerun()

        else:
            # ---- Bucket Mode -----------------------------------------------
            from backtest.buckets import StrategyBucket

            buckets = StrategyBucket.list_buckets()
            bucket_names = [b["name"] for b in buckets]

            if not bucket_names:
                st.info("No buckets found. Create one from the backtest:")
                st.code("python3 -m backtest confluence -s X,Y,Z --save-bucket 'My Bucket'")
            else:
                selected_bucket_name = st.selectbox(
                    "Bucket",
                    options=bucket_names,
                    index=0,
                    label_visibility="collapsed",
                )

                # Find the selected bucket info
                bucket_info = next(b for b in buckets if b["name"] == selected_bucket_name)

                # Load full bucket
                bucket = StrategyBucket.load(Path(bucket_info["path"]))

                # Display bucket config
                st.markdown(f"**{bucket.name}**")
                st.caption(bucket.description or f"{len(bucket.strategies)} strategies")

                # Config display
                config_cols = st.columns(3)
                config_cols[0].metric("Strategies", len(bucket.strategies))
                config_cols[1].metric("Threshold", f"{bucket.threshold}-of-{len(bucket.strategies)}")
                config_cols[2].metric("S/R", "ON" if bucket.use_sr else "OFF")

                # Strategy list
                with st.expander("Strategies in bucket"):
                    for s in bucket.strategies:
                        info = STRATEGY_REGISTRY.get(s, {})
                        cat = info.get("category", "?")
                        st.markdown(f"- `{s}` ({cat})")

                # Backtest results if available
                if bucket.backtest_result:
                    with st.expander("Backtest Results"):
                        br = bucket.backtest_result
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Trades", br.get("total_trades", 0))
                        r2.metric("Win Rate", f"{br.get('win_rate', 0):.1f}%")
                        r3.metric("PnL", f"{br.get('total_pnl_pips', 0):+.1f}")
                        r4.metric("PF", f"{br.get('profit_factor', 0):.2f}")

                # Apply bucket
                if st.session_state.active_bucket != bucket.name:
                    st.session_state.selected_strategies = list(bucket.strategies)
                    st.session_state.threshold = bucket.threshold
                    st.session_state.active_bucket = bucket.name
                    st.session_state.active_strategy = None
                    if st.session_state.data_loaded:
                        _run_backtest()
                        st.rerun()

        st.markdown("---")

        # ---- Engine Parameters --------------------------------------------
        st.markdown("### Parameters")

        lookback = st.slider(
            "Lookback",
            min_value=1,
            max_value=50,
            value=st.session_state.lookback,
            help="Confluence lookback window (bars)",
        )
        threshold = st.slider(
            "Threshold",
            min_value=1,
            max_value=10,
            value=st.session_state.threshold,
            help="Min strategies agreeing for confluence",
        )
        spread_pips = st.slider(
            "Spread (pips)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.spread_pips,
            step=0.1,
            help="Round-trip spread cost in pips",
        )
        min_rr = st.slider(
            "Min R:R",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.min_rr,
            step=0.1,
            help="Minimum risk:reward ratio",
        )

        # Detect param changes and re-run
        params_changed = (
            lookback != st.session_state.lookback
            or threshold != st.session_state.threshold
            or spread_pips != st.session_state.spread_pips
            or min_rr != st.session_state.min_rr
        )
        if params_changed:
            st.session_state.lookback = lookback
            st.session_state.threshold = threshold
            st.session_state.spread_pips = spread_pips
            st.session_state.min_rr = min_rr
            if st.session_state.data_loaded and st.session_state.selected_strategies:
                _run_backtest()
                st.rerun()

        st.markdown("---")

        # ---- Timeframe ----------------------------------------------------
        st.markdown("### Chart Timeframe")
        tf = st.selectbox(
            "Timeframe",
            options=TIMEFRAMES,
            index=TIMEFRAMES.index(st.session_state.timeframe),
            label_visibility="collapsed",
        )
        if tf != st.session_state.timeframe:
            st.session_state.timeframe = tf
            st.rerun()

        # ---- Playback speed -----------------------------------------------
        st.markdown("### Playback Speed")
        speed_label = st.select_slider(
            "Speed",
            options=list(SPEEDS.keys()),
            value=st.session_state.speed_label,
            label_visibility="collapsed",
        )
        st.session_state.speed_label = speed_label

        st.markdown("---")

        # ---- Transport controls ------------------------------------------
        st.markdown("### Controls")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶ Play", use_container_width=True,
                         disabled=st.session_state.is_playing):
                st.session_state.is_playing = True
                st.rerun()

        with col2:
            if st.button("⏸ Pause", use_container_width=True,
                         disabled=not st.session_state.is_playing):
                st.session_state.is_playing = False
                st.rerun()

        with col3:
            if st.button("⏮ Reset", use_container_width=True):
                st.session_state.is_playing = False
                if st.session_state.arrays is not None:
                    max_start = max(st.session_state.lookback, 100)
                    st.session_state.current_index = max_start
                st.rerun()

        col4, col5 = st.columns(2)
        with col4:
            if st.button("⏪ Prev", use_container_width=True):
                st.session_state.is_playing = False
                st.session_state.current_index = max(
                    max(st.session_state.lookback, 100),
                    st.session_state.current_index - 1,
                )
                st.rerun()

        with col5:
            if st.button("⏩ Next", use_container_width=True):
                st.session_state.is_playing = False
                st.session_state.current_index = min(
                    st.session_state.max_index,
                    st.session_state.current_index + 1,
                )
                st.rerun()

        st.markdown("---")

        # ---- Seek slider -------------------------------------------------
        if st.session_state.arrays is not None:
            st.markdown("### Seek")
            max_start = max(st.session_state.lookback, 100)
            seek_idx = st.slider(
                "Candle index",
                min_value=max_start,
                max_value=st.session_state.max_index,
                value=st.session_state.current_index,
                label_visibility="collapsed",
            )
            if seek_idx != st.session_state.current_index:
                st.session_state.is_playing = False
                st.session_state.current_index = seek_idx
                st.rerun()


# ===========================================================================
# Status metrics panel
# ===========================================================================

def _render_status() -> None:
    """Render the status bar with playback indicator, strategy info, and metrics."""
    arrays = st.session_state.arrays
    if arrays is None:
        return

    sym = st.session_state.symbol
    tf = st.session_state.timeframe
    current_idx = st.session_state.current_index
    max_idx = st.session_state.max_index

    ts = pd.Timestamp(arrays.timestamps[current_idx]) if current_idx < arrays.n else None

    # Playback badge
    badge = (
        '<span class="playing-badge">● LIVE</span>'
        if st.session_state.is_playing else
        '<span class="paused-badge">■ PAUSED</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)

    # Strategy info row
    strategy_names = st.session_state.selected_strategies
    n_strats = len(strategy_names)
    if n_strats > 1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", f"Confluence ({n_strats} strategies)")
        c2.metric("Threshold", f"Any {st.session_state.threshold}+ agree")
        c3.metric("Lookback", f"{st.session_state.lookback} bars")
    elif n_strats == 1:
        name = strategy_names[0]
        info = STRATEGY_REGISTRY.get(name, {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy", name.replace("_", " ").title())
        c2.metric("Timeframes", ", ".join(info.get("timeframes", ["M1"])))
        c3.metric("Category", info.get("category", "—"))

    # Metrics row
    te = st.session_state.trade_engine
    n_signals = len(st.session_state.signals_timeline)
    n_trades = len(te.trades) if te else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Symbol", sym)
    c2.metric("Timeframe", tf)
    c3.metric("Timestamp", str(ts)[:16] if ts else "—")
    c4.metric("Candle #", f"{current_idx:,} / {max_idx:,}")
    c5.metric("Signals", n_signals)
    c6.metric("Trades", n_trades)


# ===========================================================================
# Chart
# ===========================================================================

def _render_chart() -> None:
    """Render candlestick charts with indicators and signal markers."""
    arrays = st.session_state.arrays
    if arrays is None:
        st.info("No data to display.")
        return

    store = st.session_state.store
    sym = st.session_state.symbol
    current_idx = st.session_state.current_index

    ts = pd.Timestamp(arrays.timestamps[current_idx])

    # Get signals up to current index
    signals_up_to = [
        sig for idx, sig in st.session_state.signals_timeline
        if idx <= current_idx
    ]
    patterns = [_signal_to_pattern(sig) for sig in signals_up_to]

    # Get trades up to current timestamp (for trade markers)
    te = st.session_state.trade_engine
    trades_up_to = [
        t for t in te.trades
        if t.entry_time <= ts
    ]

    # Render each timeframe as a separate chart
    for tf_name in ["M1", "M5", "H1"]:
        try:
            window = store.get_window(sym, tf_name, ts, lookback=CHART_LOOKBACK)
        except KeyError:
            continue
        if window.empty:
            continue

        # Only show signal markers on M1 (entry timeframe)
        patterns_for_tf = patterns if tf_name == "M1" else None
        trades_for_tf = trades_up_to if tf_name == "M1" else None

        fig = ChartRenderer.render_chart(
            market_data=window,
            patterns=patterns_for_tf,
            trades=trades_for_tf,
            symbol=sym,
            timeframe=tf_name,
            max_candles=CHART_LOOKBACK,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ===========================================================================
# Equity curve
# ===========================================================================

def _render_equity_curve() -> None:
    """Render equity curve chart (balance over time)."""
    te = st.session_state.trade_engine
    if te is None:
        return

    balance_curve = te.balance_curve
    if len(balance_curve) <= 1:
        return

    current_idx = st.session_state.current_index
    max_start = max(st.session_state.lookback, 100)

    # Map balance curve indices to candle indices
    # balance_curve[0] = initial balance (before any bar)
    # balance_curve[k] = balance after bar (max_start + k - 1)
    # Show up to current bar
    bars_processed = current_idx - max_start + 1
    if bars_processed <= 0:
        return

    visible_curve = balance_curve[:bars_processed + 1]
    x_indices = list(range(max_start - 1, max_start - 1 + len(visible_curve)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_indices,
        y=visible_curve,
        mode="lines",
        name="Balance",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ))

    # Add initial balance reference line
    fig.add_hline(
        y=st.session_state.balance,
        line=dict(color="#8b949e", width=1, dash="dot"),
        annotation_text=f"Initial: ${st.session_state.balance:,.0f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text="Equity Curve",
            font=dict(family="JetBrains Mono, monospace", size=13, color="#c9d1d9"),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="JetBrains Mono, monospace", color="#c9d1d9"),
        xaxis=dict(
            title="Candle Index",
            gridcolor="#21262d",
            showgrid=True,
        ),
        yaxis=dict(
            title="Balance ($)",
            gridcolor="#21262d",
            showgrid=True,
            tickprefix="$",
        ),
        height=250,
        margin=dict(l=60, r=20, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# ===========================================================================
# Trade history
# ===========================================================================

def _render_trade_history() -> None:
    """Render trade history table with live PnL for open positions."""
    te = st.session_state.trade_engine
    if te is None:
        return

    current_idx = st.session_state.current_index
    ts = pd.Timestamp(st.session_state.arrays.timestamps[current_idx])

    # Completed trades up to current time
    completed = [t for t in te.trades if t.exit_time and t.exit_time <= ts]

    # Check if there's an open position
    open_pos = te.open_position
    has_open = open_pos is not None and open_pos.entry_time <= ts

    if not completed and not has_open:
        return

    st.markdown("### Trade History")

    orders_data = []

    # Open position
    if has_open:
        # Compute live PnL
        arrays = st.session_state.arrays
        current_price = float(arrays.closes[current_idx])
        if open_pos.direction == "LONG":
            live_pnl = (current_price - open_pos.entry_price) / st.session_state.pip_value
        else:
            live_pnl = (open_pos.entry_price - current_price) / st.session_state.pip_value
        live_pnl -= st.session_state.spread_pips

        orders_data.append({
            "Time": str(open_pos.entry_time)[:16],
            "Dir": open_pos.direction,
            "Entry": f"{open_pos.entry_price:.5f}",
            "TP": f"{open_pos.take_profit:.5f}",
            "SL": f"{open_pos.stop_loss:.5f}",
            "Status": "OPEN",
            "PnL (pips)": f"{live_pnl:+.1f}",
            "Strategy": open_pos.strategy,
        })

    # Completed trades
    for t in completed:
        orders_data.append({
            "Time": str(t.entry_time)[:16],
            "Dir": t.direction,
            "Entry": f"{t.entry_price:.5f}",
            "TP": f"{t.take_profit:.5f}",
            "SL": f"{t.stop_loss:.5f}",
            "Status": t.exit_reason or "CLOSED",
            "PnL (pips)": f"{t.pnl_pips:+.1f}",
            "Strategy": t.strategy,
        })

    orders_df = pd.DataFrame(orders_data)

    # Color code rows
    def highlight_orders(row):
        if row.get("Status") == "OPEN":
            return ["background-color: #3fb95011"] * len(row)
        try:
            pnl = float(row.get("PnL (pips)", "0").replace("+", ""))
        except (ValueError, TypeError):
            pnl = 0
        if pnl > 0:
            return ["background-color: #3fb95015"] * len(row)
        elif pnl < 0:
            return ["background-color: #f8514915"] * len(row)
        return [""] * len(row)

    st.dataframe(
        orders_df.style.apply(highlight_orders, axis=1),
        use_container_width=True,
        height=min(400, 35 + len(orders_df) * 35),
    )

    # Summary metrics
    open_count = 1 if has_open else 0
    closed_count = len(completed)
    winning = len([t for t in completed if t.pnl_pips > 0])
    total_pnl = sum(t.pnl_pips for t in completed)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open", open_count)
    c2.metric("Closed", closed_count)
    c3.metric("Win Rate", f"{winning / closed_count * 100:.0f}%" if closed_count else "—")
    c4.metric("Total PnL", f"{total_pnl:+.1f} pips")


# ===========================================================================
# Stats summary
# ===========================================================================

def _render_stats() -> None:
    """Render stats summary (win rate, PnL, expectancy, PF, drawdown)."""
    te = st.session_state.trade_engine
    if te is None:
        return

    stats = te.get_stats()
    total = stats.get("total_trades", 0)
    if total == 0:
        return

    st.markdown("### Performance Stats")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades", total)
    c2.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    c3.metric("Total PnL", f"{stats.get('total_pnl_pips', 0):+.1f} pips")
    c4.metric("Expectancy", f"{stats.get('expectancy_pips', 0):+.1f} pips")
    c5.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
    c6.metric("Max Drawdown", f"{stats.get('max_drawdown_pct', 0):.1f}%")

    # Second row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg MAE", f"{stats.get('avg_mae_pips', 0):+.1f} pips")
    c2.metric("Avg MFE", f"{stats.get('avg_mfe_pips', 0):+.1f} pips")
    c3.metric("Gross Profit", f"{stats.get('gross_profit', 0):+.1f} pips")
    c4.metric("Gross Loss", f"{stats.get('gross_loss', 0):+.1f} pips")


# ===========================================================================
# Main loop
# ===========================================================================

def main() -> None:
    _init_session()

    # ---- Sidebar (renders and handles events) ----------------------------
    _render_sidebar()

    # ---- Header ----------------------------------------------------------
    st.markdown("# Forex Market Replay Platform")

    if not st.session_state.data_loaded:
        st.markdown(
            '<div class="status-bar">Load a HistData.com CSV file from the sidebar to begin replay.</div>',
            unsafe_allow_html=True,
        )
        return

    if st.session_state.arrays is None:
        if st.session_state.selected_strategies:
            st.markdown(
                '<div class="status-bar">Strategies selected — running backtest…</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-bar">Select a strategy from the sidebar to begin.</div>',
                unsafe_allow_html=True,
            )
        return

    # ---- Status bar -------------------------------------------------------
    _render_status()

    # ---- Chart ------------------------------------------------------------
    _render_chart()

    # ---- Equity curve -----------------------------------------------------
    _render_equity_curve()

    # ---- Trade history ----------------------------------------------------
    _render_trade_history()

    # ---- Stats summary ----------------------------------------------------
    _render_stats()

    # ---- Playback loop: advance one index per rerun while playing ---------
    if st.session_state.is_playing:
        delay = SPEEDS[st.session_state.speed_label]
        if delay > 0:
            time.sleep(delay)

        if st.session_state.current_index < st.session_state.max_index:
            st.session_state.current_index += 1
        else:
            st.session_state.is_playing = False

        st.rerun()


if __name__ == "__main__":
    main()
