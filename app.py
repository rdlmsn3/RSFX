"""
app.py
------
Streamlit View layer for the Event-Driven Forex Market Replay Platform.

Architecture
------------
This file is ONLY responsible for:
  1. Rendering UI elements (sidebar, chart, status bar)
  2. Translating user interactions into controller method calls
  3. Storing/restoring UI state via st.session_state

All business logic lives in core/ and detectors/.
No DataFrame slicing, resampling, or pattern logic belongs here.

Playback loop (per Streamlit rerun cycle)
-----------------------------------------
  PlaybackController.tick()
       │
       └─► MarketTickEvent published on EventBus
                 │
                 ├─► PatternDetector._on_market_tick()   → scan window
                 └─► TradeEngine._on_market_tick()       → mark-to-market
       │
  UI reads current_index from controller → fetches display window → renders
"""

from __future__ import annotations
import time
import logging
from pathlib import Path

import streamlit as st
import pandas as pd

from core.data_loader import HistDataAdapter
from core.market_data_store import MarketDataStore
from core.event_bus import EventBus
from core.playback_controller import PlaybackController
from detectors.pattern_detector import PatternDetector
from detectors.strategies import (
    EMAStochasticMTFStrategy,
    STRATEGY_REGISTRY,
    CATEGORY_ORDER,
    CATEGORY_DESCRIPTIONS,
    get_strategy_class,
)
from core.trade_engine import TradeEngine
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
DEFAULT_CSV   = str(Path(__file__).parent / "data" / "DAT_ASCII_USDJPY_M1_202605.csv")
TIMEFRAMES    = ["M1", "M5", "H1", "D1"]
SPEEDS        = {"0.5×": 2.0, "1×": 1.0, "2×": 0.5, "5×": 0.2, "10×": 0.1, "20×": 0.05, "50×": 0.02, "MAX": 0.0}
CHART_LOOKBACK = 100    # candles rendered at once


# ===========================================================================
# Session state bootstrap
# ===========================================================================

def _init_session() -> None:
    """Initialise st.session_state on first load."""
    defaults: dict = {
        "data_loaded":    False,
        "csv_path":       DEFAULT_CSV,
        "symbol":         "EURUSD",
        "timeframe":      "M1",
        "speed_label":    "1×",
        # Component references (created after CSV load)
        "store":          None,
        "bus":            None,
        "controller":     None,
        "detector":       None,
        "trade_engine":   None,
        # Last tick cache
        "last_index":     -1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_data(csv_path: str, symbol: str) -> bool:
    """
    Load CSV, populate MarketDataStore, wire up EventBus and all components.
    Returns True on success.
    """
    with st.spinner(f"Loading {csv_path} …"):
        try:
            adapter = HistDataAdapter()
            m1_df   = adapter.load(csv_path)

            store   = MarketDataStore()
            store.load_symbol(symbol, m1_df)

            bus     = EventBus()
            ctrl    = PlaybackController(bus, store, symbol=symbol)
            strategy = EMAStochasticMTFStrategy()
            det     = PatternDetector(bus, store, symbol=symbol, strategy=strategy)
            te      = TradeEngine(bus, symbol=symbol, data_store=store)

            # Emit the first tick so the chart has data immediately
            ctrl.reset()

            st.session_state.store        = store
            st.session_state.bus          = bus
            st.session_state.controller   = ctrl
            st.session_state.detector     = det
            st.session_state.trade_engine = te
            st.session_state.symbol       = symbol
            st.session_state.data_loaded  = True
            st.session_state.last_index   = -1

            logger.info("Data loaded: %s, %d M1 candles.", symbol, store.length(symbol))
            return True

        except Exception as exc:
            st.error(f"❌ Failed to load data: {exc}")
            logger.exception("Data load error: %s", exc)
            return False


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
            st.session_state.symbol   = symbol_input
            _load_data(csv_path, symbol_input)
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
            available_categories = [c for c in CATEGORY_ORDER
                                    if c not in ("Single TF - Existing", "Two TF - Existing")]

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

                st.markdown("#### Parameters")
                params = {}
                for param_name, param_info in STRATEGY_REGISTRY[selected_strategy]["params"].items():
                    if param_info["type"] == "int":
                        params[param_name] = st.slider(
                            param_name,
                            min_value=param_info["min"],
                            max_value=param_info["max"],
                            value=param_info["default"],
                        )
                    elif param_info["type"] == "float":
                        params[param_name] = st.slider(
                            param_name,
                            min_value=float(param_info["min"]),
                            max_value=float(param_info["max"]),
                            value=float(param_info["default"]),
                            step=0.1,
                        )

                # Auto-apply strategy on selection change
                if selected_strategy != st.session_state.get("active_strategy"):
                    strategy_class = get_strategy_class(selected_strategy)
                    new_strategy = strategy_class(**params)
                    st.session_state.detector.strategy = new_strategy
                    st.session_state.active_strategy = selected_strategy
                    st.rerun()

                # Auto-apply on param change
                current_params = st.session_state.get("strategy_params", {})
                if params != current_params:
                    strategy_class = get_strategy_class(selected_strategy)
                    new_strategy = strategy_class(**params)
                    st.session_state.detector.strategy = new_strategy
                    st.session_state.strategy_params = params
                    st.rerun()

        else:
            # ---- Bucket Mode -----------------------------------------------
            from backtest.buckets import StrategyBucket, BUCKETS_DIR

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
                if (st.session_state.get("active_bucket") != bucket.name or
                        st.session_state.get("active_bucket_use_sr") != bucket.use_sr):
                    # Build strategy instances
                    strategy_instances = []
                    for s_name in bucket.strategies:
                        if s_name in STRATEGY_REGISTRY:
                            cls = STRATEGY_REGISTRY[s_name]["class"]
                            strategy_instances.append(cls())

                    st.session_state.detector.strategies = strategy_instances
                    st.session_state.detector.confluence_threshold = bucket.threshold
                    st.session_state.active_bucket = bucket.name
                    st.session_state.active_bucket_use_sr = bucket.use_sr
                    st.session_state.active_strategy = None
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

        ctrl: PlaybackController = st.session_state.controller

        with col1:
            if st.button("▶ Play", use_container_width=True, disabled=ctrl.is_playing):
                ctrl.play()
                st.rerun()

        with col2:
            if st.button("⏸ Pause", use_container_width=True, disabled=not ctrl.is_playing):
                ctrl.pause()
                st.rerun()

        with col3:
            if st.button("⏮ Reset", use_container_width=True):
                ctrl.reset()
                st.session_state.detector.reset()
                st.session_state.trade_engine.reset()
                st.rerun()

        col4, col5 = st.columns(2)
        with col4:
            if st.button("⏪ Prev", use_container_width=True):
                ctrl.pause()
                ctrl.step_backward()
                st.rerun()

        with col5:
            if st.button("⏩ Next", use_container_width=True):
                ctrl.pause()
                ctrl.step_forward()
                st.rerun()

        st.markdown("---")

        # ---- Seek slider -------------------------------------------------
        st.markdown("### Seek")
        store: MarketDataStore = st.session_state.store
        total = store.length(st.session_state.symbol, "M1")
        seek_idx = st.slider(
            "Candle index",
            min_value=0,
            max_value=max(0, total - 1),
            value=ctrl.current_index,
            label_visibility="collapsed",
        )
        if seek_idx != ctrl.current_index:
            ctrl.pause()
            ctrl.seek_to_index(seek_idx)
            st.rerun()


# ===========================================================================
# Status metrics panel
# ===========================================================================

def _render_status() -> None:
    ctrl:  PlaybackController = st.session_state.controller
    store: MarketDataStore    = st.session_state.store
    det:   PatternDetector    = st.session_state.detector
    te:    TradeEngine        = st.session_state.trade_engine
    sym    = st.session_state.symbol
    tf     = st.session_state.timeframe

    ts    = ctrl.current_timestamp
    total = store.length(sym, "M1")

    # Playback badge
    badge = (
        '<span class="playing-badge">● LIVE</span>'
        if ctrl.is_playing else
        '<span class="paused-badge">■ PAUSED</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)

    # Strategy info row
    if det.is_confluence_mode:
        strategy_names = [s.name for s in det.strategies]
        n_strats = len(strategy_names)
        active_bucket = getattr(st.session_state, 'active_bucket', '')
        c1, c2, c3 = st.columns(3)
        c1.metric("Mode", f"Bucket ({n_strats} strategies)")
        c2.metric("Bucket", active_bucket or "Custom")
        c3.metric("Confluence", f"Any {det.confluence_threshold}+ agree")
    else:
        active_strategy = getattr(st.session_state, 'active_strategy', 'ema_stochastic_mtf')
        strategy_info = STRATEGY_REGISTRY.get(active_strategy, {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy", active_strategy.replace("_", " ").title())
        c2.metric("Timeframes", ", ".join(strategy_info.get("timeframes", ["M1", "M5", "H1"])))
        c3.metric("Category", strategy_info.get("category", "Two TF - Existing"))

    # Metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Symbol",    sym)
    c2.metric("Timeframe", tf)
    c3.metric("Timestamp", str(ts)[:16] if ts else "—")
    c4.metric("Candle #",  f"{ctrl.current_index + 1:,} / {total:,}")
    c5.metric("Remaining", f"{ctrl.bars_remaining:,}")
    c6.metric("Patterns",  det.signal_count)


# ===========================================================================
# Chart
# ===========================================================================

def _render_chart() -> None:
    ctrl:  PlaybackController = st.session_state.controller
    store: MarketDataStore    = st.session_state.store
    det:   PatternDetector    = st.session_state.detector
    te:    TradeEngine        = st.session_state.trade_engine
    sym    = st.session_state.symbol

    ts = ctrl.current_timestamp
    if ts is None:
        st.info("No data to display.")
        return

    # Render each timeframe as a separate chart
    for tf in ["M1", "M5", "H1"]:
        try:
            window = store.get_window(sym, tf, ts, lookback=CHART_LOOKBACK)
        except KeyError:
            continue
        if window.empty:
            continue

        # Only show L/S markers on M1 (entry timeframe)
        patterns_for_tf = det.signals if tf == "M1" else None
        trades_for_tf = (te.open_positions + te.closed_positions) if tf == "M1" else None

        fig = ChartRenderer.render_chart(
            market_data=window,
            patterns=patterns_for_tf,
            trades=trades_for_tf,
            symbol=sym,
            timeframe=tf,
            max_candles=CHART_LOOKBACK,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_orders() -> None:
    """Render open and closed trade orders with live PnL."""
    te: TradeEngine = st.session_state.trade_engine
    store: MarketDataStore = st.session_state.store
    sym = st.session_state.symbol
    ctrl: PlaybackController = st.session_state.controller

    all_positions = te.open_positions + te.closed_positions
    if not all_positions:
        return

    st.markdown("### Orders")

    # Get current price for live PnL
    current_price = None
    try:
        ts = ctrl.current_timestamp
        if ts:
            m1 = store.get_window(sym, "M1", ts, lookback=1)
            if not m1.empty:
                current_price = float(m1["close"].iloc[-1])
    except Exception:
        pass

    orders_data = []
    for pos in all_positions:
        # Live PnL for open positions
        if pos.status == "OPEN" and current_price is not None:
            mult = 1 if pos.direction == "BUY" else -1
            live_pnl = mult * (current_price - pos.entry_price) * 100_000  # pips
        elif pos.pnl is not None:
            live_pnl = pos.pnl / (pos.lot_size * 100_000)  # convert to pips
        else:
            live_pnl = 0.0

        exit_reason = pos.metadata.get("exit_reason", "—") if pos.status == "CLOSED" else "—"
        strategy = pos.metadata.get("strategy", "—")
        conf_count = pos.metadata.get("confluence_count", 0)

        orders_data.append({
            "Time": str(pos.entry_time)[:16],
            "Dir": pos.direction,
            "Entry": f"{pos.entry_price:.5f}",
            "TP": f"{pos.take_profit:.5f}" if pos.take_profit else "—",
            "SL": f"{pos.stop_loss:.5f}" if pos.stop_loss else "—",
            "Status": pos.status,
            "Live PnL": f"{live_pnl:+.1f}",
            "Exit": exit_reason,
            "Strategy": strategy,
            "Conf": conf_count,
        })

    orders_df = pd.DataFrame(orders_data)

    # Color code rows
    def highlight_orders(row):
        if row.get("Status") == "OPEN":
            return ["background-color: #3fb95011"] * len(row)
        pnl = float(row.get("Live PnL", "0").replace("+", ""))
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
    open_count = len([p for p in all_positions if p.status == "OPEN"])
    closed_count = len([p for p in all_positions if p.status == "CLOSED"])
    winning = len([p for p in te.closed_positions if (p.pnl or 0) > 0])
    total_pnl = sum((p.pnl or 0) for p in te.closed_positions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open", open_count)
    c2.metric("Closed", closed_count)
    c3.metric("Win Rate", f"{winning/closed_count*100:.0f}%" if closed_count else "—")
    c4.metric("Total PnL", f"{total_pnl:+.1f}")


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

    # ---- Status bar -------------------------------------------------------
    _render_status()

    # ---- Chart ------------------------------------------------------------
    _render_chart()

    # ---- Order History ----------------------------------------------------
    _render_orders()

    # ---- Playback loop: advance one tick per rerun while playing ----------
    ctrl: PlaybackController = st.session_state.controller
    if ctrl.is_playing:
        delay = SPEEDS[st.session_state.speed_label]
        if delay > 0:
            time.sleep(delay)

        advanced = ctrl.tick()   # publishes MarketTickEvent

        if not advanced:
            # Reached end of data
            st.session_state.controller.pause()

        st.rerun()   # schedule next frame


if __name__ == "__main__":
    main()