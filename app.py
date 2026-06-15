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
SPEEDS        = {"0.5×": 2.0, "1×": 1.0, "2×": 0.5, "5×": 0.2, "10×": 0.1, "MAX": 0.0}
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
            det     = PatternDetector(bus, store, symbol=symbol)
            te      = TradeEngine(bus, symbol=symbol)

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

    # Metrics row
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Symbol",    sym)
    c2.metric("Timeframe", tf)
    c3.metric("Timestamp", str(ts)[:16] if ts else "—")
    c4.metric("Candle #",  f"{ctrl.current_index + 1:,} / {total:,}")
    c5.metric("Remaining", f"{ctrl.bars_remaining:,}")
    c6.metric("Patterns",  det.signal_count)
    c7.metric("Trades",    te.trade_count)


# ===========================================================================
# Chart
# ===========================================================================

def _render_chart() -> None:
    ctrl:  PlaybackController = st.session_state.controller
    store: MarketDataStore    = st.session_state.store
    det:   PatternDetector    = st.session_state.detector
    te:    TradeEngine        = st.session_state.trade_engine
    sym    = st.session_state.symbol
    tf     = st.session_state.timeframe

    ts = ctrl.current_timestamp
    if ts is None:
        st.info("No data to display.")
        return

    # Fetch display window through the store (no copying full DataFrame)
    window = store.get_window(sym, tf, ts, lookback=CHART_LOOKBACK)

    fig = ChartRenderer.render_chart(
        market_data=window,
        patterns=det.signals,
        trades=te.open_positions + te.closed_positions,
        symbol=sym,
        timeframe=tf,
        max_candles=CHART_LOOKBACK,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


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