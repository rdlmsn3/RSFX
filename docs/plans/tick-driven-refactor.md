# Tick-Driven System Refactor — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace bar/candle-driven clock with a fully event-driven, tick-driven system. Ticks are the only clock. Bars are derived views emitted as events. Everything communicates through EventBus.

**Architecture:**
```
                          EventBus
                             │
Tick Source ────────────► CandleStream
(file / WebSocket)        │  ingests tick
                          │  If M1/M5/H1 boundary → publish(BarEvent)
                          │
              ┌───────────┤
              │           │
         SignalEngine  TradeEngine
         (subscribed)  (subscribed to SignalEvent)
         on BarEvent        │
              │             │
              │  SignalEvent
              └──────►──────┘
                    queue_order()

Next tick ──────────► TradeEngine.on_tick()
                      fills pending, checks SL/TP
```

**Design principle:** Two call patterns coexist:
1. **Direct call** (every tick): `trade_engine.on_tick(bid, ask, ts)` — manages positions, fills pending orders
2. **EventBus** (candle close): CandleStream → BarEvent → SignalEngine → SignalEvent → TradeEngine.queue_order()

The tick loop in every UI is just two lines:
```python
for tick in tick_source:
    trade_engine.on_tick(bid, ask, ts)
    candle_stream.ingest_tick(ts, bid, ask, vol)  # EventBus handles the rest
```

**Tech Stack:** Python 3, NumPy, Pandas, existing RSFX core/

---

## Pre-Flight: Safety Branch ✅

Branch `pre-tick-driven-refactor` pushed. Current work on `tick-driven-refactor` branch.

---

## What Changes vs What Stays

### Unchanged:
- `core/engine.py` — CandleArrays, compute_tp_sl, build_result, compute_pnl, etc.
- `core/events.py` — SignalEvent, BarEvent, TradeEvent (already has what we need)
- `core/event_bus.py` — EventBus pub/sub (already has what we need)
- `core/data_loader.py` — HistDataAdapter, ParquetAdapter
- `core/trade_store.py` — SQLite persistence
- `core/tick_candle_builder.py` — batch builder (kept for ad-hoc use)
- `detectors/` — all 72 strategies (unchanged)
- `backtest/correlation.py`, `backtest/portfolio.py` — consume trades
- `backtest/__main__.py` — entry point router

### Modified:
- `core/trade_engine.py` — bar+tick dual → tick-only + EventBus subscriber for SignalEvent
- `core/signal_engine.py` — add EventBus subscriber for BarEvent (auto-evaluate on candle close)
- `ui/cli.py` — bar loop → 2-line tick loop
- `ui/backtest/server.py` — bar loop → 2-line tick loop
- `ui/streamlit_app/app.py` — bar loop → 2-line tick loop + live-ready

### New:
- `core/candle_stream.py` — IncrementalCandleBuilder + StreamingCandleArrays + BarEvent publisher

### Deprecated from core loop:
- `core/market_data_store.py` — no longer needed in tick-driven path (IncrementalCandleBuilder builds all TFs incrementally). Keep for ad-hoc analysis only.

---

## Task 1: Create `core/candle_stream.py` — EventBus-Aware

**Objective:** Incremental candle construction that publishes BarEvent through EventBus when candles close.

**Files:**
- Create: `core/candle_stream.py`

**Key difference from `new_candle_stream.py`:** The builder takes an optional `EventBus` and publishes `BarEvent` when a candle finishes. This is the bridge between tick ingestion and event-driven signal evaluation.

```python
"""
core/candle_stream.py
---------------------
Incremental, tick-driven candle construction with EventBus integration.

Ingests ticks one at a time. When a candle boundary is crossed (the
previous candle is now closed), emits a BarEvent on the EventBus.
No lookahead — a Bar is only emitted once a tick from the NEXT bucket
arrives, which is the exact moment a live feed would report the close.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.tick_candle_builder import _TF_MAP
from core.events import BarEvent

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single finished OHLCV candle."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class IncrementalCandleBuilder:
    """
    Stateful single-timeframe OHLCV aggregator driven by individual ticks.

    Call ingest_tick() once per tick. Returns a finished Bar the moment
    the timestamp crosses into a new bucket, otherwise returns None.

    If an EventBus is provided, automatically publishes BarEvent on close.
    """

    def __init__(self, timeframe: str = "M1", event_bus=None, symbol: str = "USDJPY") -> None:
        if timeframe.upper() not in _TF_MAP:
            raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(_TF_MAP.keys())}")
        self.timeframe = timeframe.upper()
        self._freq = _TF_MAP[self.timeframe]
        self._bus = event_bus
        self._symbol = symbol

        self._bucket_start: Optional[pd.Timestamp] = None
        self._o = self._h = self._l = self._c = None
        self._vol: float = 0.0

    def ingest_tick(self, ts: pd.Timestamp, bid: float, ask: float,
                    volume: float = 0.0) -> Optional[Bar]:
        ts = pd.Timestamp(ts)
        mid = (bid + ask) / 2.0
        bucket = ts.floor(self._freq)

        if self._bucket_start is None:
            self._start_bucket(bucket, mid, volume)
            return None

        if bucket == self._bucket_start:
            self._h = max(self._h, mid)
            self._l = min(self._l, mid)
            self._c = mid
            self._vol += volume
            return None

        # Bucket boundary crossed → previous candle is now closed
        finished = Bar(
            timestamp=self._bucket_start,
            open=self._o, high=self._h, low=self._l, close=self._c,
            volume=self._vol,
        )

        # Publish BarEvent on EventBus if connected
        if self._bus:
            self._bus.publish(BarEvent(
                timestamp=finished.timestamp,
                open=finished.open, high=finished.high,
                low=finished.low, close=finished.close,
                volume=finished.volume,
                symbol=self._symbol,
                timeframe=self.timeframe,
            ))

        self._start_bucket(bucket, mid, volume)
        return finished

    def flush(self) -> Optional[Bar]:
        """Call at end-of-stream to emit the final, still-open bucket."""
        if self._bucket_start is None:
            return None
        bar = Bar(
            timestamp=self._bucket_start,
            open=self._o, high=self._h, low=self._l, close=self._c,
            volume=self._vol,
        )
        self._bucket_start = None
        return bar

    def _start_bucket(self, bucket: pd.Timestamp, mid: float, volume: float) -> None:
        self._bucket_start = bucket
        self._o = self._h = self._l = self._c = mid
        self._vol = volume


class StreamingCandleArrays:
    """
    Append-only, growing replacement for a pre-built CandleArrays.

    Exposes the same attribute surface SignalEngine/strategies expect
    (.timestamps, .opens, .highs, .lows, .closes, .volumes, .n) so it
    drops straight into signal_engine.evaluate(i, arrays, tf_arrays).

    Numpy views are rebuilt lazily — only when a new bar is appended
    (once per candle close, not once per tick).
    """

    def __init__(self) -> None:
        self._ts: list[np.datetime64] = []
        self._o: list[float] = []
        self._h: list[float] = []
        self._l: list[float] = []
        self._c: list[float] = []
        self._v: list[float] = []
        self._dirty = True
        self._cache: dict[str, np.ndarray] = {}

    def append(self, bar: Bar) -> None:
        self._ts.append(np.datetime64(bar.timestamp))
        self._o.append(bar.open)
        self._h.append(bar.high)
        self._l.append(bar.low)
        self._c.append(bar.close)
        self._v.append(bar.volume)
        self._dirty = True

    def _rebuild(self) -> None:
        self._cache = {
            "timestamps": np.array(self._ts),
            "opens": np.array(self._o, dtype=float),
            "highs": np.array(self._h, dtype=float),
            "lows": np.array(self._l, dtype=float),
            "closes": np.array(self._c, dtype=float),
            "volumes": np.array(self._v, dtype=float),
        }
        self._dirty = False

    def _get(self, key: str) -> np.ndarray:
        if self._dirty:
            self._rebuild()
        return self._cache[key]

    @property
    def n(self) -> int:
        return len(self._ts)

    @property
    def timestamps(self) -> np.ndarray:
        return self._get("timestamps")

    @property
    def opens(self) -> np.ndarray:
        return self._get("opens")

    @property
    def highs(self) -> np.ndarray:
        return self._get("highs")

    @property
    def lows(self) -> np.ndarray:
        return self._get("lows")

    @property
    def closes(self) -> np.ndarray:
        return self._get("closes")

    @property
    def volumes(self) -> np.ndarray:
        return self._get("volumes")
```

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 -c "
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays
from core.event_bus import EventBus
from core.events import BarEvent
import pandas as pd

# Test 1: basic candle building
b = IncrementalCandleBuilder('M1')
bar = b.ingest_tick(pd.Timestamp('2026-01-01 10:00:00'), 1.1, 1.1002, 100)
assert bar is None
bar = b.ingest_tick(pd.Timestamp('2026-01-01 10:01:00'), 1.1005, 1.1007, 200)
assert bar is not None
assert bar.open == 1.1
print('Test 1 OK: basic candle building')

# Test 2: EventBus integration
bus = EventBus()
received = []
bus.subscribe(BarEvent, lambda e: received.append(e))
b2 = IncrementalCandleBuilder('M1', event_bus=bus, symbol='EURUSD')
b2.ingest_tick(pd.Timestamp('2026-01-01 10:00:00'), 1.1, 1.1002, 100)
assert len(received) == 0, 'No event yet'
b2.ingest_tick(pd.Timestamp('2026-01-01 10:01:00'), 1.1005, 1.1007, 200)
assert len(received) == 1, 'BarEvent should have been published'
assert received[0].timeframe == 'M1'
assert received[0].symbol == 'EURUSD'
print('Test 2 OK: EventBus integration')

# Test 3: StreamingCandleArrays
a = StreamingCandleArrays()
a.append(bar)
assert a.n == 1
assert a.opens[0] == 1.1
print('Test 3 OK: StreamingCandleArrays')
"
```

**Commit:**
```bash
git add core/candle_stream.py
git commit -m "feat: add IncrementalCandleBuilder + StreamingCandleArrays with EventBus integration"
```

---

## Task 2: Rewrite `core/trade_engine.py` — Tick-Only + EventBus Subscriber

**Objective:** Replace bar+tick dual mode with tick-only execution. Subscribe to SignalEvent via EventBus.

**Files:**
- Replace: `core/trade_engine.py`

**Key API changes:**
```
OLD                              NEW
───────────────────────────      ───────────────────────────
engine.open(signal)              engine.queue_order(signal)  ← same
engine.on_bar(bar_event)         (removed)
engine.on_tick(bid, ask, ts)     engine.on_tick(bid, ask, ts)  ← same
engine.attach_ticks(df)          (removed)
engine._real_fill_price(...)     (removed — fills at next tick)
```

**EventBus integration:** TradeEngine subscribes to `SignalEvent`. When SignalEngine publishes a SignalEvent, TradeEngine's handler calls `queue_order()` automatically.

```python
class TradeEngine:
    def __init__(self, config, event_bus=None):
        # ... same state ...
        self._bus = event_bus
        if self._bus:
            self._bus.subscribe(SignalEvent, self._on_signal_event)

    def _on_signal_event(self, signal: SignalEvent) -> None:
        """EventBus handler — called when SignalEngine publishes a signal."""
        self.queue_order(signal)
```

**What stays the same:** `TradeConfig`, `TradeRecord` (with `ticks_held` instead of `bars_held`), `PendingOrder`, `OpenPosition` (with `tick_count` instead of `bar_idx`).

**What's removed:** `on_bar()`, `attach_ticks()`, `_real_fill_price()`, `_check_exit(high, low, close, ...)`.

**What's new:** `mark_to_market(close)` for equity curve updates (no execution effect).

**Copy from `new_trade_engine.py`** with these adaptations:
- Add EventBus subscription for SignalEvent
- Keep `queue_order()` and `on_tick()` as the core API
- `TradeRecord.to_dict()`: change `bars_held` → `ticks_held`

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 -c "
from core.trade_engine import TradeEngine, TradeConfig
from core.events import SignalEvent
from core.event_bus import EventBus
import pandas as pd

# Test 1: direct queue_order
config = TradeConfig(symbol='USDJPY', pip_value=0.01, lot_size=0.01, initial_balance=10000, spread_pips=0.5)
engine = TradeEngine(config)
sig = SignalEvent(
    strategy_name='test', direction='LONG',
    entry_price=150.0, take_profit=150.50, stop_loss=149.70,
    timestamp=pd.Timestamp('2026-01-01 10:00:00'),
)
engine.queue_order(sig)
assert engine.pending_order is not None
engine.on_tick(bid=150.001, ask=150.003, timestamp=pd.Timestamp('2026-01-01 10:00:01'))
assert engine.open_position is not None
assert engine.open_position.entry_price == 150.003
print('Test 1 OK: direct queue_order + fill')

# Test 2: EventBus subscriber
bus = EventBus()
engine2 = TradeEngine(config, event_bus=bus)
bus.publish(sig)
assert engine2.pending_order is not None
engine2.on_tick(bid=150.001, ask=150.003, timestamp=pd.Timestamp('2026-01-01 10:00:01'))
assert engine2.open_position is not None
print('Test 2 OK: EventBus subscriber')
"
```

**Commit:**
```bash
git add core/trade_engine.py
git commit -m "refactor: tick-only TradeEngine with EventBus SignalEvent subscriber"
```

---

## Task 3: Update `core/signal_engine.py` — EventBus BarEvent Subscriber

**Objective:** SignalEngine subscribes to BarEvent via EventBus. When a candle closes, it auto-evaluates strategies and publishes SignalEvent.

**Files:**
- Modify: `core/signal_engine.py`

**Key change:** Add EventBus subscription so SignalEngine reacts to BarEvent automatically:

```python
class SignalEngine:
    def __init__(self, strategy_names, lookback=5, threshold=2, event_bus=None):
        # ... existing init ...
        self._bus = event_bus
        if self._bus:
            self._bus.subscribe(BarEvent, self._on_bar_event)

    def _on_bar_event(self, event: BarEvent) -> None:
        """EventBus handler — called when CandleStream publishes a BarEvent."""
        # Determine which arrays to use based on timeframe
        if event.timeframe == "M1" and self._m1_arrays is not None:
            arrays = self._m1_arrays
            tf_arrays = self._tf_arrays
        elif event.timeframe in self._tf_arrays:
            # Higher TF bar closed — nothing to evaluate on this TF directly
            return
        else:
            return

        if arrays.n == 0:
            return

        i = arrays.n - 1
        signals = self.evaluate(i, arrays, tf_arrays)
        for sig in signals:
            if self._bus:
                self._bus.publish(sig)
```

**Also:** SignalEngine needs references to `m1_arrays` and `tf_arrays` for the EventBus path. Add `attach_arrays(m1_arrays, tf_arrays)` method:

```python
def attach_arrays(self, m1_arrays, tf_arrays: dict) -> None:
    """Bind streaming arrays for EventBus-driven evaluation."""
    self._m1_arrays = m1_arrays
    self._tf_arrays = tf_arrays
```

**Keep existing `evaluate(i, arrays, tf_arrays)` method** — it still works for manual/backward-compatible use. The EventBus path calls it internally.

**Keep existing `precompute()` method** — still needed for the refresh-every-500-candles optimization.

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 -c "
from core.signal_engine import SignalEngine
from core.event_bus import EventBus
from core.events import BarEvent
from core.candle_stream import StreamingCandleArrays, Bar
import pandas as pd

# Verify SignalEngine can subscribe to EventBus
bus = EventBus()
# Need at least one valid strategy for init
engine = SignalEngine(['tweezer_reversal'], lookback=5, threshold=1, event_bus=bus)
assert hasattr(engine, '_on_bar_event')
print('Test 1 OK: SignalEngine subscribes to EventBus')

# Verify evaluate still works directly
from core.candle_stream import IncrementalCandleBuilder
m1_arrays = StreamingCandleArrays()
tf_arrays = {}
b = IncrementalCandleBuilder('M1')
# Feed enough ticks to build a few candles
for i in range(120):
    ts = pd.Timestamp('2026-01-01 10:00:00') + pd.Timedelta(seconds=i*30)
    bar = b.ingest_tick(ts, 150.0 + i*0.001, 150.002 + i*0.001, 100)
    if bar:
        m1_arrays.append(bar)

if m1_arrays.n > 5:
    signals = engine.evaluate(m1_arrays.n - 1, m1_arrays, tf_arrays)
    print(f'Test 2 OK: evaluate() returned {len(signals)} signals at i={m1_arrays.n-1}')
else:
    print('Test 2 SKIP: not enough candles built')
"
```

**Commit:**
```bash
git add core/signal_engine.py
git commit -m "refactor: SignalEngine subscribes to BarEvent via EventBus"
```

---

## Task 4: Wire Up CLI — EventBus Tick Loop

**Objective:** Replace `ui/cli.py` bar loop with EventBus-driven tick loop.

**Files:**
- Modify: `ui/cli.py`

**The tick loop becomes minimal** — EventBus handles signal evaluation and trade queueing:

```python
from core.event_bus import EventBus
from core.trade_engine import TradeConfig, TradeEngine
from core.signal_engine import SignalEngine
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays

def run_single_backtest(strategies, args, csv_path, adapter, run_number=None, total_runs=None):
    # --- Validate tick data ---
    if not hasattr(adapter, 'raw_ticks') or adapter.raw_ticks is None:
        raise ValueError(f"No tick data (bid/ask) in {csv_path}. Only tick files supported.")
    raw_ticks = adapter.raw_ticks

    # --- Wire up EventBus ---
    bus = EventBus()

    config = TradeConfig(
        symbol=args.symbol, pip_value=0.01, lot_size=args.lot_size,
        initial_balance=args.balance, spread_pips=args.spread, min_rr=args.min_rr,
    )
    trade_engine = TradeEngine(config, event_bus=bus)
    signal_engine = SignalEngine(
        strategy_names=strategies, lookback=args.lookback,
        threshold=args.threshold, event_bus=bus,
    )

    # --- Determine needed timeframes from strategy registry ---
    from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry
    _populate_registry()
    needed_tfs = set()
    for name in strategies:
        needed_tfs.update(STRATEGY_REGISTRY[name]["timeframes"])
    needed_tfs.discard("M1")

    # --- Streaming arrays + EventBus-aware builders ---
    m1_arrays = StreamingCandleArrays()
    tf_arrays = {tf: StreamingCandleArrays() for tf in needed_tfs}

    m1_builder = IncrementalCandleBuilder("M1", event_bus=bus, symbol=args.symbol)
    tf_builders = {tf: IncrementalCandleBuilder(tf, event_bus=bus, symbol=args.symbol)
                   for tf in needed_tfs}

    # Bind arrays to SignalEngine for EventBus-driven evaluation
    signal_engine.attach_arrays(m1_arrays, tf_arrays)

    # --- Tick loop: 2 lines of core logic ---
    candles_seen = 0
    t0 = time.perf_counter()

    for ts, row in raw_ticks.iterrows():
        bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))

        # 1) Manage positions — direct call, every tick
        trade_engine.on_tick(bid, ask, ts)

        # 2) Feed candle builder — BarEvent → EventBus → SignalEngine → TradeEngine
        m1_builder.ingest_tick(ts, bid, ask, vol)

        # 3) Feed higher TF builders
        for tf, builder in tf_builders.items():
            builder.ingest_tick(ts, bid, ask, vol)

        candles_seen += 1

    # Force close at end
    if trade_engine.open_position:
        last_bid = float(raw_ticks.iloc[-1]["bid"])
        last_ask = float(raw_ticks.iloc[-1]["ask"])
        lp = last_bid if trade_engine.open_position.direction == "LONG" else last_ask
        trade_engine.force_close(lp, raw_ticks.index[-1], "EOD")

    stats = trade_engine.get_stats()
    elapsed = time.perf_counter() - t0
    # ... print results ...
    return stats
```

**Note on precompute:** For backtest performance, call `signal_engine.precompute(m1_arrays, tf_arrays)` every 500 candles inside the loop. The EventBus path works without precompute (falls back to slow evaluate), but precompute makes it ~100x faster.

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 ui/cli.py -s tweezer_reversal --csv data/DAT_ASCII_USDJPY_M1_202605.csv --no-save
# Should print backtest results (trades, win%, PnL, etc.)
```

**Commit:**
```bash
git add ui/cli.py
git commit -m "refactor: CLI uses EventBus-driven tick loop"
```

---

## Task 5: Wire Up FastAPI — EventBus Tick Loop

**Objective:** Replace `ui/backtest/server.py` bar loop with EventBus-driven tick loop.

**Files:**
- Modify: `ui/backtest/server.py`

**Same pattern as CLI** — wire EventBus, 2-line tick loop:

```python
# In /run endpoint:
bus = EventBus()
config = TradeConfig(...)
trade_engine = TradeEngine(config, event_bus=bus)
signal_engine = SignalEngine(req.strategies, lookback=req.lookback,
                             threshold=req.threshold, event_bus=bus)

# Streaming arrays
m1_arrays = StreamingCandleArrays()
tf_arrays = {tf: StreamingCandleArrays() for tf in needed_tfs}
m1_builder = IncrementalCandleBuilder("M1", event_bus=bus, symbol=req.symbol)
tf_builders = {tf: IncrementalCandleBuilder(tf, event_bus=bus, symbol=req.symbol)
               for tf in needed_tfs}
signal_engine.attach_arrays(m1_arrays, tf_arrays)

# Precompute for performance
signal_engine.precompute(m1_arrays, tf_arrays)

# Tick loop
for ts, row in raw_ticks.iterrows():
    bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))
    trade_engine.on_tick(bid, ask, ts)
    m1_builder.ingest_tick(ts, bid, ask, vol)
    for builder in tf_builders.values():
        builder.ingest_tick(ts, bid, ask, vol)

# Force close
if trade_engine.open_position:
    lp = float(raw_ticks.iloc[-1]["bid"]) if trade_engine.open_position.direction == "LONG" \
         else float(raw_ticks.iloc[-1]["ask"])
    trade_engine.force_close(lp, raw_ticks.index[-1], "EOD")
```

**Also update:**
- `durations = [t.bars_held ...]` → `durations = [t.ticks_held ...]`
- `result["avg_bars_held"]` → `result["avg_ticks_held"]`

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 ui/backtest/server.py &
sleep 3
curl -s -X POST http://localhost:8502/run -H 'Content-Type: application/json' -d '{
    "strategies": ["tweezer_reversal"],
    "lookback": 5, "threshold": 1,
    "csv_file": "DAT_ASCII_USDJPY_M1_202605.csv",
    "spread_pips": 0.5
}' | /usr/bin/python3 -m json.tool | head -30
kill %1
```

**Commit:**
```bash
git add ui/backtest/server.py
git commit -m "refactor: FastAPI uses EventBus-driven tick loop"
```

---

## Task 6: Wire Up Streamlit — EventBus + Live-Ready

**Objective:** Replace Streamlit backtest loop with EventBus-driven tick loop. Design for future live tick stream.

**Files:**
- Modify: `ui/streamlit_app/app.py`

**Same EventBus pattern** but with `signals_timeline` collection for chart rendering:

```python
def _run_backtest() -> bool:
    bus = EventBus()
    config = TradeConfig(...)
    trade_engine = TradeEngine(config, event_bus=bus)
    signal_engine = SignalEngine(strategy_names, lookback, threshold, event_bus=bus)

    m1_arrays = StreamingCandleArrays()
    tf_arrays = {tf: StreamingCandleArrays() for tf in needed_tfs}
    m1_builder = IncrementalCandleBuilder("M1", event_bus=bus, symbol=symbol)
    tf_builders = {tf: IncrementalCandleBuilder(tf, event_bus=bus, symbol=symbol)
                   for tf in needed_tfs}
    signal_engine.attach_arrays(m1_arrays, tf_arrays)
    signal_engine.precompute(m1_arrays, tf_arrays)

    # Collect signals for chart rendering
    signals_timeline = []
    def _capture_signal(sig: SignalEvent):
        signals_timeline.append((m1_arrays.n, sig))
    bus.subscribe(SignalEvent, _capture_signal)

    # Tick loop
    candles_seen = 0
    for ts, row in raw_ticks.iterrows():
        bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))
        trade_engine.on_tick(bid, ask, ts)
        m1_builder.ingest_tick(ts, bid, ask, vol)
        for builder in tf_builders.values():
            builder.ingest_tick(ts, bid, ask, vol)
        candles_seen += 1
        if candles_seen % 500 == 0:
            signal_engine.precompute(m1_arrays, tf_arrays)

    # Force close
    ...

    # Store results
    st.session_state.arrays = m1_arrays
    st.session_state.tf_arrays = tf_arrays
    st.session_state.signals_timeline = signals_timeline
    st.session_state.trades_completed = list(trade_engine.trades)
```

**Live-ready design:** The EventBus pattern means swapping the tick source is trivial:
```python
# Backtest: iterate CSV
for ts, row in raw_ticks.iterrows():
    trade_engine.on_tick(row["bid"], row["ask"], ts)
    candle_stream.ingest_tick(ts, row["bid"], row["ask"], row["volume"])

# Live: iterate WebSocket/MT5 feed (future)
async for tick in live_feed:
    trade_engine.on_tick(tick.bid, tick.ask, tick.ts)
    candle_stream.ingest_tick(tick.ts, tick.bid, tick.ask, tick.volume)
```

Same code path. Same EventBus. Different tick source.

**Verification:**
```bash
cd /home/rudi/RSFX && streamlit run ui/streamlit_app/app.py
# Open http://localhost:8501
# Load a CSV tick file, select a strategy, verify backtest runs and charts render
```

**Commit:**
```bash
git add ui/streamlit_app/app.py
git commit -m "refactor: Streamlit uses EventBus-driven tick loop + live-ready"
```

---

## Task 7: Update `README.md`

**Objective:** Update architecture docs to reflect event-driven tick system.

**Files:**
- Modify: `README.md`

**Changes:**
- Architecture diagram: tick → CandleStream → EventBus → SignalEngine → TradeEngine
- TradeEngine: `queue_order()` + `on_tick()` (no `open()`, no `on_bar()`)
- SignalEngine: subscribes to BarEvent, publishes SignalEvent
- Folder structure: add `core/candle_stream.py`
- Remove references to MarketDataStore in the core engine loop
- Update "What's Done" table

**Commit:**
```bash
git add README.md
git commit -m "docs: update README for event-driven tick architecture"
```

---

## Task 8: Full Verification + Push

**Objective:** Verify everything works end-to-end.

**Verification:**
```bash
cd /home/rudi/RSFX

# 1. Import test
/usr/bin/python3 -c "
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays
from core.trade_engine import TradeEngine, TradeConfig, TradeRecord
from core.signal_engine import SignalEngine
from core.event_bus import EventBus
from core.events import BarEvent, SignalEvent
print('All imports OK')
"

# 2. CLI backtest
/usr/bin/python3 ui/cli.py -s tweezer_reversal --csv data/DAT_ASCII_USDJPY_M1_202605.csv --no-save

# 3. FastAPI backtest
/usr/bin/python3 ui/backtest/server.py &
sleep 2
curl -s -X POST http://localhost:8502/run -H 'Content-Type: application/json' \
  -d '{"strategies":["tweezer_reversal"],"lookback":5,"threshold":1,"csv_file":"DAT_ASCII_USDJPY_M1_202605.csv"}' \
  | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Trades: {d[\"result\"][\"total_trades\"]}, Win%: {d[\"result\"][\"win_rate\"]}')"
kill %1

# 4. Push
git push -u origin tick-driven-refactor
```

---

## Migration Notes

### `bars_held` → `ticks_held`
All UIs displaying `bars_held` must update to `ticks_held`.

### Precompute Strategy
The EventBus path works without precompute (falls back to slow evaluate). For backtest performance, refresh every 500 candles:
```python
if candles_seen % 500 == 0:
    signal_engine.precompute(m1_arrays, tf_arrays)
```

### MarketDataStore Deprecated
No longer needed in tick-driven path. IncrementalCandleBuilder builds all timeframes incrementally from ticks. Keep `market_data_store.py` for ad-hoc analysis (correlation/portfolio tools) but remove from core engine loop.

### Raw Tick Data Only
Ticks only. Bar-only CSVs (HistData.com M1 format) are NOT supported. Input must have bid/ask columns.

### Compatibility
- `StreamingCandleArrays` has same attribute surface as `CandleArrays` → strategies unchanged
- `SignalEngine.evaluate(i, arrays, tf_arrays)` still works for manual/backward-compatible use
- All 72 strategies need zero modifications
- CLI/FastAPI/Streamlit validate tick columns and reject bar-only files

### Two Call Patterns
- **Direct** (every tick): `trade_engine.on_tick(bid, ask, ts)` — manages positions
- **EventBus** (candle close): CandleStream → BarEvent → SignalEngine → SignalEvent → TradeEngine.queue_order()
