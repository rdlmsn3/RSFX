# Tick-Driven System Refactor — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace the bar/candle-driven clock with a tick-driven system where every tick is the only clock. Bars are derived from ticks for signal evaluation only.

**Architecture:** The tick stream becomes the single source of truth. Each tick: (1) manages open positions and fills pending orders via `on_tick()`, (2) feeds incremental candle builders for M1/M5/H1. When M1 closes, strategies are evaluated and signals queued. No more `open()` — only `queue_order()` + `on_tick()`.

**Tech Stack:** Python 3, NumPy, Pandas, existing RSFX core/

---

## Pre-Flight: Safety Branch ✅

Branch `pre-tick-driven-refactor` pushed. Current work on `tick-driven-refactor` branch.

---

## What Changes vs What Stays

### Unchanged (zero modifications):
- `core/engine.py` — CandleArrays, compute_tp_sl, build_result, compute_pnl, etc.
- `core/signal_engine.py` — evaluate(i, arrays, tf_arrays) works identically
- `core/events.py` — SignalEvent, BarEvent, TradeEvent
- `core/event_bus.py` — EventBus pub/sub
- `core/data_loader.py` — HistDataAdapter, ParquetAdapter
- `core/market_data_store.py` — pre-computes M5/H1/D1
- `core/trade_store.py` — SQLite persistence
- `core/tick_candle_builder.py` — batch builder (kept for compatibility)
- `detectors/` — all 72 strategies
- `backtest/correlation.py`, `backtest/portfolio.py` — consume trades, not loop
- `backtest/__main__.py` — entry point router

### Replaced:
- `core/trade_engine.py` — bar+tick dual mode → tick-only (queue_order + on_tick)
- `ui/cli.py` backtest loop — bar loop → inline tick loop
- `ui/backtest/server.py` backtest loop — bar loop → inline tick loop
- `ui/streamlit_app/app.py` backtest loop — bar loop → inline tick loop + live-ready

### New file:
- `core/candle_stream.py` — IncrementalCandleBuilder + StreamingCandleArrays

---

## Task 1: Create `core/candle_stream.py`

**Objective:** Add incremental, tick-driven candle construction and streaming arrays.

**Files:**
- Create: `core/candle_stream.py` (from `core/new_candle_stream.py` with adaptations)

**Content:** Copy `core/new_candle_stream.py` → `core/candle_stream.py`. The file contains:
- `Bar` dataclass (timestamp, OHLCV)
- `IncrementalCandleBuilder` — feeds ticks one at time, emits finished `Bar` on bucket boundary
- `StreamingCandleArrays` — append-only, exposes same `.timestamps/.opens/.highs/.lows/.closes/.volumes/.n` as `CandleArrays`, so `signal_engine.evaluate(i, arrays, tf_arrays)` works unchanged

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 -c "
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays
import pandas as pd

b = IncrementalCandleBuilder('M1')
a = StreamingCandleArrays()

# Simulate 2 ticks in same minute → no bar yet
bar = b.ingest_tick(pd.Timestamp('2026-01-01 10:00:00'), 1.1, 1.1002, 100)
assert bar is None, 'Should not emit yet'

# Tick in next minute → emits finished bar
bar = b.ingest_tick(pd.Timestamp('2026-01-01 10:01:00'), 1.1005, 1.1007, 200)
assert bar is not None
assert bar.open == 1.1  # midprice of first tick
assert bar.close == (1.1005 + 1.1007) / 2

a.append(bar)
assert a.n == 1
assert a.opens[0] == 1.1
print('OK: candle_stream works')
"
```

**Commit:**
```bash
git add core/candle_stream.py
git commit -m "feat: add IncrementalCandleBuilder + StreamingCandleArrays"
```

---

## Task 2: Rewrite `core/trade_engine.py` — Tick-Only

**Objective:** Replace bar+tick dual mode with tick-only execution. No `open()`, no `on_bar()`, no `attach_ticks()`.

**Files:**
- Replace: `core/trade_engine.py` (from `core/new_trade_engine.py` with adaptations)

**Key API changes:**
```
OLD API                          NEW API
─────────────────────────────    ──────────────────────────────
engine.open(signal)              engine.queue_order(signal)
engine.on_bar(bar_event)         engine.on_tick(bid, ask, ts)
engine.on_tick(bid, ask, ts)     engine.on_tick(bid, ask, ts)  ← same
engine.attach_ticks(df)          (removed — ticks stream directly)
engine._real_fill_price(...)     (removed — fills at next tick)
```

**Critical differences:**
1. `queue_order(signal)` — stores `PendingOrder` with risk/reward *distances* only (no absolute TP/SL prices yet)
2. `on_tick(bid, ask, ts)` — the ONLY execution entry point:
   - If position open → check SL/TP on bid/ask
   - If flat + order pending → fill at ask(LONG)/bid(SHORT), re-anchor TP/SL from fill price
3. `mark_to_market(close)` — equity curve only, no execution effect
4. `TradeRecord.bars_held` → `TradeRecord.ticks_held`
5. `OpenPosition.bar_idx/current_bar_idx` → `OpenPosition.tick_count`

**Content:** Copy `core/new_trade_engine.py` → `core/trade_engine.py`. Adapt:
- Import path for `compute_tp_sl` stays the same (from `core.engine`)
- Import `check_min_rr`, `check_dedup`, `update_equity` from `core.engine`
- Keep `TradeConfig` dataclass identical (same fields)
- Keep `TradeRecord.to_dict()` but change `bars_held` key to `ticks_held`

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 -c "
from core.trade_engine import TradeEngine, TradeConfig
from core.events import SignalEvent
import pandas as pd

config = TradeConfig(symbol='USDJPY', pip_value=0.01, lot_size=0.01, initial_balance=10000, spread_pips=0.5)
engine = TradeEngine(config)

# Queue a signal
sig = SignalEvent(
    strategy_name='test', direction='LONG',
    entry_price=150.0, take_profit=150.50, stop_loss=149.70,
    timestamp=pd.Timestamp('2026-01-01 10:00:00'),
)
engine.queue_order(sig)
assert engine.pending_order is not None, 'Should have pending order'
assert engine.open_position is None, 'No position yet'

# Fill on next tick
engine.on_tick(bid=150.001, ask=150.003, timestamp=pd.Timestamp('2026-01-01 10:00:01'))
assert engine.open_position is not None, 'Should be filled'
assert engine.pending_order is None, 'Pending consumed'
assert engine.open_position.entry_price == 150.003, f'Fill at ask, got {engine.open_position.entry_price}'

# SL/TP re-anchored from fill
pos = engine.open_position
risk = abs(pos.entry_price - pos.stop_loss) / 0.01
reward = abs(pos.take_profit - pos.entry_price) / 0.01
print(f'Risk: {risk:.1f} pips, Reward: {reward:.1f} pips')

print('OK: trade_engine tick-driven works')
"
```

**Commit:**
```bash
git add core/trade_engine.py
git commit -m "refactor: replace bar+tick TradeEngine with tick-only execution"
```

---
## Task 3: Rewrite CLI Backtest Loop — Tick-Driven

**Objective:** Replace `ui/cli.py` bar-driven loop with inline tick-driven loop.

**Files:**
- Modify: `ui/cli.py`

**Key changes:**
1. Remove `CandleArrays.from_dataframe()` and bar loop entirely
2. Load tick data via adapter (`adapter.raw_ticks`) — reject if None
3. Inline the tick-driven loop (same pattern as `new_run_backtest_tick_driven.py`)
4. No `open()` → use `queue_order()` after signal evaluation
5. No `on_bar()` → use `on_tick()` + `mark_to_market()`
6. Display results from `engine.get_stats()`

**Tick-driven loop embedded in `run_single_backtest()`:**
```python
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays
from core.trade_engine import TradeConfig, TradeEngine
from core.signal_engine import SignalEngine

def run_single_backtest(strategies, args, csv_path, adapter, run_number=None, total_runs=None):
    # --- Validate tick data ---
    if not hasattr(adapter, 'raw_ticks') or adapter.raw_ticks is None:
        raise ValueError(f"No tick data (bid/ask) in {csv_path}. Only tick files supported.")
    raw_ticks = adapter.raw_ticks

    config = TradeConfig(
        symbol=args.symbol, pip_value=0.01, lot_size=args.lot_size,
        initial_balance=args.balance, spread_pips=args.spread, min_rr=args.min_rr,
    )
    signal_engine = SignalEngine(strategy_names=strategies, lookback=args.lookback, threshold=args.threshold)
    trade_engine = TradeEngine(config)

    # --- Incremental candle builders ---
    from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry
    _populate_registry()
    needed_tfs = set()
    for name in strategies:
        needed_tfs.update(STRATEGY_REGISTRY[name]["timeframes"])
    needed_tfs.discard("M1")

    m1_builder = IncrementalCandleBuilder("M1")
    m1_arrays = StreamingCandleArrays()
    tf_builders = {tf: IncrementalCandleBuilder(tf) for tf in needed_tfs}
    tf_arrays = {tf: StreamingCandleArrays() for tf in needed_tfs}

    # --- Tick loop ---
    candles_seen = 0
    t0 = time.perf_counter()

    for ts, row in raw_ticks.iterrows():
        bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))

        # 1) Manage position / fill queue — always first
        trade_engine.on_tick(bid, ask, ts)

        # 2) Feed higher TF builders
        for tf, builder in tf_builders.items():
            bar = builder.ingest_tick(ts, bid, ask, vol)
            if bar: tf_arrays[tf].append(bar)

        # 3) M1 boundary → evaluate signals → queue
        bar = m1_builder.ingest_tick(ts, bid, ask, vol)
        if bar is None:
            continue
        m1_arrays.append(bar)
        trade_engine.mark_to_market(bar.close)
        candles_seen += 1

        if candles_seen % 500 == 0:
            signal_engine.precompute(m1_arrays, tf_arrays)

        i = m1_arrays.n - 1
        signals = signal_engine.evaluate(i, m1_arrays, tf_arrays)
        for sig in signals:
            trade_engine.queue_order(sig)

    # Force close at end
    if trade_engine.open_position:
        last_bid, last_ask = float(raw_ticks.iloc[-1]["bid"]), float(raw_ticks.iloc[-1]["ask"])
        last_price = last_bid if trade_engine.open_position.direction == "LONG" else last_ask
        trade_engine.force_close(last_price, raw_ticks.index[-1], "EOD")

    # Print stats
    stats = trade_engine.get_stats()
    elapsed = time.perf_counter() - t0
    # ... print same format as before
    return stats
```

**Verification:**
```bash
cd /home/rudi/RSFX && /usr/bin/python3 ui/cli.py -s tweezer_reversal --csv data/DAT_ASCII_USDJPY_M1_202605.csv --no-save
# Should print backtest results (trades, win%, PnL, etc.)
```

**Commit:**
```bash
git add ui/cli.py
git commit -m "refactor: CLI backtest loop switched to tick-driven execution"
```

---

## Task 4: Rewrite FastAPI Backtest Loop — Tick-Driven

**Objective:** Replace `ui/backtest/server.py` `/run` endpoint bar loop with inline tick-driven loop.

**Files:**
- Modify: `ui/backtest/server.py`

**Key changes:**
1. Remove `CandleArrays.from_dataframe()` and bar loop
2. Get `raw_ticks` from `get_store()` (already cached there)
3. Inline tick-driven loop in `/run` endpoint
4. No `open()` → `queue_order()`, no `on_bar()` → `on_tick()` + `mark_to_market()`
5. Update `bars_held` → `ticks_held` in extended stats

**Tick-driven loop in `/run` endpoint:**
```python
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays

# ... (config, signal_engine, trade_engine setup same as before) ...

m1_builder = IncrementalCandleBuilder("M1")
m1_arrays = StreamingCandleArrays()
tf_builders = {tf: IncrementalCandleBuilder(tf) for tf in needed_tfs}
tf_arrays_stream = {tf: StreamingCandleArrays() for tf in needed_tfs}

candles_seen = 0
for ts, row in raw_ticks.iterrows():
    bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))
    trade_engine.on_tick(bid, ask, ts)

    for tf, builder in tf_builders.items():
        bar = builder.ingest_tick(ts, bid, ask, vol)
        if bar: tf_arrays_stream[tf].append(bar)

    bar = m1_builder.ingest_tick(ts, bid, ask, vol)
    if bar is None: continue
    m1_arrays.append(bar)
    trade_engine.mark_to_market(bar.close)
    candles_seen += 1

    if candles_seen % 500 == 0:
        signal_engine.precompute(m1_arrays, tf_arrays_stream)

    i = m1_arrays.n - 1
    signals = signal_engine.evaluate(i, m1_arrays, tf_arrays_stream)
    for sig in signals:
        trade_engine.queue_order(sig)

# Force close at end
if trade_engine.open_position:
    last_bid = float(raw_ticks.iloc[-1]["bid"])
    last_ask = float(raw_ticks.iloc[-1]["ask"])
    lp = last_bid if trade_engine.open_position.direction == "LONG" else last_ask
    trade_engine.force_close(lp, raw_ticks.index[-1], "EOD")
```

**Also update:**
- `durations = [t.bars_held for t in trades]` → `durations = [t.ticks_held for t in trades]`
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
# Should return result with trades, win_rate, total_pnl_pips, etc.
kill %1
```

**Commit:**
```bash
git add ui/backtest/server.py
git commit -m "refactor: FastAPI backtest loop switched to tick-driven execution"
```

---

## Task 5: Rewrite Streamlit — Tick-Driven + Live-Ready

**Objective:** Replace Streamlit backtest loop with tick-driven execution, and design the tick loop to accept a live tick stream later.

**Files:**
- Modify: `ui/streamlit_app/app.py`

**Design principle:** The tick loop is a generator-friendly pattern. For backtest: iterate `raw_ticks.iterrows()`. For live: iterate an async tick feed (WebSocket, MT5 stream, etc.) — same `on_tick()` + `ingest_tick()` calls.

**Key changes:**
1. Remove `CandleArrays.from_dataframe()` and bar loop
2. Load tick data via adapter — reject if None
3. Inline tick-driven loop with `signals_timeline` collection for chart rendering
4. No `open()` → `queue_order()`, no `on_bar()` → `on_tick()` + `mark_to_market()`
5. Store `m1_arrays` (StreamingCandleArrays) in session state for replay
6. Structure tick processing as a callable `_process_tick()` helper — same function works for backtest iteration AND future live feed

**Tick-driven loop in `_run_backtest()`:**
```python
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays

# --- Setup builders ---
m1_builder = IncrementalCandleBuilder("M1")
m1_arrays = StreamingCandleArrays()
tf_builders = {tf: IncrementalCandleBuilder(tf) for tf in needed_tfs}
tf_arrays = {tf: StreamingCandleArrays() for tf in needed_tfs}

candles_seen = 0
signals_timeline = []

for ts, row in raw_ticks.iterrows():
    bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))
    trade_engine.on_tick(bid, ask, ts)

    for tf, builder in tf_builders.items():
        bar = builder.ingest_tick(ts, bid, ask, vol)
        if bar: tf_arrays[tf].append(bar)

    bar = m1_builder.ingest_tick(ts, bid, ask, vol)
    if bar is None: continue
    m1_arrays.append(bar)
    trade_engine.mark_to_market(bar.close)
    candles_seen += 1

    if candles_seen % 500 == 0:
        signal_engine.precompute(m1_arrays, tf_arrays)

    i = m1_arrays.n - 1
    signals = signal_engine.evaluate(i, m1_arrays, tf_arrays)
    for sig in signals:
        trade_engine.queue_order(sig)
        signals_timeline.append((candles_seen, sig))

# Force close at end
if trade_engine.open_position:
    last_bid = float(raw_ticks.iloc[-1]["bid"])
    last_ask = float(raw_ticks.iloc[-1]["ask"])
    lp = last_bid if trade_engine.open_position.direction == "LONG" else last_ask
    trade_engine.force_close(lp, raw_ticks.index[-1], "EOD")

# Store results in session state
st.session_state.arrays = m1_arrays
st.session_state.tf_arrays = tf_arrays
st.session_state.signals_timeline = signals_timeline
st.session_state.trades_completed = list(trade_engine.trades)
```

**Live-ready design:** Extract tick processing into a helper that works for both backtest and live:
```python
def _process_tick(trade_engine, signal_engine, m1_builder, m1_arrays,
                  tf_builders, tf_arrays, bid, ask, vol, ts, candles_seen):
    """Process a single tick — works for backtest (historical) and live (streaming)."""
    trade_engine.on_tick(bid, ask, ts)

    for tf, builder in tf_builders.items():
        bar = builder.ingest_tick(ts, bid, ask, vol)
        if bar: tf_arrays[tf].append(bar)

    bar = m1_builder.ingest_tick(ts, bid, ask, vol)
    if bar is None:
        return candles_seen, []

    m1_arrays.append(bar)
    trade_engine.mark_to_market(bar.close)
    candles_seen += 1

    if candles_seen % 500 == 0:
        signal_engine.precompute(m1_arrays, tf_arrays)

    i = m1_arrays.n - 1
    signals = signal_engine.evaluate(i, m1_arrays, tf_arrays)
    for sig in signals:
        trade_engine.queue_order(sig)

    return candles_seen, signals
```

**Also update:**
- Store `m1_arrays` as `arrays` in session state for chart rendering
- `TradeRecord.bars_held` → `ticks_held` in any display code
- Keep `_signal_to_pattern()` working with new signal indices
```

**Verification:**
```bash
cd /home/rudi/RSFX && streamlit run ui/streamlit_app/app.py
# Open http://localhost:8501
# Load a CSV tick file, select a strategy, verify backtest runs and charts render
```

**Commit:**
```bash
git add ui/streamlit_app/app.py
git commit -m "refactor: Streamlit tick-driven + live-ready architecture"


---

## Task 6: Update `README.md`

**Objective:** Update architecture docs to reflect tick-driven system.

**Files:**
- Modify: `README.md`

**Changes:**
- Architecture diagram: tick stream → IncrementalCandleBuilder → M1 bars → SignalEngine
- TradeEngine description: `queue_order()` + `on_tick()` instead of `open()` + `on_bar()`
- Folder structure: add `core/candle_stream.py`
- "What's Done" table: update TradeEngine description
- "What's Next": remove tick-driven items

**Commit:**
```bash
git add README.md
git commit -m "docs: update README for tick-driven architecture"
```

---

## Task 7: Run Full Test Suite

**Objective:** Verify nothing is broken.

**Verification:**
```bash
cd /home/rudi/RSFX

# 1. Import test
/usr/bin/python3 -c "
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays
from core.trade_engine import TradeEngine, TradeConfig, TradeRecord
from core.signal_engine import SignalEngine
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays
print('All imports OK')
"

# 2. Quick backtest via CLI
/usr/bin/python3 ui/cli.py -s tweezer_reversal --csv data/DAT_ASCII_USDJPY_M1_202605.csv --no-save

# 3. FastAPI test
/usr/bin/python3 ui/backtest/server.py &
sleep 2
curl -s -X POST http://localhost:8502/run -H 'Content-Type: application/json' -d '{"strategies":["tweezer_reversal"],"lookback":5,"threshold":1,"csv_file":"DAT_ASCII_USDJPY_M1_202605.csv"}' | /usr/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Trades: {d[\"result\"][\"total_trades\"]}, Win%: {d[\"result\"][\"win_rate\"]}')"
kill %1

# 4. Full test suite if tests exist
/usr/bin/python3 -m pytest tests/ -v 2>/dev/null || echo "No pytest tests found"

# 5. Push
git push -u origin tick-driven-refactor
```

---

## Migration Notes

### `bars_held` → `ticks_held`
Any UI displaying `bars_held` needs to update to `ticks_held`. The FastAPI server calculates `avg_bars_held` — change to `avg_ticks_held`.

### Precompute Strategy
With tick-driven execution, `precompute()` runs on growing arrays. Two options:
- **(a)** Skip precompute entirely (use slow path) — correct but O(lookback) per candle
- **(b)** Refresh every 500 candles — pragmatic middle ground (recommended default)

The `run_tick_backtest` function accepts `precompute_refresh_every` param.

### Raw Tick Data Requirement
**Ticks only.** The tick-driven system only accepts raw tick data (bid/ask/volume per tick). Bar-only CSVs (HistData.com M1 format) are NOT supported. If the input doesn't have bid/ask columns, reject it. No synthetic tick generation.

### Compatibility
- `StreamingCandleArrays` exposes the same attribute surface as `CandleArrays` → strategies work unchanged
- `SignalEngine.evaluate(i, arrays, tf_arrays)` works with both `CandleArrays` and `StreamingCandleArrays`
- All 72 strategies need zero modifications
- CLI/FastAPI/Streamlit must validate input has tick columns (bid/ask) and reject bar-only files with a clear error
