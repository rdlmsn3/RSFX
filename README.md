# RSFX — Forex Market Replay & Strategy Research Platform

Tick-driven event-driven market replay with 72+ strategies, backtesting, correlation analysis, portfolio optimization, confluence trading, support/resistance detection, and a web UI.

> **Unified Engine Architecture:** All UIs (Streamlit, CLI, FastAPI) share a single `EventBus` + `CandleStream` + `SignalEngine` + `TickEngine`. Ticks are the only clock. Bars are derived views emitted as events.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the replay UI
streamlit run ui/streamlit_app/app.py
# → http://localhost:8501

# Run backtester (all strategies)
python3 -m backtest run

# Run backtester (specific strategies)
python3 -m backtest run -s tweezer_reversal,h1_trend_m5_rsi,cci_ema

# Run strategy correlation analysis
python3 -m backtest correlation

# Run portfolio optimizer
python3 -m backtest portfolio

# Run CLI backtest (quick one-liner)
python3 ui/cli.py -s h1_trend_m5_rsi,cci_ema --csv data/DAT_ASCII_EURUSD_M1_202605.csv

# Run FastAPI web UI
python3 ui/backtest/server.py
# → http://localhost:8502

# See all commands
python3 -m backtest --help
```

---

## Architecture

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
         (subscribed)  (subscribed)
         on BarEvent        │
              │             │
              │  SignalEvent
              └──────►──────┘
                    queue_order()

Next tick ──────────► TradeEngine.on_tick()
                      fills pending, checks SL/TP
```

**Key principle:** Two call patterns coexist:
1. **Direct** (every tick): `trade_engine.on_tick(bid, ask, ts)` — manages positions
2. **EventBus** (candle close): CandleStream → BarEvent → SignalEngine → SignalEvent → TradeEngine.queue_order()

The tick loop in every UI is just two lines:
```python
for tick in tick_source:
    trade_engine.on_tick(bid, ask, ts)
    candle_stream.ingest_tick(ts, bid, ask, vol)  # EventBus handles the rest
```

---

## Folder Structure

```
RSFX/
├── core/                           # SHARED ENGINE LAYER
│   ├── engine.py                   # CandleArrays + trading math (TP/SL, PnL, equity, stats)
│   ├── candle_stream.py            # IncrementalCandleBuilder + StreamingCandleArrays (EventBus)
│   ├── trade_engine.py             # Tick-only trade executor (EventBus SignalEvent subscriber)
│   ├── signal_engine.py            # Strategy evaluation + EventBus BarEvent subscriber
│   ├── trade_store.py              # SQLite persistence (runs + trades)
│   ├── data_loader.py              # Adapter-pattern loaders: CSV + Parquet (auto-detect)
│   ├── tick_candle_builder.py      # Tick → M1 OHLCV aggregation (midprice)
│   ├── market_data_store.py        # Multi-symbol, multi-timeframe store
│   ├── playback_controller.py      # Replay cursor and tick publisher
│   ├── event_bus.py                # Pub/Sub message broker
│   └── events.py                   # SignalEvent, BarEvent, TradeEvent
│
├── ui/                             # CONSUMER LAYERS (thin, no logic)
│   ├── cli.py                      # CLI — argparse → engine → stdout/SQLite
│   ├── backtest/                   # FastAPI web UI — HTTP → engine → JSON
│   │   ├── server.py               # + _sanitize() for numpy/inf/nan
│   │   └── index.html
│   └── streamlit_app/              # Streamlit — replay + live trading
│       ├── app.py
│       └── components/
│           └── chart_renderer.py
│
├── backtest/                       # Analysis tools (not UI)
│   ├── __main__.py                 # Unified entry point (run, correlation, portfolio, ui)
│   ├── correlation.py              # Strategy correlation analysis
│   ├── portfolio.py                # Portfolio optimizer (brute-force)
│   └── buckets.py                  # Named strategy bucket system
│
├── detectors/                      # Strategy definitions
│   ├── pattern_detector.py         # Pattern detection (pluggable)
│   ├── support_resistance.py       # Pivot-based S/R with ATR-adaptive tolerance
│   └── strategies/
│       ├── base.py                 # BaseStrategy (precompute, evaluate_fast)
│       ├── registry.py             # Auto-discovery registry (72+ strategies)
│       └── *.py                    # Individual strategy files
│
├── results/                        # Generated outputs (gitignored)
│   ├── bt_*.csv                    # Backtest results + trades
│   ├── corr_*.csv                  # Correlation outputs
│   ├── portfolio_*.csv             # Portfolio optimizer outputs
│   └── trades.db                   # SQLite database (all runs + trades)
│
├── data/                           # HistData.com CSV + Parquet files
│   └── *.csv / *.parquet
│
├── buckets/                        # Saved strategy buckets (JSON)
│
└── requirements.txt
```

---

## Tools

### 1. Streamlit Replay UI (`ui/streamlit_app/app.py`)

Interactive candle-by-candle replay. Load any M1 CSV or Parquet, play back with charts, equity curve, and strategy selection.

```bash
streamlit run ui/streamlit_app/app.py
# → http://localhost:8501
```

### 2. Backtester (`python3 -m backtest run`)

Unified backtester using SignalEngine + TradeEngine. Saves results to SQLite.

```bash
# All strategies
python3 -m backtest run

# Specific strategies (comma-separated, exact names)
python3 -m backtest run -s tweezer_reversal,h1_trend_m5_rsi

# Custom data file
python3 -m backtest run --csv data/DAT_ASCII_EURUSD_M1_202605.csv --symbol EURUSD

# Options
python3 -m backtest run -s tweezer_reversal --spread 0.5 --min-rr 1.0 --no-save
```

**Note:** Strategy names must be exact (e.g. `h1_trend_m5_rsi`, not `h1_trend`). Use `python3 -m backtest run --help` to see all options.

### 3. Strategy Correlation (`python3 -m backtest correlation`)

Analyzes relationships between strategies based on backtest results.

```bash
python3 -m backtest correlation
python3 -m backtest correlation --trades results/bt_trades_xxx.csv --summary results/bt_xxx.csv
```

**Produces:**
- **Trade overlap matrix** — % of trades that overlap in time per pair
- **PnL correlation** — Pearson correlation of per-trade PnL
- **Equity curve correlation** — Correlation of resampled balance curves
- **Confluence analysis** — Win rate when N strategies fire together
- **Diversification scores** — Per-strategy diversification ranking
- **Portfolio baskets** — Top 20 lowest mutual correlation combos

**Output:** `results/corr_*.csv`

### 4. Portfolio Optimizer (`python3 -m backtest portfolio`)

Brute-force evaluation of all 2-strategy and 3-strategy combinations.

```bash
python3 -m backtest portfolio                          # 2 + 3 combos
python3 -m backtest portfolio --max-combo 4            # also test 4-strategy
python3 -m backtest portfolio --top 30                 # show top 30
```

**Metrics:** Sharpe ratio, Sortino ratio, Calmar ratio, Max DD, Profit Factor, total PnL, win rate.

**Output:** `results/portfolio_results.csv`, `results/portfolio_top50.csv`

### 5. CLI Backtester (`python3 ui/cli.py`)

Thin CLI wrapper for quick backtesting from the command line.

```bash
# Basic usage
python3 ui/cli.py -s tweezer_reversal,h1_trend_m5_rsi --csv data/DAT_ASCII_EURUSD_M1_202605.csv

# With options
python3 ui/cli.py -s h1_trend_m5_rsi --lookback 5 --threshold 2 --spread 0.5 --min-rr 1.0

# Skip saving to SQLite
python3 ui/cli.py -s cci_ema --no-save
```

**Options:**
- `-s / --strategies`: Comma-separated strategy names (exact match required)
- `--csv`: Path to CSV or Parquet data file
- `-l / --lookback`: Confluence lookback window (default: 5)
- `-t / --threshold`: Min strategies agreeing (default: 2)
- `--spread`: Round-trip spread in pips (default: 0.5)
- `--min-rr`: Minimum risk:reward ratio (default: 1.0)
- `--lot-size`: Lot size (default: 0.01)
- `--balance`: Starting balance (default: 1000)
- `--no-save`: Skip saving results to SQLite

**Output:** Prints summary table and saves to `results/trades.db`.

### 6. FastAPI Web UI (`python3 ui/backtest/server.py`)

Browser-based interface for the backtester. All responses sanitized for JSON (numpy types, inf, nan handled).

```bash
python3 ui/backtest/server.py
# → http://localhost:8502
```

**Features:**
- Searchable strategy picker with quick presets (All, Top 3, H1 Trend, Divergence)
- Single-strategy or multi-strategy backtesting
- Data file selector (M1 bars + tick data auto-detected)
- Configurable: lookback, threshold, spread (pips), min R:R
- S/R-aware TP/SL toggle
- Live progress bar during backtest (polls every 1s)
- Equity tracking: start/final balance, PnL ($), max drawdown %, bankruptcy flag
- Extended stats: avg/median win/loss, avg duration, expectancy
- Strategy bucket system (save/load named configurations)
- localStorage persistence across sessions
- **SQLite persistence** — every run saved to `results/trades.db`

### 7. Support/Resistance Detection (`detectors/support_resistance.py`)

Pivot-based S/R with ATR-adaptive tolerance — auto-tunes for any pair.

```python
from detectors.support_resistance import SupportResistance

sr = SupportResistance(df)  # ATR-adaptive tolerance
levels = sr.find_levels()

support = sr.nearest_support(150.250)
resistance = sr.nearest_resistance(150.250)
tp, sl = sr.get_tp_sl(150.250, "LONG", atr_sl=0.30)
```

**ATR-adaptive tolerance:** `tolerance = ATR(14) × 0.3` — scales with pair volatility automatically.

---

## Core Engine API

### `core/engine.py` — Trading Math

```python
from core.engine import CandleArrays, compute_tp_sl, build_result, compute_pnl

# Convert DataFrame to fast NumPy arrays
arrays = CandleArrays.from_dataframe(df)

# Compute TP/SL from signal + ATR fallback + optional S/R override
tp, sl = compute_tp_sl(signal, arrays, i, lookback=100, use_sr=False)

# Build stats dict from trade list
stats = build_result("strategy_name", trades, max_dd, balance_curve)
# → {strategy, total_trades, win_rate, total_pnl_pips, profit_factor, ...}
```

### `core/signal_engine.py` — Strategy Evaluation

```python
from core.signal_engine import SignalEngine

engine = SignalEngine(strategy_names=["h1_trend_m5_rsi", "cci_ema"], lookback=5, threshold=2)
engine.precompute(arrays, tf_arrays)  # optional fast path
signals = engine.evaluate(i, arrays, tf_arrays)
# → list[SignalEvent] with confluence buffer + ATR TP/SL fallback
```

### `core/trade_engine.py` — Trade Lifecycle (Tick-Driven)

```python
from core.trade_engine import TradeEngine, TradeConfig
from core.event_bus import EventBus

bus = EventBus()
config = TradeConfig(symbol="USDJPY", spread_pips=0.5, min_rr=1.0, lot_size=0.01, initial_balance=10000)
engine = TradeEngine(config, event_bus=bus)

engine.queue_order(signal)           # → stores PendingOrder with risk/reward distances
engine.on_tick(bid, ask, ts)        # → fills pending at next tick, checks SL/TP
engine.mark_to_market(close)        # → equity curve only (no execution)
trades = engine.trades              # → list[TradeRecord]
stats = engine.get_stats()          # → dict with win_rate, PF, etc.
```

---

## Adding a New Strategy

```python
# detectors/strategies/my_strategy.py
from detectors.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    @property
    def required_timeframes(self) -> list[str]:
        return ["M5"]

    def evaluate(self, windows: dict, ts) -> list[PatternSignal]:
        df = windows["M5"]
        # ... your logic ...
        return [PatternSignal(name="MY_SIGNAL", ...)]

    # Optional: fast path for backtester
    def precompute(self, arrays, tf_arrays) -> dict:
        return {"ema_20": ..., "rsi_14": ...}

    def evaluate_fast(self, i, arrays, precomputed) -> list[PatternSignal]:
        if precomputed["rsi_14"][i] < 30:
            return [PatternSignal(...)]
        return []
```

Place in `detectors/strategies/` — auto-discovered by the registry.

---

## Performance Notes

- Higher timeframes pre-computed **once** at load time via `pd.DataFrame.resample()`
- `get_window()` uses `searchsorted()` (O log n) — no full-frame copies
- Backtester pre-computes indicators once, runs walk-forward with NumPy arrays only (~0.01s/strategy)
- Parallel execution across CPU cores via `ProcessPoolExecutor`
- S/R detection uses ATR-adaptive tolerance — auto-tunes for any pair
- Tick data auto-converted to M1 via `TickCandleBuilder` (midprice aggregation)
- Tick-level exit scanner uses vectorized NumPy — ~50x faster than Python loop
- Tested on 2M+ tick rows and 30K+ M1 candles without noticeable lag

---

## Test Results (Post-Refactor)

**79/82 tests pass** across 5 phases after major architecture refactor.

| Phase | Tests | Pass | Fail | Notes |
|-------|-------|------|------|-------|
| 1. Core Engine | 48 | 48 | 0 | Data loader, engine math, trade engine, signal engine, trade store |
| 2. CLI Backtest | 10 | 7 | 3 | CLI polish items (see below) |
| 3. FastAPI UI | 15 | 15 | 0 | All endpoints, JSON serialization, SQLite save |
| 5. Integration | 9 | 9 | 0 | End-to-end pipelines, cross-UI consistency |

### Bugs Fixed During Testing

| Bug | Fix |
|-----|-----|
| `float("inf")` in profit_factor → JSON parse error | Changed to `0.0` in `build_result()` |
| numpy types not JSON serializable | Added `_sanitize()` wrapper in FastAPI server |
| TP/SL=0.0 (ATR fallback missing after refactor) | Added `compute_tp_sl()` call in `SignalEngine` |
| `get_trades()` missing JOIN clause | Fixed column-description query in `trade_store.py` |

### Known CLI Issues (low priority)

| Issue | Impact |
|-------|--------|
| No partial name matching (`-s h1_trend` fails) | Must use exact names like `h1_trend_m5_rsi` |
| `--use-sr` flag not in CLI argparse | S/R toggle only works via FastAPI JSON body |
| CLI doesn't create `bt_*.csv` output files | Saves to SQLite only, not CSV |

---

## Results Summary (USDJPY May 2026)

### Best Single Strategies
| Strategy | Trades | Win% | PnL (pips) | PF | MaxDD |
|---|---|---|---|---|---|
| tweezer_reversal | 2,035 | 55.3% | +1,110.8 | 1.57 | 0.42% |
| h1_trend_m5_rsi | 731 | 62.2% | +860.2 | 2.36 | 0.32% |
| ema_ribbon_pullback | 2,138 | 54.5% | +1,282.5 | 1.53 | 0.33% |
| marubozu_trend | 615 | 67.2% | +736.9 | 2.58 | 0.18% |

### S/R-Aware Exits
| Strategy | Mode | Trades | Win% | PnL (pips) | PF |
|---|---|---|---|---|---|
| h1_trend_m5_rsi | ATR | 731 | 62.2% | +860.2 | 2.36 |
| h1_trend_m5_rsi | **S/R** | 323 | **64.4%** | +773.2 | **2.68** |

### Best Portfolio Combos (3-strategy)
| Combo | Sharpe | PnL (pips) | Win% | PF |
|---|---|---|---|---|
| tweezer + h1_rsi + cci_ema | +38.8 | +2,325 | 56.7% | 1.65 |
| tweezer + h1_rsi + h1_macd | +37.6 | +2,283 | 57.7% | 1.78 |
| tweezer + h1_rsi + h1_stoch | +37.2 | +2,288 | 57.8% | 1.82 |

### Confluence Trading
| Config | Trades | Win% | PnL (pips) | PF |
|---|---|---|---|---|
| Independent (sum) | 3,486 | ~57% | +2,324.9 | ~1.7 |
| 2-of-3 same-candle | 252 | 63.9% | +251.7 | 2.25 |
| **2-of-3 lb=5** | **376** | **62.2%** | **+378.3** | **2.10** |
| 2-of-3 lb=10 | 508 | 59.2% | +404.2 | ~1.8 |

---

## What's Done ✅

| Component | Status |
|---|---|
| Data Loader (CSV + Parquet, auto-detect bar vs tick) | ✅ |
| SQLite trade persistence (runs + trades) | ✅ |
| MarketDataStore (M1 → M5, H1, D1) | ✅ |
| EventBus (pub/sub) | ✅ |
| PlaybackController (replay cursor) | ✅ |
| PatternDetector (pluggable strategies) | ✅ |
| Support/Resistance detection (ATR-adaptive) | ✅ |
| core/engine.py — CandleArrays + all trading math | ✅ |
| CandleStream (IncrementalCandleBuilder + StreamingCandleArrays) | ✅ |
| EventBus-driven tick loop (CandleStream → SignalEngine → TradeEngine) | ✅ |
| TradeEngine (tick-only, queue_order + on_tick) | ✅ |
| SignalEngine (BarEvent subscriber + confluence) | ✅ |
| Live-ready architecture (same loop for backtest + live feed) | ✅ |
| CLI backtester (ui/cli.py) | ✅ |
| FastAPI web UI (ui/backtest/server.py) | ✅ |
| Streamlit replay UI (ui/streamlit_app/app.py) | ✅ |
| ChartRenderer (3 subplots) | ✅ |
| 72 strategies (auto-discovered) | ✅ |
| Backtester (parallel, pre-computed) | ✅ |
| Strategy correlation analysis | ✅ |
| Portfolio optimizer (brute-force) | ✅ |
| Confluence web UI | ✅ |
| S/R-aware TP/SL toggle | ✅ |
| Tick-level backtest (bid/ask, MAE/MFE) | ✅ |
| Spread cost modeling (configurable pips) | ✅ |
| Min R:R filter (skip low-quality trades) | ✅ |
| Equity tracking + bankruptcy stop | ✅ |
| Live progress bar (poll-based) | ✅ |
| Strategy bucket system (save/load) | ✅ |
| JSON serialization (numpy/inf/nan safe) | ✅ |
| Post-refactor test suite (79/82 pass) | ✅ |

## What's Next 🔜

| Component | Priority | Notes |
|---|---|---|
| CLI partial name matching | Low | `-s h1_trend` should match all h1_trend_* |
| CLI `--use-sr` flag | Low | Add S/R toggle to argparse |
| CLI CSV output files | Low | Create bt_*.csv alongside SQLite save |
| ML model integration | High | Train on strategy signals → predict best combos |
| Risk Manager | High | Position sizing, max exposure, drawdown limits |
| Strategy Engine | Medium | Dynamic strategy switching based on regime |
| Performance Analytics | Medium | Sharpe, Sortino, Calmar, equity curve analysis |
| Journal System | Medium | Trade journal with notes, screenshots, tags |
| Secondary symbol feeds | Low | DXY, Gold, Oil as context for strategy decisions |
| Live trading mode | High | Paper trading → real execution (EventBus ready) |
