# RSFX — Forex Market Replay & Strategy Research Platform

Event-driven market replay with 72+ strategies, backtesting, correlation analysis, portfolio optimization, confluence trading, support/resistance detection, and a web UI.

> **Unified Engine Architecture:** All UIs (Streamlit, CLI, FastAPI) share a single `SignalEngine` + `TradeEngine` for consistent signal evaluation and trade execution.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the replay UI
streamlit run ui/streamlit_app/app.py
# → http://localhost:8501

# Run backtester (all strategies)
python3 -m backtest backtest

# Run backtester (specific strategies)
python3 -m backtest backtest -s tweezer_reversal,h1_trend_m5_rsi,cci_ema

# Run backtester with S/R-aware TP/SL
python3 -m backtest backtest -s h1_trend_m5_rsi --use-sr

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
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│                                                                 │
│  CSV / Parquet (M1 bars or tick data)                           │
│        ↓                                                        │
│  get_adapter()  →  HistDataAdapter | ParquetAdapter             │
│        ↓                                                        │
│  TickCandleBuilder (if tick data)  →  M1 bars + raw ticks       │
│        ↓                                                        │
│  MarketDataStore  ←  pre-computes M1 → M5, H1, D1 at load      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     ENGINE LAYER (core/)                         │
│                                                                 │
│  SignalEngine.evaluate()                                        │
│    ├── runs 72 strategies via StrategyRegistry                  │
│    ├── confluence buffer (lookback window, threshold)           │
│    └── outputs → [SignalEvent]                                  │
│                             │                                   │
│  TradeEngine.open(signal)   │                                   │
│    ├── compute_tp_sl()      │  ← core/engine.py                 │
│    ├── check_min_rr()       │  ← core/engine.py                 │
│    ├── check_dedup()        │  ← core/engine.py                 │
│    ├── applies spread cost  │                                   │
│    └── on_bar() → TradeRecord + equity curve                    │
│                             │                                   │
│  TradeStore (SQLite)        │  ← persists runs + trades         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     UI LAYER (ui/)                               │
│                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐       │
│  │  CLI     │  │ FastAPI Web  │  │ Streamlit Replay    │       │
│  │ cli.py   │  │ server.py    │  │ app.py              │       │
│  │→ engine  │  │ → engine     │  │ → engine            │       │
│  │→ stdout  │  │ → JSON       │  │ → charts + controls │       │
│  └──────────┘  └──────────────┘  └─────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle:** No component holds a direct reference to any other. All three UIs are thin wrappers that call `SignalEngine` + `TradeEngine` from `core/`.

---

## Folder Structure

```
RSFX/
├── core/                           # SHARED ENGINE LAYER
│   ├── engine.py                   # CandleArrays + trading math (TP/SL, PnL, equity, stats)
│   ├── trade_engine.py             # Unified bar/tick trade executor
│   ├── signal_engine.py            # Strategy evaluation + confluence buffer
│   ├── trade_store.py              # SQLite persistence (runs + trades)
│   ├── data_loader.py              # Adapter-pattern loaders: CSV + Parquet (auto-detect)
│   ├── tick_candle_builder.py      # Tick → M1 OHLCV aggregation (midprice)
│   ├── market_data_store.py        # Multi-symbol, multi-timeframe store
│   ├── playback_controller.py      # Replay cursor and tick publisher
│   ├── event_bus.py                # Pub/Sub message broker
│   └── events.py                   # SignalEvent, BarEvent, TradeEvent
│
├── ui/                             # CONSUMER LAYERS (thin, no logic)
│   ├── cli.py                      # CLI — argparse → engine → stdout/CSV
│   ├── backtest/                   # FastAPI web UI — HTTP → engine → JSON
│   │   ├── server.py
│   │   └── index.html
│   └── streamlit_app/              # Streamlit — replay + live trading
│       ├── app.py
│       └── components/
│           └── chart_renderer.py
│
├── backtest/                       # Analysis tools (not UI)
│   ├── __main__.py                 # Unified entry point (subcommands)
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

### 2. Backtester (`python3 -m backtest backtest`)

Parallel walk-forward backtester. Pre-computes indicators once, runs strategies across CPU cores.

```bash
# All strategies
python3 -m backtest backtest

# Specific strategies (comma-separated, partial match)
python3 -m backtest backtest -s tweezer_reversal,h1_trend_m5_rsi

# All H1 trend strategies
python3 -m backtest backtest -s h1_trend

# Custom data file
python3 -m backtest backtest --csv data/DAT_ASCII_EURUSD_M1_202605.csv --symbol EURUSD

# With S/R-aware TP/SL (support/resistance levels for exits)
python3 -m backtest backtest -s h1_trend_m5_rsi --use-sr

# Options
python3 -m backtest backtest -s tweezer_reversal --workers 4 --top 10 --no-save
```

**Output:** Timestamped CSVs in `results/` with metadata (data file, symbol, strategies, run time). Latest symlink for convenience.

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
- `-s / --strategies`: Comma-separated strategy names
- `--csv`: Path to CSV or Parquet data file
- `-l / --lookback`: Confluence lookback window (default: 5)
- `-t / --threshold`: Min strategies agreeing (default: 2)
- `--spread`: Round-trip spread in pips (default: 0.5)
- `--min-rr`: Minimum risk:reward ratio (default: 1.0)
- `--lot-size`: Lot size (default: 0.01)
- `--balance`: Starting balance (default: 1000)
- `--no-save`: Skip saving results to SQLite

**Output:** Prints summary table and optionally saves to `results/trades.db`.

### 6. FastAPI Web UI (`python3 ui/backtest/server.py`)

Browser-based interface for the confluence backtester.

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

engine = SignalEngine(strategies=["h1_trend_m5_rsi", "cci_ema"], lookback=5, threshold=2)
signals = engine.evaluate(market_data, candle_arrays)
# → list[SignalEvent] with confluence buffer applied
```

### `core/trade_engine.py` — Trade Lifecycle

```python
from core.trade_engine import TradeEngine, TradeConfig

config = TradeConfig(spread_pips=0.5, min_rr=1.0, lot_size=0.01, balance=1000)
engine = TradeEngine(config)

engine.open(signal)  # → opens position with TP/SL
engine.on_bar(candle)  # → checks TP/SL hits, updates equity
trades = engine.get_trades()  # → list[TradeRecord]
stats = engine.get_stats()    # → dict with win_rate, PF, etc.
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
| Unified TradeEngine (bar + tick) | ✅ |
| Unified SignalEngine (confluence buffer) | ✅ |
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

## What's Next 🔜

| Component | Priority | Notes |
|---|---|---|
| ML model integration | High | Train on strategy signals → predict best combos |
| Risk Manager | High | Position sizing, max exposure, drawdown limits |
| Strategy Engine | Medium | Dynamic strategy switching based on regime |
| Performance Analytics | Medium | Sharpe, Sortino, Calmar, equity curve analysis |
| Journal System | Medium | Trade journal with notes, screenshots, tags |
| Secondary symbol feeds | Low | DXY, Gold, Oil as context for strategy decisions |
| Live trading mode | Low | Paper trading → real execution |
| Web dashboard | Low | Persistent analytics, no manual CSV loading |
