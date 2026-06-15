# RSFX — Forex Market Replay & Strategy Research Platform

Event-driven market replay with 72+ strategies, backtesting, correlation analysis, portfolio optimization, confluence trading, support/resistance detection, and a web UI.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the replay UI
streamlit run app.py
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

# Run confluence backtester
python3 -m backtest confluence -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5

# Run confluence web UI
python3 -m backtest ui
# → http://localhost:8502

# See all commands
python3 -m backtest --help
```

---

## Architecture

```
CSV Data Source (M1 bars or tick data)
      ↓
DataAdapter (HistDataAdapter — auto-detects M1 vs tick format)
      ├── M1 bars → MarketDataStore
      └── Tick data → TickCandleBuilder → M1 bars + raw ticks → MarketDataStore
      ↓
MarketDataStore   ← pre-computes M1 → M5, H1, D1 at load time
      ↓
PlaybackController
      ↓
EventBus  (publish / subscribe)
  ├── PatternDetector   → subscribes to MarketTickEvent
  ├── TradeEngine       → subscribes to MarketTickEvent
  ├── SupportResistance → pivot-based S/R detection (ATR-adaptive)
  └── (future) MLEngine, RiskManager, StrategyEngine, …

app.py (Streamlit View)
  → calls controller methods
  → reads state from controller / detector / trade_engine
  → passes data windows to ChartRenderer
  → renders go.Figure from ChartRenderer
```

No component holds a direct reference to any other component.
All communication flows through the `EventBus`.

---

## Folder Structure

```
RSFX/
├── app.py                          # Streamlit replay UI
├── requirements.txt
├── README.md
│
├── backtest/                       # Backtest module (python3 -m backtest)
│   ├── __init__.py
│   ├── __main__.py                 # Unified entry point (subcommands)
│   ├── backtester.py               # Parallel strategy backtester + tick-level exit scanner
│   ├── correlation.py              # Strategy correlation analysis
│   ├── portfolio.py                # Portfolio optimizer (brute-force)
│   ├── confluence.py               # Signal-buffer confluence backtester (equity tracking)
│   ├── buckets.py                  # Named strategy bucket system
│   └── ui/
│       ├── index.html              # Confluence web UI frontend
│       └── server.py               # FastAPI backend (port 8502)
│
├── results/                        # Generated backtest outputs (gitignored)
│   ├── bt_*.csv                    # Backtest results + trades
│   ├── corr_*.csv                  # Correlation outputs
│   ├── portfolio_*.csv             # Portfolio optimizer outputs
│   ├── confluence_*.csv            # Confluence outputs
│   └── *_latest.csv                # Symlinks to most recent
│
├── core/
│   ├── data_loader.py              # Adapter-pattern CSV loaders (M1 + tick auto-detect)
│   ├── tick_candle_builder.py      # Tick → M1 OHLCV aggregation (midprice)
│   ├── market_data_store.py        # Multi-symbol, multi-timeframe store
│   ├── playback_controller.py      # Replay cursor and tick publisher
│   ├── event_bus.py                # Pub/Sub message broker
│   ├── events.py                   # Event dataclasses
│   └── trade_engine.py             # Trade simulation engine
│
├── detectors/
│   ├── pattern_detector.py         # Pattern detection (pluggable)
│   ├── support_resistance.py       # Pivot-based S/R with ATR-adaptive tolerance
│   └── strategies/
│       ├── base.py                 # BaseStrategy (precompute, evaluate_fast)
│       ├── registry.py             # Auto-discovery registry (72+ strategies)
│       └── *.py                    # Individual strategy files
│
├── views/
│   └── chart_renderer.py           # Plotly figure factory
│
├── buckets/                        # Saved strategy buckets (JSON)
│
└── data/
    └── *.csv                       # HistData.com format data files (M1 + tick)
```

---

## Tools

### 1. Replay UI (`app.py`)
Interactive Streamlit-based market replay. Load any M1 CSV, play back candle-by-candle with charts.

```bash
streamlit run app.py
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

### 5. Confluence Backtester (`python3 -m backtest confluence`)
Signal-buffer confluence — strategies fire independently, trades execute when N out of M agree within a candle window.

```bash
# 2-of-3 must agree within 5 candles
python3 -m backtest confluence -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5

# 3-of-5 must agree
python3 -m backtest confluence -s tweezer_reversal,h1_trend_m5_rsi,cci_ema,ema_ribbon_pullback,marubozu_trend --lookback 5 --threshold 3

# With S/R-aware exits
python3 -m backtest confluence -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5 --use-sr
```

**How it works:**
1. Each strategy evaluates independently at every candle
2. When a strategy fires, its signal enters a buffer (active for N candles)
3. If another strategy fires within that window and agrees → confluence trade
4. TP/SL from the most recent (triggering) signal

**Tick-level execution:**
- Entry: first tick at/after signal candle → midprice `(bid+ask)/2`
- Exit: vectorized scan over raw ticks for SL/TP hit
- LONG exits use bid price, SHORT exits use ask price (proper spread modeling)
- MAE/MFE calculated from tick-level excursion

**Equity tracking:**
- Starts with configurable balance (default $1,000)
- Dollar PnL per trade: `pnl_pips × pip_value × lot_size × 100,000`
- Bankruptcy stop: halts trading when balance ≤ $0
- Tracks: peak balance, max drawdown %, final balance

**Output:** `results/confluence_*.csv`

### 6. Support/Resistance Detection (`detectors/support_resistance.py`)
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

### 7. Confluence Web UI (`python3 -m backtest ui`)
Browser-based interface for the confluence backtester.

```bash
python3 -m backtest ui
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

## Roadmap

| Component | Status |
|---|---|
| Data Loader (HistData) | ✅ |
| Tick data support (auto-detect + M1 conversion) | ✅ |
| MarketDataStore (M1/M5/H1/D1) | ✅ |
| EventBus | ✅ |
| PlaybackController | ✅ |
| PatternDetector (pluggable strategy) | ✅ |
| TradeEngine | ✅ |
| ChartRenderer (3 subplots) | ✅ |
| Streamlit UI | ✅ |
| MTF Strategy (H1 trend + M5 momentum + M1 entry) | ✅ |
| Candlestick pattern recognition | ✅ (via TA-Lib) |
| Backtester (parallel, pre-computed) | ✅ |
| Strategy correlation analysis | ✅ |
| Portfolio optimizer (brute-force) | ✅ |
| Confluence backtester (signal-buffer) | ✅ |
| Confluence web UI | ✅ |
| Support/Resistance detection (ATR-adaptive) | ✅ |
| S/R-aware TP/SL toggle | ✅ |
| Tick-level backtest (bid/ask, MAE/MFE from ticks) | ✅ |
| Single-strategy backtest support | ✅ |
| Spread cost modeling (configurable pips) | ✅ |
| Min R:R filter (skip low-quality trades) | ✅ |
| Equity tracking + bankruptcy stop | ✅ |
| Live progress bar (poll-based) | ✅ |
| Strategy bucket system (save/load) | ✅ |
| ML model integration | 🔜 |
| Strategy Engine | 🔜 |
| Risk Manager | 🔜 |
| Performance Analytics | 🔜 |
| Journal System | 🔜 |
| Secondary symbol feeds (DXY, Gold) | 🔜 |
