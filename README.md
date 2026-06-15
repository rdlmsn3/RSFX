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
python3 backtest.py

# Run backtester (specific strategies)
python3 backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema

# Run backtester with S/R-aware TP/SL
python3 backtest.py -s h1_trend_m5_rsi --use-sr

# Run strategy correlation analysis
python3 strategy_correlation.py

# Run portfolio optimizer (2 + 3 strategy combos)
python3 portfolio_optimizer.py

# Run confluence backtester
python3 confluence_backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5

# Run confluence backtester with S/R-aware exits
python3 confluence_backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5 --use-sr

# Run confluence web UI
python3 confluence_ui/server.py
# → http://localhost:8502
```

---

## Architecture

```
CSV Data Source
      ↓
DataAdapter (HistDataAdapter / MT5Adapter / …)
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
├── backtest.py                     # Independent strategy backtester (parallel)
├── strategy_correlation.py         # Correlation analysis between strategies
├── portfolio_optimizer.py          # Brute-force portfolio combinatorics
├── confluence_backtest.py          # Signal-buffer confluence backtester
│
├── confluence_ui/
│   ├── index.html                  # Web UI frontend
│   └── server.py                   # FastAPI backend (port 8502)
│
├── core/
│   ├── data_loader.py              # Adapter-pattern CSV loaders
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
└── data/
    └── *.csv                       # HistData.com format data files
```

---

## Tools

### 1. Replay UI (`app.py`)
Interactive Streamlit-based market replay. Load any M1 CSV, play back candle-by-candle with charts.

```bash
streamlit run app.py
```

### 2. Backtester (`backtest.py`)
Parallel walk-forward backtester. Pre-computes indicators once, runs strategies across CPU cores.

```bash
# All strategies
python3 backtest.py

# Specific strategies (comma-separated, partial match)
python3 backtest.py -s tweezer_reversal,h1_trend_m5_rsi

# All H1 trend strategies
python3 backtest.py -s h1_trend

# Custom data file
python3 backtest.py --csv data/DAT_ASCII_EURUSD_M1_202605.csv --symbol EURUSD

# With S/R-aware TP/SL (support/resistance levels for exits)
python3 backtest.py -s h1_trend_m5_rsi --use-sr

# Options
python3 backtest.py -s tweezer_reversal --workers 4 --top 10 --no-save
```

**Output:** Timestamped CSVs with metadata (data file, symbol, strategies used, run time). Latest symlink for convenience.

### 3. Strategy Correlation (`strategy_correlation.py`)
Analyzes relationships between strategies based on backtest results.

```bash
python3 strategy_correlation.py
```

**Produces:**
- **Trade overlap matrix** — % of trades that overlap in time per pair
- **PnL correlation** — Pearson correlation of per-trade PnL
- **Equity curve correlation** — Correlation of resampled balance curves
- **Confluence analysis** — Win rate when N strategies fire together
- **Diversification scores** — Per-strategy diversification ranking
- **Portfolio baskets** — Top 20 lowest mutual correlation combos

**Output CSVs:** `corr_overlap.csv`, `corr_pnl.csv`, `corr_equity.csv`, `corr_confluence.csv`, `corr_baskets.csv`, `corr_diversification.csv`

### 4. Portfolio Optimizer (`portfolio_optimizer.py`)
Brute-force evaluation of all 2-strategy and 3-strategy combinations.

```bash
python3 portfolio_optimizer.py                      # 2 + 3 combos
python3 portfolio_optimizer.py --max-combo 4        # also test 4-strategy
python3 portfolio_optimizer.py --top 30             # show top 30
```

**Metrics per combo:** Sharpe ratio, Sortino ratio, Calmar ratio, Max DD, Profit Factor, total PnL, win rate.

**Output:** `portfolio_results.csv` (all 22K+ combos), `portfolio_top50.csv`

### 5. Confluence Backtester (`confluence_backtest.py`)
Signal-buffer confluence — strategies fire independently, trades execute when N out of M agree within a candle window.

```bash
# 2-of-3 must agree within 5 candles
python3 confluence_backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5

# 3-of-5 must agree
python3 confluence_backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema,ema_ribbon_pullback,marubozu_trend --lookback 5 --threshold 3

# With S/R-aware exits
python3 confluence_backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5 --use-sr

# See who voted when
python3 confluence_backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5 --show-votes
```

**How it works:**
1. Each strategy evaluates independently at every candle
2. When a strategy fires, its signal enters a buffer (active for N candles)
3. If another strategy fires within that window and agrees on direction → confluence trade
4. TP/SL from the most recent (triggering) signal

**Output:** Timestamped CSVs with metadata.

### 6. Support/Resistance Detection (`detectors/support_resistance.py`)
Pivot-based S/R with ATR-adaptive tolerance — auto-tunes for any pair, no manual parameters needed.

```python
from detectors.support_resistance import SupportResistance

sr = SupportResistance(df)  # ATR-adaptive tolerance
levels = sr.find_levels()   # list of SRLevel

# Query nearest levels
support = sr.nearest_support(150.250)
resistance = sr.nearest_resistance(150.250)

# Get S/R-aware TP/SL
tp, sl = sr.get_tp_sl(150.250, "LONG", atr_sl=0.30)
```

**How it works:**
1. Find swing highs/lows using rolling window extrema
2. Cluster nearby pivots within ATR-based tolerance
3. Score by touch count, recency, proximity, and cleanliness
4. Returns sorted levels (strongest first)

**ATR-adaptive tolerance:** `tolerance = ATR(14) × 0.3` — automatically scales with pair volatility:
- EURUSD (ATR ~0.0008) → tolerance ~0.00024 (2.4 pips)
- USDJPY (ATR ~0.008) → tolerance ~0.0024 (0.24 pips)
- GBPJPY (ATR ~0.05) → tolerance ~0.015 (1.5 pips)

**Toggle in backtester:** `--use-sr` flag overrides ATR-based TP/SL with S/R-targeted exits.

### 7. Confluence Web UI (`confluence_ui/`)
Browser-based interface for the confluence backtester.

```bash
python3 confluence_ui/server.py
# → http://localhost:8502
```

**Features:**
- Searchable multi-select strategy picker (72 strategies)
- Quick presets: Top 3, H1 Trend, Divergence
- Data file selector with file size display
- Configurable lookback + threshold
- Results: stats, trade table, strategy participation chart
- Settings persist across sessions (localStorage)

---

## Results Summary (USDJPY May 2026)

### Best Single Strategies
| Strategy | Trades | Win% | PnL (pips) | PF | MaxDD |
|---|---|---|---|---|---|
| tweezer_reversal | 2,035 | 55.3% | +1,110.8 | 1.57 | 0.42% |
| h1_trend_m5_rsi | 731 | 62.2% | +860.2 | 2.36 | 0.32% |
| ema_ribbon_pullback | 2,138 | 54.5% | +1,282.5 | 1.53 | 0.33% |
| marubozu_trend | 615 | 67.2% | +736.9 | 2.58 | 0.18% |

### S/R-Aware Exits (Toggle Comparison)
| Strategy | Mode | Trades | Win% | PnL (pips) | PF |
|---|---|---|---|---|---|
| h1_trend_m5_rsi | ATR | 731 | 62.2% | +860.2 | 2.36 |
| h1_trend_m5_rsi | **S/R** | 323 | **64.4%** | +773.2 | **2.68** |
| tweezer_reversal | ATR | 2,035 | 55.3% | +1,110.8 | 1.57 |
| tweezer_reversal | S/R | 937 | 57.0% | +322.6 | 1.17 |

S/R toggle helps some strategies (h1_trend_m5_rsi: WR +2.2%, PF +0.32) but not others — depends on market regime. Works best in ranging markets with clear S/R zones.

### Best Portfolio Combos (3-strategy)
| Combo | Sharpe | PnL (pips) | Win% | PF |
|---|---|---|---|---|
| tweezer + h1_rsi + cci_ema | +38.8 | +2,325 | 56.7% | 1.65 |
| tweezer + h1_rsi + h1_macd | +37.6 | +2,283 | 57.7% | 1.78 |
| tweezer + h1_rsi + h1_stoch | +37.2 | +2,288 | 57.8% | 1.82 |

### Confluence Trading (Signal-Buffer)
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

    # Optional: fast path for backtester (pre-compute indicators)
    def precompute(self, arrays, tf_arrays) -> dict:
        # Return pre-computed NumPy arrays
        return {"ema_20": ..., "rsi_14": ...}

    def evaluate_fast(self, i, arrays, precomputed) -> list[PatternSignal]:
        # Use pre-computed arrays instead of DataFrame slicing
        if precomputed["rsi_14"][i] < 30:
            return [PatternSignal(...)]
        return []
```

Place in `detectors/strategies/` — auto-discovered by the registry. No other files need to change.

---

## Performance Notes

- Higher timeframes (M5, H1, D1) pre-computed **once** at load time via `pd.DataFrame.resample()`
- `get_window()` uses `searchsorted()` (O log n) — no full-frame copies during playback
- Backtester pre-computes indicators once per strategy, runs walk-forward loop with NumPy arrays only (~0.01s/strategy vs ~8s/strategy with pandas rolling)
- Parallel execution across CPU cores via `ProcessPoolExecutor`
- S/R detection uses ATR-adaptive tolerance — auto-tunes for any pair without manual parameters
- Tested on 2M+ row DataFrames without noticeable playback lag

---

## Roadmap

| Component | Status |
|---|---|
| Data Loader (HistData) | ✅ |
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
| ML model integration | 🔜 |
| Strategy Engine | 🔜 |
| Risk Manager | 🔜 |
| Performance Analytics | 🔜 |
| Journal System | 🔜 |
| Secondary symbol feeds (DXY, Gold) | 🔜 |
