# Plan: Deploy Backtest Strategies to Streamlit Playback UI

**Date:** 2026-06-15
**Goal:** Make all 72+ backtest strategies available in the Streamlit replay app, with strategy bucket system shared between backtester and replay UI.

---

## Current State

The replay app (`app.py`) already has the infrastructure:
- **PlaybackController** publishes `MarketTickEvent` on each candle advance
- **PatternDetector** subscribes, fetches M1/M5/H1 windows, calls `strategy.evaluate()`
- **Strategy hot-swap**: sidebar selector changes `detector.strategy` at runtime
- **Signal rendering**: `_render_signals()` shows a table of fired signals
- **Chart markers**: `ChartRenderer` plots L/S markers from signals

**What already works:**
- Single strategy evaluation during playback
- Strategy switching via sidebar
- Signal history table
- Chart overlay of signals

**What's missing:**
1. Need to verify all 72+ strategies are registered and selectable in the UI
2. No bucket system — strategies exist in isolation, no way to group them with config
3. No multi-strategy mode — can only run 1 strategy at a time
4. Bucket config not shared between backtester and replay UI
5. No S/R level overlay during playback

---

## Bucket Format (Shared Between Backtester + Replay UI)

A bucket is a JSON file that captures a **complete, tested configuration**:

```json
{
  "name": "Top 3 Confluence",
  "strategies": ["tweezer_reversal", "h1_trend_m5_rsi", "cci_ema"],
  "use_sr": true,
  "lookback": 5,
  "threshold": 2,
  "description": "Best 3-strategy combo from May 2026 backtest",
  "created": "2026-06-15",
  "backtest_result": {
    "total_trades": 376,
    "win_rate": 62.2,
    "total_pnl_pips": 378.3,
    "profit_factor": 2.10
  }
}
```

**Usage in backtester:**
```bash
# Run confluence with a bucket
python3 -m backtest confluence --bucket buckets/top3.json

# Save backtest result as a bucket
python3 -m backtest confluence -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5 --use-sr --save-bucket "Top 3 Confluence"
```

**Usage in replay UI:**
- Sidebar dropdown: select a bucket from `buckets/` directory
- Loads all strategies + config (use_sr, lookback, threshold)
- Runs in multi-strategy confluence mode automatically

---

## Plan

### Step 1: Verify All 72+ Strategies Are Registered in the UI
**File:** `detectors/strategies/registry.py`

Audit `_populate_registry()`:
- All 72+ strategies have `category`, `timeframes`, `description`, `params`
- Categories are meaningful for UI grouping
- Each strategy has params exposed in sidebar

### Step 2: Bucket System — Core Module
**New file:** `backtest/buckets.py`

```python
@dataclass
class StrategyBucket:
    name: str
    strategies: list[str]
    use_sr: bool = False
    lookback: int = 5
    threshold: int = 2
    description: str = ""
    created: str = ""
    backtest_result: dict = field(default_factory=dict)

    def save(self, path: Path):
        """Save bucket to JSON file."""
        ...

    @classmethod
    def load(cls, path: Path) -> "StrategyBucket":
        """Load bucket from JSON file."""
        ...

    @classmethod
    def list_buckets(cls, buckets_dir: Path) -> list[str]:
        """List all available bucket names."""
        ...
```

**Bucket directory:** `RSFX/buckets/`
```
buckets/
├── top3_confluence.json
├── mtf_all.json
├── divergence.json
└── custom_my_combo.json
```

**Pre-defined buckets (auto-generated):**
- `all_single_tf.json` — every Single TF strategy, threshold=1 (any signal triggers)
- `all_mtf.json` — every MTF strategy, threshold=2
- `all_strategies.json` — everything, threshold=3

### Step 3: Bucket CLI Integration
**File:** `backtest/confluence.py`

Add `--bucket` and `--save-bucket` flags:

```bash
# Load a bucket and run
python3 -m backtest confluence --bucket buckets/top3.json

# Run and save result as bucket
python3 -m backtest confluence \
  -s tweezer_reversal,h1_trend_m5_rsi,cci_ema \
  --lookback 5 --use-sr --threshold 2 \
  --save-bucket "Top 3 Confluence"
```

The `--save-bucket` flag creates a JSON file in `buckets/` with the config + backtest results.

### Step 4: Bucket UI in Streamlit Replay
**File:** `app.py`

Sidebar changes:

```
Strategy Mode: [Single] [Bucket]

─── If Bucket mode ───
Bucket: [Top 3 Confluence ▼] [Custom...]
  
  Shows loaded config:
  ┌─────────────────────────────────┐
  │ Strategies: tweezer, h1_rsi, cci│
  │ S/R: ON | Lookback: 5 | Thresh: 2│
  │ Backtest: 62.2% WR, +378 pips   │
  └─────────────────────────────────┘
  
  [Load Bucket] [Edit] [Save As]

─── If Custom... ───
Multi-select: [tweezer_reversal ✓] [h1_trend_m5_rsi ✓] [cci_ema ✓] ...
S/R Toggle: [ON]
Lookback: [5]
Threshold: [2]
[Save as Bucket]
```

### Step 5: Multi-Strategy Evaluation Engine
**File:** `detectors/pattern_detector.py`

Extend PatternDetector for multi-strategy:

```python
class PatternDetector:
    def __init__(self, ..., strategies: list[BaseStrategy] = None):
        self._strategies = strategies or [strategy or EMAStochasticStrategy()]

    def _on_market_tick(self, event):
        all_signals = []
        for strategy in self._strategies:
            signals = strategy.evaluate(windows, timestamp)
            for s in signals:
                s.metadata["strategy"] = strategy.name
            all_signals.extend(signals)

        # Confluence: 2+ strategies agree on same candle
        if len(self._strategies) > 1:
            all_signals = self._mark_confluence(all_signals)

        for signal in all_signals:
            self._signals.append(signal)
            self._bus.publish(PatternDetectedEvent(...))

    def _mark_confluence(self, signals):
        """Group by timestamp+direction, mark when count >= threshold."""
        ...
```

**Behavior:**
- Single strategy: backward compatible, no change
- Multi strategy: runs all, tags each signal with source
- Confluence: 5-candle buffer — if strategy A fires at candle 100 and strategy B fires at candle 103, and both agree on direction, it counts as confluence
- Threshold from bucket config (default: 2-of-N must agree)

### Step 6: Enhanced Chart Signal Visualization
**File:** `views/chart_renderer.py`

**Single strategy mode:** unchanged (L/S markers)

**Multi-strategy / bucket mode:**
- Each strategy gets a unique color from a palette
- Signals as colored dots (color = source strategy)
- Confluence signals: larger marker with white border
- Hover tooltip: strategy name, direction, confidence
- Mini legend showing strategy colors

### Step 7: S/R Level Overlay on Chart
**Files:** `app.py`, `views/chart_renderer.py`

- Compute S/R on visible window (last 100 candles) using `SupportResistance`
- Draw horizontal lines: green (support), red (resistance)
- Opacity based on strength
- Toggle in sidebar: "Show S/R Levels" (OFF by default)
- When signal fires near S/R, annotate "Near Support" / "Near Resistance"

### Step 8: Prepare for Live Data Integration
**Files:** `core/data_loader.py`, `core/playback_controller.py`

Architecture already supports it:
- `PlaybackController.tick()` can be called by live feed
- `MarketDataStore` can update with new candles
- Only need: `LiveDataFeed` class that publishes `MarketTickEvent` on each new candle
- **No code changes now** — document the integration point

---

## Files to Change

| File | Change |
|---|---|
| `detectors/strategies/registry.py` | Audit + fix missing categories/params |
| `backtest/buckets.py` | **New** — bucket dataclass, load/save/list |
| `backtest/confluence.py` | Add `--bucket` and `--save-bucket` flags |
| `app.py` | Bucket selector UI, multi-strategy mode, S/R toggle |
| `detectors/pattern_detector.py` | Multi-strategy evaluation, 5-candle confluence buffer |
| `views/chart_renderer.py` | Color-coded signals, confluence markers, S/R lines |
| `buckets/` | **New directory** — pre-defined + user buckets |

---

## Verification

1. **Backtest path**: `python3 -m backtest confluence -s X,Y,Z --lookback 5 --use-sr --save-bucket "My Bucket"` → creates `buckets/my_bucket.json`
2. **Backtest with bucket**: `python3 -m backtest confluence --bucket buckets/my_bucket.json` → loads config, runs confluence
3. **Replay UI**: load bucket from sidebar → verify all strategies run, S/R toggle works, confluence markers appear
4. **Compare**: backtest result should match replay UI behavior for same bucket config
5. **S/R overlay**: toggle on → verify lines appear on chart
6. **Custom bucket**: build ad-hoc in UI → save → reload → verify persistence

---

## Risks

- **Performance**: Running many strategies per tick. Mitigation: only run selected bucket, pre-compute where possible.
- **UI clutter**: Too many signals. Mitigation: filter by strategy, show only confluence, opacity.
- **Streamlit rerun**: Each tick = full page rerun. Mitigation: already works at 10× speed.
- **Bucket drift**: UI and backtester diverge. Mitigation: same JSON format, same `StrategyBucket` class.

---

## Decisions (Final)

1. **No pre-defined buckets** — only user-created buckets
2. **S/R computation: visible window only** — fast, no caching needed
3. **Save to `buckets/` directory** — standard location
4. **Confluence buffer: 5 candles** — configurable per bucket
