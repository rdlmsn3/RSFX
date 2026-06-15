# Scalping Strategies Implementation Plan

## Goal
Build 70 pluggable scalping strategies for the RSFX backtesting platform and add frontend UI for strategy selection and parameter configuration.

---

## Current Architecture (What Exists)

```
detectors/
├── strategies/
│   ├── base.py                    # BaseStrategy ABC
│   ├── ema_stochastic.py          # M1 only: EMA + Stoch + Candle
│   └── ema_stochastic_mtf.py      # H1+M5+M1: 3-layer MTF
├── pattern_detector.py            # Hot-swappable strategy holder
└── signal.py                      # PatternSignal dataclass

app.py                             # Streamlit UI (hardcoded strategy)
```

**Key interfaces:**
- `BaseStrategy.evaluate(windows, timestamp) → list[PatternSignal]`
- `PatternDetector.strategy` setter for hot-swap
- `PatternSignal` with `name`, `start_time`, `end_time`, `confidence`, `metadata`

---

## Proposed Approach

### Phase 1: Strategy Registry & Catalog

Create a central registry that maps strategy names to classes and metadata.

**New file: `detectors/strategies/registry.py`**

```python
STRATEGY_REGISTRY = {
    "ema_stochastic": {
        "class": EMAStochasticStrategy,
        "category": "Single TF",
        "timeframes": ["M1"],
        "description": "EMA crossover + Stochastic + Candlestick",
        "params": {
            "ema_fast": {"type": "int", "default": 9, "min": 3, "max": 50},
            "ema_slow": {"type": "int", "default": 21, "min": 5, "max": 100},
            "stoch_k": {"type": "int", "default": 5, "min": 3, "max": 20},
            "stoch_d": {"type": "int", "default": 3, "min": 1, "max": 10},
            "oversold": {"type": "float", "default": 20.0, "min": 5, "max": 40},
            "overbought": {"type": "float", "default": 80.0, "min": 60, "max": 95},
        },
    },
    # ... 68 more strategies
}
```

**Benefits:**
- Single source of truth for all strategies
- Frontend can dynamically render parameter controls
- Easy to add new strategies (just register)
- Category grouping for UI organization

---

### Phase 2: Strategy Implementation (70 Strategies)

#### Group 1: Single TF - EMA/RSI Based (1-10)
| # | Name | File | Indicators |
|---|------|------|------------|
| 1 | `ema_rsi_cross` | `ema_rsi_cross.py` | EMA 9/21 + RSI 14 |
| 2 | `ema_stoch_cross` | `ema_stoch_cross.py` | EMA 9/21 + Stoch 5,3,3 |
| 3 | `bb_rsi_bounce` | `bb_rsi_bounce.py` | Bollinger + RSI 14 |
| 4 | `bb_squeeze_breakout` | `bb_squeeze_breakout.py` | BB Width + Volume |
| 5 | `macd_ema_trend` | `macd_ema_trend.py` | MACD + EMA 50 |
| 6 | `macd_histogram_div` | `macd_histogram_div.py` | MACD Histogram divergence |
| 7 | `rsi_ema_trend` | `rsi_ema_trend.py` | RSI 14 + EMA 50 |
| 8 | `stoch_ema_trend` | `stoch_ema_trend.py` | Stoch 14,3,3 + EMA 9/21 |
| 9 | `cci_ema` | `cci_ema.py` | CCI 20 + EMA 50 |
| 10 | `williams_ema` | `williams_ema.py` | Williams %R 14 + EMA 9/21 |

#### Group 2: Single TF - Trend Following (11-20)
| # | Name | File | Indicators |
|---|------|------|------------|
| 11 | `supertrend_ema` | `supertrend_ema.py` | Supertrend + EMA 50 |
| 12 | `parabolic_sar_ema` | `parabolic_sar_ema.py` | PSAR + EMA 9/21 |
| 13 | `adx_di_ema` | `adx_di_ema.py` | ADX + DI + EMA 50 |
| 14 | `keltner_breakout` | `keltner_breakout.py` | Keltner Channel + Volume |
| 15 | `donchian_breakout` | `donchian_breakout.py` | Donchian 20 + Volume |
| 16 | `heikin_ashi_ema` | `heikin_ashi_ema.py` | Heikin Ashi + EMA 9/21 |
| 17 | `ma_ribbon_pullback` | `ma_ribbon_pullback.py` | EMA 5,8,13,21 ribbon |
| 18 | `ma_envelope_bounce` | `ma_envelope_bounce.py` | MA Envelope + RSI |
| 19 | `rsi_divergence_ema` | `rsi_divergence_ema.py` | RSI divergence + EMA |
| 20 | `stoch_divergence_ema` | `stoch_divergence_ema.py` | Stoch divergence + EMA |

#### Group 3: Two TF (H1+M5) (21-26)
| # | Name | File | Indicators |
|---|------|------|------------|
| 21 | `h1_trend_m5_ema_cross` | `h1_trend_m5_ema_cross.py` | H1 EMA + M5 EMA cross |
| 22 | `h1_trend_m5_rsi` | `h1_trend_m5_rsi.py` | H1 EMA + M5 RSI bounce |
| 23 | `h1_trend_m5_stoch` | `h1_trend_m5_stoch.py` | H1 EMA + M5 Stoch |
| 24 | `h1_trend_m5_macd` | `h1_trend_m5_macd.py` | H1 EMA + M5 MACD |
| 25 | `h1_trend_m5_bb` | `h1_trend_m5_bb.py` | H1 EMA + M5 Bollinger |
| 26 | `h1_adx_m5_ema` | `h1_adx_m5_ema.py` | H1 ADX + M5 EMA cross |

#### Group 4: Price Action (27-34)
| # | Name | File | Indicators |
|---|------|------|------------|
| 27 | `pin_bar_ema` | `pin_bar_ema.py` | Pin bar + EMA 50 |
| 28 | `engulfing_ema` | `engulfing_ema.py` | Engulfing + EMA 9/21 |
| 29 | `inside_bar_breakout` | `inside_bar_breakout.py` | Inside bar + EMA |
| 30 | `three_bar_reversal` | `three_bar_reversal.py` | 3-bar pattern + RSI |
| 31 | `morning_evening_star` | `morning_evening_star.py` | Star patterns + EMA 50 |
| 32 | `harami_trend` | `harami_trend.py` | Harami + EMA + RSI |
| 33 | `tweezer_reversal` | `tweezer_reversal.py` | Tweezer + EMA |
| 34 | `marubozu_trend` | `marubozu_trend.py` | Marubozu + EMA |

#### Group 5: Volume Based (35-38)
| # | Name | File | Indicators |
|---|------|------|------------|
| 35 | `volume_spike_ema` | `volume_spike_ema.py` | Volume spike + EMA |
| 36 | `volume_profile_ema` | `volume_profile_ema.py` | Volume profile + EMA |
| 37 | `obv_ema` | `obv_ema.py` | OBV + EMA |
| 38 | `ad_ema` | `ad_ema.py` | A/D line + EMA |

#### Group 6: Channel Based (39-41)
| # | Name | File | Indicators |
|---|------|------|------------|
| 39 | `donchian_rsi` | `donchian_rsi.py` | Donchian + RSI |
| 40 | `keltner_rsi` | `keltner_rsi.py` | Keltner + RSI |
| 41 | `atr_channel_breakout` | `atr_channel_breakout.py` | ATR channel + Volume |

#### Group 7: Momentum + Mean Reversion (42-44)
| # | Name | File | Indicators |
|---|------|------|------------|
| 42 | `rsi_bb_squeeze` | `rsi_bb_squeeze.py` | RSI + BB squeeze |
| 43 | `stoch_bb_bounce` | `stoch_bb_bounce.py` | Stoch + BB bounce |
| 44 | `macd_bb_breakout` | `macd_bb_breakout.py` | MACD + BB breakout |

#### Group 8: Multi-Indicator Confluence (45-47)
| # | Name | File | Indicators |
|---|------|------|------------|
| 45 | `triple_confirm` | `triple_confirm.py` | EMA + RSI + MACD |
| 46 | `trend_momentum_vol` | `trend_momentum_vol.py` | EMA + Stoch + BB |
| 47 | `adx_rsi_ema` | `adx_rsi_ema.py` | ADX + RSI + EMA |

#### Group 9: Session/Pivot (48-53)
| # | Name | File | Indicators |
|---|------|------|------------|
| 48 | `london_ny_breakout` | `london_ny_breakout.py` | Session range + EMA |
| 49 | `asian_range_breakout` | `asian_range_breakout.py` | Asian range + EMA |
| 50 | `pivot_ema_bounce` | `pivot_ema_bounce.py` | Pivot + EMA |
| 51 | `pivot_rsi_bounce` | `pivot_rsi_bounce.py` | Pivot + RSI |
| 52 | `fib_ema_bounce` | `fib_ema_bounce.py` | Fibonacci + EMA |
| 53 | `fib_rsi_bounce` | `fib_rsi_bounce.py` | Fibonacci + RSI |

#### Group 10: Trend Following Scalps (54-58)
| # | Name | File | Indicators |
|---|------|------|------------|
| 54 | `ema_ribbon_pullback` | `ema_ribbon_pullback.py` | EMA ribbon pullback |
| 55 | `vwap_ema_cross` | `vwap_ema_cross.py` | VWAP + EMA cross |
| 56 | `vwap_bounce` | `vwap_bounce.py` | VWAP bounce + RSI |
| 57 | `ichimoku_cloud_bounce` | `ichimoku_cloud_bounce.py` | Ichimoku bounce |
| 58 | `ichimoku_cloud_break` | `ichimoku_cloud_break.py` | Ichimoku breakout |

#### Group 11: Divergence (59-62)
| # | Name | File | Indicators |
|---|------|------|------------|
| 59 | `rsi_divergence` | `rsi_divergence.py` | RSI divergence |
| 60 | `macd_divergence` | `macd_divergence.py` | MACD divergence |
| 61 | `stoch_divergence` | `stoch_divergence.py` | Stoch divergence |
| 62 | `volume_divergence` | `volume_divergence.py` | OBV divergence |

#### Group 12: Hybrid (63-67)
| # | Name | File | Indicators |
|---|------|------|------------|
| 63 | `trend_mean_reversion` | `trend_mean_reversion.py` | EMA trend + RSI pullback |
| 64 | `breakout_retest` | `breakout_retest.py` | Breakout + retest + EMA |
| 65 | `momentum_exhaustion` | `momentum_exhaustion.py` | Momentum candle + Stoch |
| 66 | `scalp_pullback` | `scalp_pullback.py` | ADX + EMA + RSI pullback |
| 67 | `gap_fill` | `gap_fill.py` | Gap + RSI extreme |

#### Group 13: Advanced (68-70)
| # | Name | File | Indicators |
|---|------|------|------------|
| 68 | `ema_macd_rsi_confluence` | `ema_macd_rsi_confluence.py` | 3-indicator confluence |
| 69 | `bb_stoch_volume` | `bb_stoch_volume.py` | BB + Stoch + Volume |
| 70 | `supertrend_rsi_ema` | `supertrend_rsi_ema.py` | Supertrend + RSI + EMA |

---

### Phase 3: Strategy Categorization

**New file: `detectors/strategies/categories.py`**

```python
STRATEGY_CATEGORIES = {
    "Single TF - EMA/RSI": [1-10],
    "Single TF - Trend Following": [11-20],
    "Two TF (H1+M5)": [21-26],
    "Price Action": [27-34],
    "Volume Based": [35-38],
    "Channel Based": [39-41],
    "Momentum + Mean Reversion": [42-44],
    "Multi-Indicator Confluence": [45-47],
    "Session/Pivot": [48-53],
    "Trend Following Scalps": [54-58],
    "Divergence": [59-62],
    "Hybrid": [63-67],
    "Advanced": [68-70],
}
```

---

### Phase 4: Frontend Changes

#### 4.1 Strategy Selector in Sidebar

**Location:** `app.py` → `_render_sidebar()` after Data Source section

```python
# ---- Strategy Selection ------------------------------------------------
st.markdown("### Strategy")

# Category dropdown
category = st.selectbox(
    "Category",
    options=list(STRATEGY_CATEGORIES.keys()),
    index=0,
)

# Strategy dropdown (filtered by category)
category_strategies = STRATEGY_CATEGORIES[category]
strategy_names = [s["name"] for s in category_strategies]
selected_name = st.selectbox(
    "Strategy",
    options=strategy_names,
    index=0,
)

# Show description
st.caption(STRATEGY_REGISTRY[selected_name]["description"])

# Dynamic parameter controls
st.markdown("#### Parameters")
params = {}
for param_name, param_info in STRATEGY_REGISTRY[selected_name]["params"].items():
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

# Apply strategy button
if st.button("Apply Strategy", use_container_width=True):
    strategy_class = STRATEGY_REGISTRY[selected_name]["class"]
    new_strategy = strategy_class(**params)
    st.session_state.detector.strategy = new_strategy
    st.rerun()
```

#### 4.2 Strategy Info Panel

**Location:** `app.py` → Status bar area

```python
# Show current strategy info
c1, c2, c3 = st.columns(3)
c1.metric("Strategy", det.strategy.name)
c2.metric("Timeframes", ", ".join(STRATEGY_REGISTRY[det.strategy.name]["timeframes"]))
c3.metric("Category", STRATEGY_REGISTRY[det.strategy.name]["category"])
```

#### 4.3 Signal List Panel

**Location:** `app.py` → Below chart

```python
# Signal history
if det.signal_count > 0:
    st.markdown("### Signals")
    signals_df = pd.DataFrame([
        {
            "Time": s.end_time,
            "Direction": s.metadata.get("direction", "?"),
            "Strategy": s.metadata.get("strategy", "?"),
            "Confidence": s.confidence,
        }
        for s in det.signals
    ])
    st.dataframe(signals_df, use_container_width=True, height=300)
```

---

### Phase 5: File Structure

```
detectors/
├── strategies/
│   ├── __init__.py              # Update: export all strategies
│   ├── base.py                  # Existing
│   ├── registry.py              # NEW: strategy registry + categories
│   ├── categories.py            # NEW: category definitions
│   │
│   ├── # Group 1: Single TF EMA/RSI
│   ├── ema_rsi_cross.py         # NEW
│   ├── ema_stoch_cross.py       # NEW
│   ├── bb_rsi_bounce.py         # NEW
│   ├── bb_squeeze_breakout.py   # NEW
│   ├── macd_ema_trend.py        # NEW
│   ├── macd_histogram_div.py    # NEW
│   ├── rsi_ema_trend.py         # NEW
│   ├── stoch_ema_trend.py       # NEW
│   ├── cci_ema.py               # NEW
│   ├── williams_ema.py          # NEW
│   │
│   ├── # Group 2: Trend Following
│   ├── supertrend_ema.py        # NEW
│   ├── parabolic_sar_ema.py     # NEW
│   ├── adx_di_ema.py            # NEW
│   ├── keltner_breakout.py      # NEW
│   ├── donchian_breakout.py     # NEW
│   ├── heikin_ashi_ema.py       # NEW
│   ├── ma_ribbon_pullback.py    # NEW
│   ├── ma_envelope_bounce.py    # NEW
│   ├── rsi_divergence_ema.py    # NEW
│   ├── stoch_divergence_ema.py  # NEW
│   │
│   ├── # ... (60 more strategy files)
│   │
│   ├── ema_stochastic.py        # Existing
│   └── ema_stochastic_mtf.py    # Existing
│
├── pattern_detector.py          # Existing (no changes needed)
└── signal.py                    # Existing (no changes needed)

app.py                           # Modify: add strategy selector UI
```

---

### Phase 6: Implementation Order

#### Batch 1: Foundation (Do First)
1. Create `detectors/strategies/registry.py`
2. Create `detectors/strategies/categories.py`
3. Update `detectors/strategies/__init__.py`
4. Modify `app.py` sidebar for strategy selection

#### Batch 2: Core Strategies (High Value)
5. Implement Group 1: Single TF EMA/RSI (10 strategies)
6. Implement Group 2: Trend Following (10 strategies)
7. Implement Group 3: Two TF (6 strategies)

#### Batch 3: Pattern Strategies
8. Implement Group 4: Price Action (8 strategies)
9. Implement Group 5: Volume Based (4 strategies)
10. Implement Group 6: Channel Based (3 strategies)

#### Batch 4: Advanced Strategies
11. Implement Group 7-8: Momentum + Confluence (6 strategies)
12. Implement Group 9: Session/Pivot (6 strategies)
13. Implement Group 10: Trend Following Scalps (5 strategies)
14. Implement Group 11-13: Divergence + Hybrid + Advanced (11 strategies)

---

### Phase 7: Testing & Validation

#### 7.1 Unit Tests
```python
# tests/test_strategies.py
def test_all_strategies_implement_base():
    """All registered strategies must implement BaseStrategy."""
    for name, info in STRATEGY_REGISTRY.items():
        assert issubclass(info["class"], BaseStrategy)

def test_all_strategies_have_required_metadata():
    """All strategies must have category, timeframes, description, params."""
    for name, info in STRATEGY_REGISTRY.items():
        assert "category" in info
        assert "timeframes" in info
        assert "description" in info
        assert "params" in info
```

#### 7.2 Integration Test
```python
# Test strategy hot-swap in PatternDetector
def test_strategy_hot_swap():
    bus = EventBus()
    store = MarketDataStore()
    det = PatternDetector(bus, store, strategy=EMAStochasticStrategy())
    assert det.strategy.name == "ema_stochastic"
    
    det.strategy = EMAStochasticMTFStrategy()
    assert det.strategy.name == "ema_stochastic_mtf"
```

#### 7.3 Manual Testing Checklist
- [ ] Load CSV data
- [ ] Select each strategy category
- [ ] Select each strategy within category
- [ ] Adjust parameters
- [ ] Click "Apply Strategy"
- [ ] Verify signals appear on chart
- [ ] Verify signal list updates
- [ ] Hot-swap strategies during playback

---

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 70 strategies = large codebase | Maintenance burden | Use consistent template, shared utilities |
| TA-Lib dependency for most strategies | Missing indicators | Graceful fallback when TA-Lib unavailable |
| Strategy parameter validation | Invalid configs | Use dataclass validators, min/max in UI |
| Frontend performance with many strategies | Slow dropdown | Category filtering, lazy loading |
| Indicator calculation performance | Slow playback | Pre-compute indicators, cache results |

---

### Open Questions

1. **Indicator caching:** Should we cache computed indicators (EMA, RSI, etc.) across ticks to avoid recomputation? Current implementation recomputes every tick.

2. **Strategy favorites:** Should we add a "Favorites" section for frequently used strategies?

3. **Strategy import/export:** Should users be able to save/load custom strategy configurations?

4. **Backtest comparison:** Should we add a "Compare Strategies" mode that runs multiple strategies on same data?

---

## Next Steps

After plan approval:
1. Implement Phase 1 (Registry + Categories)
2. Implement Phase 4 (Frontend changes)
3. Implement Phase 2 in batches (strategies)
4. Test each batch before moving to next

**Estimated effort:** 4-6 hours for foundation + 2-3 hours per batch of 10 strategies.
