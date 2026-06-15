# Forex Market Replay Platform

An event-driven, local web-based market replay platform built with Python, Streamlit, and Plotly.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

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
project/
├── app.py                      # Streamlit View layer
├── requirements.txt
├── README.md
│
├── core/
│   ├── data_loader.py          # Adapter-pattern CSV loaders
│   ├── market_data_store.py    # Multi-symbol, multi-timeframe store
│   ├── playback_controller.py  # Replay cursor and tick publisher
│   ├── event_bus.py            # Pub/Sub message broker
│   ├── events.py               # Event dataclasses
│   └── trade_engine.py         # Trade simulation engine
│
├── detectors/
│   └── pattern_detector.py     # Pattern detection (pluggable)
│
├── views/
│   └── chart_renderer.py       # Plotly figure factory
│
└── data/
    └── sample.csv              # HistData.com format sample
```

---

## Using Real Data

Download M1 data from [HistData.com](https://www.histdata.com/download-free-forex-data/) (ASCII format).

Enter the full path to the `.csv` file in the sidebar **CSV File Path** field and click **Load / Reload Data**.

The platform handles files with and without header rows automatically.

---

## Extending the Platform

### Adding a new data source

```python
# core/data_loader.py
class MT5Adapter(DataAdapter):
    def load(self, path: str) -> pd.DataFrame:
        # Parse MT5 CSV format
        # Return clean DataFrame with DatetimeIndex
        ...
```

No other files need to change.

### Adding a pattern detector

```python
# detectors/pattern_detector.py → PatternDetector.scan_for_patterns()
def _scan_candlestick(self, window: pd.DataFrame) -> list[PatternSignal]:
    import talib
    result = talib.CDLENGULFING(window.open, window.high, window.low, window.close)
    signals = []
    for i, val in enumerate(result):
        if val != 0:
            signals.append(PatternSignal(
                name="ENGULFING_BULL" if val > 0 else "ENGULFING_BEAR",
                start_time=window.index[i-1],
                end_time=window.index[i],
                confidence=1.0,
            ))
    return signals
```

### Adding a new engine (e.g. ML Engine)

```python
# engines/ml_engine.py
class MLEngine:
    def __init__(self, event_bus, data_store):
        event_bus.subscribe(MarketTickEvent, self._on_tick)

    def _on_tick(self, event: MarketTickEvent):
        window = self._store.get_window(event.symbol, "M1", event.timestamp, 100)
        signal = self._model.predict(window)
        if signal:
            self._bus.publish(PatternDetectedEvent(...))
```

Register it in `app.py` during `_load_data()` – zero changes to any other module.

---

## Performance Notes

- Higher timeframes (M5, H1, D1) are pre-computed **once** at load time via `pd.DataFrame.resample()`.
- `get_window()` uses `searchsorted()` (O log n) — no full-frame copies during playback.
- Tested on 2M+ row DataFrames without noticeable playback lag.

---

## Roadmap

| Component | Status |
|---|---|
| Data Loader (HistData) | ✅ |
| MarketDataStore (M1/M5/H1/D1) | ✅ |
| EventBus | ✅ |
| PlaybackController | ✅ |
| PatternDetector (pluggable strategy) | ✅ |
| TradeEngine (placeholder) | ✅ |
| ChartRenderer (3 stacked subplots) | ✅ |
| Streamlit UI | ✅ |
| MTF Strategy (H1 trend + M5 momentum + M1 entry) | ✅ |
| Candlestick pattern recognition | ✅ (via TA-Lib) |
| Support/Resistance detection | 🔜 |
| ML model integration | 🔜 |
| Strategy Engine | 🔜 |
| Risk Manager | 🔜 |
| Performance Analytics | 🔜 |
| Journal System | 🔜 |
| Secondary symbol feeds (DXY, Gold) | 🔜 |
| Database storage | 🔜 |