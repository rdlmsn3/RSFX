"""
detectors/pattern_detector.py
------------------------------
Strategy-based pattern detection.

Generates Long/Short signals using:
- Trend: 9 EMA vs 21 EMA
- Momentum: Fast Stochastic oversold/overbought
- Candlestick triggers: Engulfing, Hammer, Shooting Star
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

try:
    import talib
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("TA-Lib not installed. Strategy signals disabled.")

from core.event_bus import EventBus
from core.events import MarketTickEvent, PatternDetectedEvent
from core.market_data_store import MarketDataStore

logger = logging.getLogger(__name__)


@dataclass
class PatternSignal:
    name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    confidence: float
    metadata: dict = field(default_factory=dict)


class PatternDetector:
    def __init__(
        self,
        event_bus: EventBus,
        data_store: MarketDataStore,
        symbol: str = "EURUSD",
        lookback: int = 200,
    ) -> None:
        self._bus = event_bus
        self._store = data_store
        self._symbol = symbol
        self._lookback = lookback

        self._signals: list[PatternSignal] = []
        self._bus.subscribe(MarketTickEvent, self._on_market_tick)

        if not TA_AVAILABLE:
            logger.warning("TA-Lib not installed – no signals will be generated.")

        logger.info("Strategy detector initialised (lookback=%d).", lookback)

    @property
    def signals(self) -> list[PatternSignal]:
        return list(self._signals)

    @property
    def signal_count(self) -> int:
        return len(self._signals)

    def reset(self) -> None:
        self._signals.clear()

    # ------------------------------------------------------------------
    # Core strategy logic
    # ------------------------------------------------------------------

    def scan_for_patterns(
        self,
        window: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        """
        Evaluate the strategy on the latest candle and return a signal if conditions match.
        """
        detected = []
        if not TA_AVAILABLE or len(window) < 22:  # need at least 22 for 21 EMA
            return detected

        # Convert to numpy arrays for TA-Lib
        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        open_p = window["open"].values.astype(np.float64)

        # --- 1. Trend: 9 EMA and 21 EMA ---
        ema9 = talib.EMA(close, timeperiod=9)
        ema21 = talib.EMA(close, timeperiod=21)
        uptrend = ema9[-1] > ema21[-1]
        downtrend = ema9[-1] < ema21[-1]

        # --- 2. Fast Stochastic (5,3) ---
        # fastk_period=5, fastd_period=3
        fastk, _ = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3)
        # Check oversold/overbought within last 2 candles (current or previous)
        oversold_last2 = (fastk[-1] < 20) or (fastk[-2] < 20)
        overbought_last2 = (fastk[-1] > 80) or (fastk[-2] > 80)

        # --- 3. Candlestick triggers (only at the current candle) ---
        # Bullish triggers
        bullish_engulfing = talib.CDLENGULFING(open_p, high, low, close)
        hammer = talib.CDLHAMMER(open_p, high, low, close)
        is_bullish = (bullish_engulfing[-1] == 100) or (hammer[-1] == 100)

        # Bearish triggers
        bearish_engulfing = talib.CDLENGULFING(open_p, high, low, close)
        shooting_star = talib.CDLSHOOTINGSTAR(open_p, high, low, close)
        is_bearish = (bearish_engulfing[-1] == -100) or (shooting_star[-1] == -100)

        # --- Generate signals ---
        if uptrend and oversold_last2 and is_bullish:
            detected.append(PatternSignal(
                name="STRATEGY_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata = {
                    "direction": "LONG",
                    "trend": "up" if uptrend else "down",
                    "ema9": ema9[-1],
                    "ema21": ema21[-1],
                    "stoch_fastk": fastk[-1],
                    "stoch_oversold_last2": oversold_last2,
                    "candle_trigger_idx": window.index[-1],   # the candle that fired
                    "pattern": "bullish_engulfing" if bullish_engulfing[-1] == 100 else "hammer"
                }
            ))
            logger.info("LONG signal at %s", current_timestamp)

        elif downtrend and overbought_last2 and is_bearish:
            detected.append(PatternSignal(
                name="STRATEGY_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata = {
                    "direction": "SHORT",
                    "trend": "down",
                    "ema9": ema9[-1],
                    "ema21": ema21[-1],
                    "stoch_fastk": fastk[-1],
                    "stoch_overbought": overbought_last2,
                    "candle_trigger_idx": window.index[-1],   # the candle that fired
                    "pattern": "bearish_candle"
                }
            ))
            logger.info("SHORT signal at %s", current_timestamp)

        return detected

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _on_market_tick(self, event: MarketTickEvent) -> None:
        window = self._store.get_window(
            symbol=event.symbol,
            timeframe="M1",
            current_timestamp=event.timestamp,
            lookback=self._lookback,
        )

        new_signals = self.scan_for_patterns(window, event.timestamp)

        for signal in new_signals:
            self._signals.append(signal)
            self._bus.publish(PatternDetectedEvent(
                pattern_name=signal.name,
                timestamp=signal.end_time,
                confidence=signal.confidence,
                metadata=signal.metadata,
                symbol=event.symbol,
                timeframe="M1",
            ))