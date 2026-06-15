"""
detectors/strategies/ma_envelope_bounce.py
-------------------------------------------
MA envelope bounce + RSI filter (M5 only).

Envelope = EMA20 ± 1%.

Rules:
  LONG:  Price touches lower envelope + RSI < 40
  SHORT: Price touches upper envelope + RSI > 60
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .base import BaseStrategy
from detectors.signal import PatternSignal

logger = logging.getLogger(__name__)

try:
    import talib
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("TA-Lib not installed. MA Envelope Bounce signals disabled.")


class MAEnvelopeBounceStrategy(BaseStrategy):
    """MA envelope bounce with RSI filter (M5)."""

    name = "ma_envelope_bounce"

    def __init__(
        self,
        ema_period: int = 20,
        envelope_pct: float = 0.01,
        rsi_period: int = 14,
        rsi_long_level: float = 40.0,
        rsi_short_level: float = 60.0,
    ) -> None:
        self.ema_period = ema_period
        self.envelope_pct = envelope_pct
        self.rsi_period = rsi_period
        self.rsi_long_level = rsi_long_level
        self.rsi_short_level = rsi_short_level

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ema_period, self.rsi_period) + 10
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)

        # Envelope
        ema20 = talib.EMA(close, timeperiod=self.ema_period)
        upper_env = ema20 * (1.0 + self.envelope_pct)
        lower_env = ema20 * (1.0 - self.envelope_pct)

        # RSI
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        rsi_now = rsi[-1]
        price_close = close[-1]
        price_low = low[-1]
        price_high = high[-1]

        # Touch lower envelope: low touches or goes below lower band
        touches_lower = price_low <= lower_env[-1]
        # Touch upper envelope: high touches or goes above upper band
        touches_upper = price_high >= upper_env[-1]

        # Long: price touches lower envelope + RSI < 40
        if touches_lower and rsi_now < self.rsi_long_level:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "rsi": float(rsi_now),
                    "lower_env": float(lower_env[-1]),
                    "ema20": float(ema20[-1]),
                    "price": float(price_close),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: price touches upper envelope + RSI > 60
        elif touches_upper and rsi_now > self.rsi_short_level:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "rsi": float(rsi_now),
                    "upper_env": float(upper_env[-1]),
                    "ema20": float(ema20[-1]),
                    "price": float(price_close),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
