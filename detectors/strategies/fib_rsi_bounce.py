"""
detectors/strategies/fib_rsi_bounce.py
---------------------------------------
Fibonacci 61.8% retracement bounce + RSI extreme confirmation.

Rules:
  Find swing high/low over lookback period.
  LONG:  price at 61.8% retracement + RSI < 35
  SHORT: price at 61.8% retracement + RSI > 65
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
    logger.warning("TA-Lib not installed. FibRsiBounce strategy signals disabled.")


class FibRsiBounceStrategy(BaseStrategy):
    """Fibonacci 61.8% bounce with RSI extreme confirmation."""

    name = "fib_rsi_bounce"

    def __init__(
        self,
        lookback: int = 50,
        rsi_period: int = 14,
        rsi_long: float = 35.0,
        rsi_short: float = 65.0,
        tolerance_pct: float = 0.1,
    ) -> None:
        self.lookback = lookback
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.tolerance_pct = tolerance_pct

    def _find_swing_points(self, high: np.ndarray, low: np.ndarray) -> tuple[float, float]:
        """Find swing high and swing low over the lookback window."""
        lookback = min(self.lookback, len(high))
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]
        return float(recent_high.max()), float(recent_low.min())

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.lookback + self.rsi_period:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        if np.isnan(rsi[-1]):
            return detected

        # --- Fibonacci levels ---
        swing_high, swing_low = self._find_swing_points(high, low)
        swing_range = swing_high - swing_low

        if swing_range <= 0:
            return detected

        # 61.8% retracement level
        fib_618 = swing_high - 0.618 * swing_range

        # --- Check if price is at 61.8% level ---
        current_close = close[-1]
        distance_pct = abs(current_close - fib_618) / fib_618 * 100

        if distance_pct > self.tolerance_pct:
            return detected

        # --- Generate signals ---
        if current_close >= fib_618 and rsi[-1] < self.rsi_long:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "fib_618": float(fib_618),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif current_close <= fib_618 and rsi[-1] > self.rsi_short:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "swing_high": swing_high,
                    "swing_low": swing_low,
                    "fib_618": float(fib_618),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
