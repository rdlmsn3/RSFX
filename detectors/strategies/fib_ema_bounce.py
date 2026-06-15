"""
detectors/strategies/fib_ema_bounce.py
---------------------------------------
Fibonacci 61.8% retracement bounce + EMA confirmation.

Rules:
  Find swing high/low over lookback period.
  LONG:  price at 61.8% retracement + EMA9 > EMA21
  SHORT: price at 61.8% retracement + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. FibEmaBounce strategy signals disabled.")


class FibEmaBounceStrategy(BaseStrategy):
    """Fibonacci 61.8% bounce with EMA trend confirmation."""

    name = "fib_ema_bounce"

    def __init__(
        self,
        lookback: int = 50,
        ema_fast: int = 9,
        ema_slow: int = 21,
        tolerance_pct: float = 0.1,
    ) -> None:
        self.lookback = lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
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
        if window is None or not TA_AVAILABLE or len(window) < self.lookback + self.ema_slow:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]):
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

        # Determine direction based on swing context
        # If price came from high (downtrend) → expect bounce at 61.8% = LONG
        # If price came from low (uptrend) → expect rejection at 61.8% = SHORT
        # Use EMA to determine context: price above fib but below swing high suggests uptrend pullback
        in_uptrend = ema_f[-1] > ema_s[-1]

        # --- Generate signals ---
        # LONG: bounce from 61.8% support in uptrend (pullback buy)
        if in_uptrend and current_close >= fib_618:
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
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # SHORT: rejection at 61.8% resistance in downtrend (pullback sell)
        elif not in_uptrend and current_close <= fib_618:
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
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
