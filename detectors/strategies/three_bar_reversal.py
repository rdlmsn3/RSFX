"""
detectors/strategies/three_bar_reversal.py
-------------------------------------------
Three consecutive bars pattern + RSI confirmation (M5 only).

Rules:
  LONG:  3 higher lows (ascending) + RSI > 50
  SHORT: 3 lower highs (descending) + RSI < 50
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
    logger.warning("TA-Lib not installed. ThreeBarReversal strategy signals disabled.")


class ThreeBarReversalStrategy(BaseStrategy):
    """Three consecutive bars pattern confirmed by RSI."""

    name = "three_bar_reversal"

    def __init__(
        self,
        rsi_period: int = 14,
    ) -> None:
        self.rsi_period = rsi_period

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.rsi_period + 3:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Indicators
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        # Check 3 consecutive bars: bars [-3], [-2], [-1]
        # LONG: 3 higher lows (each low higher than previous)
        three_higher_lows = (low[-2] > low[-3]) and (low[-1] > low[-2])
        # SHORT: 3 lower highs (each high lower than previous)
        three_lower_highs = (high[-2] < high[-3]) and (high[-1] < high[-2])

        rsi_val = rsi[-1]

        if three_higher_lows and rsi_val > 50:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "rsi": float(rsi_val),
                    "low_1": float(low[-3]),
                    "low_2": float(low[-2]),
                    "low_3": float(low[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif three_lower_highs and rsi_val < 50:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "rsi": float(rsi_val),
                    "high_1": float(high[-3]),
                    "high_2": float(high[-2]),
                    "high_3": float(high[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
