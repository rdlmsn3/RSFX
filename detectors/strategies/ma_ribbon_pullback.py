"""
detectors/strategies/ma_ribbon_pullback.py
-------------------------------------------
EMA ribbon alignment (5,8,13,21) + pullback to EMA13 (M5 only).

Rules:
  LONG:  All EMAs aligned bullishly (5>8>13>21) + price pulls back to EMA13
  SHORT: All EMAs aligned bearishly (5<8<13<21) + price pulls back to EMA13
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
    logger.warning("TA-Lib not installed. MA Ribbon Pullback signals disabled.")


class MARibbonPullbackStrategy(BaseStrategy):
    """EMA ribbon alignment with pullback to EMA13 (M5)."""

    name = "ma_ribbon_pullback"

    def __init__(
        self,
        periods: list[int] | None = None,
        pullback_tolerance: float = 0.001,
    ) -> None:
        self.periods = periods or [5, 8, 13, 21]
        self.pullback_tolerance = pullback_tolerance

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.periods) + 10
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Compute EMAs
        emas = [talib.EMA(close, timeperiod=p) for p in self.periods]

        # Check ribbon alignment (fastest first: 5, 8, 13, 21)
        ema5, ema8, ema13, ema21 = [e[-1] for e in emas]

        bullish_alignment = ema5 > ema8 > ema13 > ema21
        bearish_alignment = ema5 < ema8 < ema13 < ema21

        # Pullback: price low touched or came within tolerance of EMA13
        price_low = low[-1]
        price_close = close[-1]
        ema13_val = emas[2][-1]

        tolerance = ema13_val * self.pullback_tolerance
        touched_ema13 = abs(price_low - ema13_val) <= tolerance

        # Also check previous bars for the pullback
        prev_touched = False
        for i in range(-3, -1):
            if abs(low[i] - emas[2][i]) <= tolerance:
                prev_touched = True
                break

        pullback = touched_ema13 or prev_touched

        # Long: bullish alignment + pullback to EMA13
        if bullish_alignment and pullback and price_close >= ema13_val:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema5": float(ema5),
                    "ema8": float(ema8),
                    "ema13": float(ema13_val),
                    "ema21": float(ema21),
                    "price": float(price_close),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: bearish alignment + pullback to EMA13
        elif bearish_alignment and pullback and price_close <= ema13_val:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema5": float(ema5),
                    "ema8": float(ema8),
                    "ema13": float(ema13_val),
                    "ema21": float(ema21),
                    "price": float(price_close),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
