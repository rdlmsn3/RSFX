"""
detectors/strategies/breakout_retest.py
---------------------------------------
Key level breakout + retest (M5 only) — Group 12 Hybrid.

Rules:
  LONG:  price breaks above 20-period high, then retests the breakout level
         + EMA9 > EMA21 alignment
  SHORT: price breaks below 20-period low, then retests the breakout level
         + EMA9 < EMA21 alignment
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
    logger.warning("TA-Lib not installed. Breakout Retest signals disabled.")


class BreakoutRetestStrategy(BaseStrategy):
    """Key level breakout + retest on M5."""

    name = "breakout_retest"

    def __init__(
        self,
        donchian_period: int = 20,
        ema_fast: int = 9,
        ema_slow: int = 21,
        retest_tolerance_pct: float = 0.001,
    ) -> None:
        self.donchian_period = donchian_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.retest_tolerance_pct = retest_tolerance_pct

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.donchian_period, self.ema_slow) + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Donchian channel levels (rolling high/low of N periods)
        period = self.donchian_period
        upper_level = talib.MAX(high, timeperiod=period)
        lower_level = talib.MIN(low, timeperiod=period)

        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # Previous bar's breakout level (before current bar retests)
        prev_upper = upper_level[-2]
        prev_lower = lower_level[-2]
        current_close = close[-1]
        current_low = low[-1]
        current_high = high[-1]

        ema_fast_val = ema_f[-1]
        ema_slow_val = ema_s[-1]
        tolerance = self.retest_tolerance_pct * current_close

        # Long: breakout above 20-period high + retest + EMA alignment
        if ema_fast_val > ema_slow_val:
            # Check if a breakout happened recently (current close broke above previous upper)
            broke_above = close[-1] > prev_upper or close[-2] > upper_level[-3]
            # Retest: current low touches near the breakout level
            retesting_up = abs(current_low - prev_upper) <= tolerance or current_low <= prev_upper
            if broke_above and retesting_up:
                detected.append(PatternSignal(
                    name=f"{self.name}_LONG",
                    start_time=window.index[-2],
                    end_time=window.index[-1],
                    confidence=0.82,
                    metadata={
                        "strategy": self.name,
                        "direction": "LONG",
                        "breakout_level": float(prev_upper),
                        "current_low": float(current_low),
                        "ema_fast": float(ema_fast_val),
                        "ema_slow": float(ema_slow_val),
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: breakout below 20-period low + retest + EMA alignment
        if ema_fast_val < ema_slow_val:
            broke_below = close[-1] < prev_lower or close[-2] < lower_level[-3]
            retesting_down = abs(current_high - prev_lower) <= tolerance or current_high >= prev_lower
            if broke_below and retesting_down:
                detected.append(PatternSignal(
                    name=f"{self.name}_SHORT",
                    start_time=window.index[-2],
                    end_time=window.index[-1],
                    confidence=0.82,
                    metadata={
                        "strategy": self.name,
                        "direction": "SHORT",
                        "breakout_level": float(prev_lower),
                        "current_high": float(current_high),
                        "ema_fast": float(ema_fast_val),
                        "ema_slow": float(ema_slow_val),
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
