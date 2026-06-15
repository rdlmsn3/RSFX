"""
detectors/strategies/asian_range_breakout.py
---------------------------------------------
Asian session range breakout + EMA confirmation.

Rules:
  Asian session: hours 0-8.
  LONG:  price breaks above Asian session high + EMA9 > EMA21
  SHORT: price breaks below Asian session low  + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. AsianRangeBreakout strategy signals disabled.")


class AsianRangeBreakoutStrategy(BaseStrategy):
    """Asian session range breakout with EMA trend confirmation."""

    name = "asian_range_breakout"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.ema_slow + 5:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]):
            return detected

        # --- Determine Asian session range ---
        # Asian session: hours 0-8
        asian_mask = (window.index.hour >= 0) & (window.index.hour <= 8)

        if asian_mask.sum() < 2:
            return detected

        asian_high = high[asian_mask.values].max()
        asian_low = low[asian_mask.values].min()

        # --- Breakout detection ---
        current_close = close[-1]
        current_high = high[-1]
        current_low = low[-1]

        # Break above Asian high
        break_above = current_high > asian_high and current_close > asian_high
        # Break below Asian low
        break_below = current_low < asian_low and current_close < asian_low

        # --- Generate signals ---
        if break_above and ema_f[-1] > ema_s[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "asian_high": float(asian_high),
                    "asian_low": float(asian_low),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif break_below and ema_f[-1] < ema_s[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "asian_high": float(asian_high),
                    "asian_low": float(asian_low),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
