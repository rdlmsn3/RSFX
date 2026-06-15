"""
detectors/strategies/trend_mean_reversion.py
--------------------------------------------
EMA trend + RSI pullback (M5 only) — Group 12 Hybrid.

Rules:
  LONG:  EMA9 > EMA21 (uptrend) + RSI drops to 40 (pullback in uptrend)
  SHORT: EMA9 < EMA21 (downtrend) + RSI rises to 60 (pullback in downtrend)
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
    logger.warning("TA-Lib not installed. Trend Mean Reversion signals disabled.")


class TrendMeanReversionStrategy(BaseStrategy):
    """EMA trend + RSI pullback on M5."""

    name = "trend_mean_reversion"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_pullback_long: float = 40.0,
        rsi_pullback_short: float = 60.0,
        rsi_tolerance: float = 5.0,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_pullback_long = rsi_pullback_long
        self.rsi_pullback_short = rsi_pullback_short
        self.rsi_tolerance = rsi_tolerance

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ema_slow, self.rsi_period) + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)

        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        current_rsi = rsi[-1]
        ema_fast_val = ema_f[-1]
        ema_slow_val = ema_s[-1]

        # Long: uptrend (EMA9 > EMA21) + RSI near pullback level (around 40)
        if ema_fast_val > ema_slow_val:
            if abs(current_rsi - self.rsi_pullback_long) <= self.rsi_tolerance:
                detected.append(PatternSignal(
                    name=f"{self.name}_LONG",
                    start_time=window.index[-1],
                    end_time=window.index[-1],
                    confidence=0.80,
                    metadata={
                        "strategy": self.name,
                        "direction": "LONG",
                        "ema_fast": float(ema_fast_val),
                        "ema_slow": float(ema_slow_val),
                        "rsi": float(current_rsi),
                        "pullback_level": self.rsi_pullback_long,
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: downtrend (EMA9 < EMA21) + RSI near pullback level (around 60)
        if ema_fast_val < ema_slow_val:
            if abs(current_rsi - self.rsi_pullback_short) <= self.rsi_tolerance:
                detected.append(PatternSignal(
                    name=f"{self.name}_SHORT",
                    start_time=window.index[-1],
                    end_time=window.index[-1],
                    confidence=0.80,
                    metadata={
                        "strategy": self.name,
                        "direction": "SHORT",
                        "ema_fast": float(ema_fast_val),
                        "ema_slow": float(ema_slow_val),
                        "rsi": float(current_rsi),
                        "pullback_level": self.rsi_pullback_short,
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
