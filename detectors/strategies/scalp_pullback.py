"""
detectors/strategies/scalp_pullback.py
--------------------------------------
ADX strong trend + RSI pullback (M5 only) — Group 12 Hybrid.

Rules:
  ADX > 30 confirms strong trend + EMA9 > EMA21 for uptrend direction.
  LONG:  RSI pulls back to 40 in uptrend
  SHORT: RSI pulls back to 60 in downtrend
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
    logger.warning("TA-Lib not installed. Scalp Pullback signals disabled.")


class ScalpPullbackStrategy(BaseStrategy):
    """ADX strong trend + RSI pullback on M5."""

    name = "scalp_pullback"

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 30.0,
        rsi_period: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_pullback_long: float = 40.0,
        rsi_pullback_short: float = 60.0,
        rsi_tolerance: float = 5.0,
    ) -> None:
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
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
        min_bars = max(self.adx_period, self.ema_slow, self.rsi_period) + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        current_adx = adx[-1]
        current_rsi = rsi[-1]
        ema_fast_val = ema_f[-1]
        ema_slow_val = ema_s[-1]

        # Must have strong trend confirmed by ADX
        if current_adx < self.adx_threshold:
            return detected

        # Long: strong uptrend (EMA9 > EMA21) + RSI pulls back to 40
        if ema_fast_val > ema_slow_val:
            if abs(current_rsi - self.rsi_pullback_long) <= self.rsi_tolerance:
                detected.append(PatternSignal(
                    name=f"{self.name}_LONG",
                    start_time=window.index[-1],
                    end_time=window.index[-1],
                    confidence=0.82,
                    metadata={
                        "strategy": self.name,
                        "direction": "LONG",
                        "adx": float(current_adx),
                        "ema_fast": float(ema_fast_val),
                        "ema_slow": float(ema_slow_val),
                        "rsi": float(current_rsi),
                        "pullback_level": self.rsi_pullback_long,
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: strong downtrend (EMA9 < EMA21) + RSI pulls back to 60
        if ema_fast_val < ema_slow_val:
            if abs(current_rsi - self.rsi_pullback_short) <= self.rsi_tolerance:
                detected.append(PatternSignal(
                    name=f"{self.name}_SHORT",
                    start_time=window.index[-1],
                    end_time=window.index[-1],
                    confidence=0.82,
                    metadata={
                        "strategy": self.name,
                        "direction": "SHORT",
                        "adx": float(current_adx),
                        "ema_fast": float(ema_fast_val),
                        "ema_slow": float(ema_slow_val),
                        "rsi": float(current_rsi),
                        "pullback_level": self.rsi_pullback_short,
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
