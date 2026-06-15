"""
detectors/strategies/adx_rsi_ema.py
------------------------------------
ADX trend strength + RSI momentum + EMA alignment (M5 only).

Rules:
  LONG:  ADX > 25 (trending) + RSI > 50 + EMA9 > EMA21
  SHORT: ADX > 25 (trending) + RSI < 50 + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. AdxRsiEma strategy signals disabled.")


class AdxRsiEmaStrategy(BaseStrategy):
    """ADX trend filter + RSI momentum + EMA alignment confluence."""

    name = "adx_rsi_ema"

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = max(self.adx_period + self.adx_period, self.ema_slow) + 1
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # --- Trend strength condition ---
        trending = adx[-1] > self.adx_threshold

        # --- RSI conditions ---
        rsi_bullish = rsi[-1] > 50
        rsi_bearish = rsi[-1] < 50

        # --- EMA alignment conditions ---
        ema_bullish = ema_f[-1] > ema_s[-1]
        ema_bearish = ema_f[-1] < ema_s[-1]

        # --- Generate signals ---
        if trending and rsi_bullish and ema_bullish:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "adx": float(adx[-1]),
                    "rsi": float(rsi[-1]),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s, ADX=%.1f)", current_timestamp, self.name, adx[-1])

        elif trending and rsi_bearish and ema_bearish:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "adx": float(adx[-1]),
                    "rsi": float(rsi[-1]),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s, ADX=%.1f)", current_timestamp, self.name, adx[-1])

        return detected
