"""
detectors/strategies/adx_di_ema.py
-----------------------------------
ADX + DI cross + EMA 50 trend following strategy (M5 only).

Setup: ADX > 25 (trending market)
Rules:
  LONG:  +DI crosses above -DI + price > EMA50
  SHORT: -DI crosses above +DI + price < EMA50
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
    logger.warning("TA-Lib not installed. AdxDiEma strategy signals disabled.")


class AdxDiEmaStrategy(BaseStrategy):
    """ADX trend strength + DI crossover + EMA 50 trend filter (M5 only)."""

    name = "adx_di_ema"

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        ema_trend: int = 50,
    ) -> None:
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ema_trend = ema_trend

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < max(self.adx_period + 2, self.ema_trend + 1):
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # ADX + DI
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

        # EMA 50 for trend filter
        ema50 = talib.EMA(close, timeperiod=self.ema_trend)

        # Need valid values
        if len(adx) < 2 or np.isnan(adx[-1]) or np.isnan(ema50[-1]):
            return detected

        # Setup: ADX > threshold (trending market)
        if adx[-1] < self.adx_threshold:
            return detected

        current_close = close[-1]

        # DI cross detection
        plus_crosses_above_minus = (plus_di[-2] < minus_di[-2]) and (plus_di[-1] > minus_di[-1])
        minus_crosses_above_plus = (minus_di[-2] < plus_di[-2]) and (minus_di[-1] > plus_di[-1])

        # --- LONG ---
        if plus_crosses_above_minus and current_close > ema50[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "adx": adx[-1],
                    "plus_di": plus_di[-1],
                    "minus_di": minus_di[-1],
                    "ema50": ema50[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # --- SHORT ---
        elif minus_crosses_above_plus and current_close < ema50[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "adx": adx[-1],
                    "plus_di": plus_di[-1],
                    "minus_di": minus_di[-1],
                    "ema50": ema50[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
