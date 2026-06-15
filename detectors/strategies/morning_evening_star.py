"""
detectors/strategies/morning_evening_star.py
---------------------------------------------
Morning/Evening star candlestick pattern + EMA 50 (M5 only).

Rules:
  LONG:  Morning star pattern + price > EMA50
  SHORT: Evening star pattern + price < EMA50
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
    logger.warning("TA-Lib not installed. MorningEveningStar strategy signals disabled.")


class MorningEveningStarStrategy(BaseStrategy):
    """Morning/Evening star pattern confirmed by EMA 50 trend."""

    name = "morning_evening_star"

    def __init__(
        self,
        ema_trend: int = 50,
    ) -> None:
        self.ema_trend = ema_trend

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.ema_trend + 3:
            return detected

        close = window["close"].values.astype(np.float64)
        open_ = window["open"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Indicators
        ema = talib.EMA(close, timeperiod=self.ema_trend)

        # TA-Lib morning star (+100) and evening star (-100)
        morning_star = talib.CDLMORNINGSTAR(open_, high, low, close)
        evening_star = talib.CDLEVENINGSTAR(open_, high, low, close)

        ema_val = ema[-1]
        price = close[-1]

        if morning_star[-1] > 0 and price > ema_val:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema50": float(ema_val),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif evening_star[-1] < 0 and price < ema_val:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema50": float(ema_val),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
