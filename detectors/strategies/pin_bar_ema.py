"""
detectors/strategies/pin_bar_ema.py
------------------------------------
Pin bar pattern + EMA 50 trend filter (M5 only).

Rules:
  LONG:  Bullish pin bar (long lower wick > 2x body) + price > EMA50
  SHORT: Bearish pin bar (long upper wick > 2x body) + price < EMA50
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
    logger.warning("TA-Lib not installed. PinBarEma strategy signals disabled.")


class PinBarEmaStrategy(BaseStrategy):
    """Pin bar pattern confirmed by EMA 50 trend."""

    name = "pin_bar_ema"

    def __init__(
        self,
        ema_trend: int = 50,
        wick_ratio: float = 2.0,
    ) -> None:
        self.ema_trend = ema_trend
        self.wick_ratio = wick_ratio

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.ema_trend + 1:
            return detected

        close = window["close"].values.astype(np.float64)
        open_ = window["open"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Indicators
        ema = talib.EMA(close, timeperiod=self.ema_trend)

        # Current candle
        c_o = close[-1] - open_[-1]
        body = abs(c_o)
        range_ = high[-1] - low[-1]
        if range_ == 0 or body == 0:
            return detected

        upper_wick = high[-1] - max(open_[-1], close[-1])
        lower_wick = min(open_[-1], close[-1]) - low[-1]

        # Bullish pin bar: long lower wick
        is_bullish_pin = lower_wick > self.wick_ratio * body
        # Bearish pin bar: long upper wick
        is_bearish_pin = upper_wick > self.wick_ratio * body

        price = close[-1]
        ema_val = ema[-1]

        if is_bullish_pin and price > ema_val:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=min(1.0, lower_wick / (self.wick_ratio * body)),
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema50": float(ema_val),
                    "lower_wick": float(lower_wick),
                    "body": float(body),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif is_bearish_pin and price < ema_val:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=min(1.0, upper_wick / (self.wick_ratio * body)),
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema50": float(ema_val),
                    "upper_wick": float(upper_wick),
                    "body": float(body),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
