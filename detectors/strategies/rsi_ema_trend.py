"""
detectors/strategies/rsi_ema_trend.py
--------------------------------------
RSI bounce + EMA 50 trend strategy (M5 only).

Rules:
  LONG:  RSI bounces from 40 level + price > EMA 50
  SHORT: RSI bounces from 60 level + price < EMA 50
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
    logger.warning("TA-Lib not installed. RSI EMA Trend signals disabled.")


class RSIEMATrendStrategy(BaseStrategy):
    """RSI bounce off support/resistance + EMA 50 trend filter (M5)."""

    name = "rsi_ema_trend"

    def __init__(
        self,
        rsi_period: int = 14,
        ema_period: int = 50,
        bounce_level_long: float = 40.0,
        bounce_level_short: float = 60.0,
        bounce_tolerance: float = 5.0,
        oscillator_lookback: int = 5,
    ) -> None:
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.bounce_level_long = bounce_level_long
        self.bounce_level_short = bounce_level_short
        self.bounce_tolerance = bounce_tolerance
        self.oscillator_lookback = oscillator_lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.rsi_period, self.ema_period) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)

        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        ema50 = talib.EMA(close, timeperiod=self.ema_period)

        # Check bounce: RSI was near/below level within lookback window and is now rising
        lb = self.oscillator_lookback
        rsi_now = rsi[-1]
        price_now = close[-1]
        ema_now = ema50[-1]
        rsi_was_low_recent = bool(np.any(rsi[-lb - 1 : -1] <= self.bounce_level_long + self.bounce_tolerance))
        rsi_was_high_recent = bool(np.any(rsi[-lb - 1 : -1] >= self.bounce_level_short - self.bounce_tolerance))
        rsi_now_bouncing = rsi_now > rsi[-2]
        rsi_now_falling = rsi_now < rsi[-2]

        # Long: RSI was near/below 40, now bouncing up + price > EMA50
        if rsi_was_low_recent and rsi_now_bouncing and price_now > ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "rsi_prev": rsi_prev,
                    "rsi_now": rsi_now,
                    "ema50": ema_now,
                    "price": price_now,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: RSI was near/above 60, now falling + price < EMA50
        elif rsi_was_high_recent and rsi_now_falling and price_now < ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "rsi_prev": rsi_prev,
                    "rsi_now": rsi_now,
                    "ema50": ema_now,
                    "price": price_now,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
