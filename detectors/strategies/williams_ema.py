"""
detectors/strategies/williams_ema.py
------------------------------------
Williams %R crossover + EMA 9/21 trend strategy (M5 only).

Rules:
  LONG:  Williams %R crosses above -80 + EMA9 > EMA21
  SHORT: Williams %R crosses below -20 + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. Williams %R EMA signals disabled.")


class WilliamsEMAStrategy(BaseStrategy):
    """Williams %R crossover + EMA 9/21 trend confirmation (M5)."""

    name = "williams_ema"

    def __init__(
        self,
        williams_period: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
        oscillator_lookback: int = 5,
    ) -> None:
        self.williams_period = williams_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.oscillator_lookback = oscillator_lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.williams_period, self.ema_slow) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Williams %R returns values from -100 to 0
        willr = talib.WILLR(high, low, close, timeperiod=self.williams_period)

        # EMAs
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # Cross above -80 (within lookback window)
        lb = self.oscillator_lookback
        willr_cross_above_neg80 = bool(np.any(
            (willr[-lb - 1 : -1] <= -80.0) &
            (willr[-lb:] > -80.0)
        ))
        # Cross below -20 (within lookback window)
        willr_cross_below_neg20 = bool(np.any(
            (willr[-lb - 1 : -1] >= -20.0) &
            (willr[-lb:] < -20.0)
        ))

        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # Long: Williams %R crosses above -80 + uptrend
        if willr_cross_above_neg80 and uptrend:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "willr_prev": willr[-2],
                    "willr_now": willr[-1],
                    "ema_fast": ema_f[-1],
                    "ema_slow": ema_s[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: Williams %R crosses below -20 + downtrend
        elif willr_cross_below_neg20 and downtrend:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "willr_prev": willr[-2],
                    "willr_now": willr[-1],
                    "ema_fast": ema_f[-1],
                    "ema_slow": ema_s[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
