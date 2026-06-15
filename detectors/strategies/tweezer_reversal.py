"""
detectors/strategies/tweezer_reversal.py
-----------------------------------------
Tweezer top/bottom pattern + EMA 9/21 trend (M5 only).

Rules:
  LONG:  Tweezer bottom (two candles with same low) + EMA9 > EMA21
  SHORT: Tweezer top (two candles with same high) + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. TweezerReversal strategy signals disabled.")


class TweezerReversalStrategy(BaseStrategy):
    """Tweezer top/bottom confirmed by EMA 9/21 trend."""

    name = "tweezer_reversal"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        tolerance_pct: float = 0.01,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.tolerance_pct = tolerance_pct

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.ema_slow + 2:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Indicators
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        ema_f_val = ema_f[-1]
        ema_s_val = ema_s[-1]

        # Tweezer bottom: two consecutive candles with approximately the same low
        low_diff = abs(low[-1] - low[-2])
        avg_price = (close[-1] + close[-2]) / 2.0
        low_within_tolerance = low_diff <= (self.tolerance_pct / 100.0) * avg_price

        # Tweezer top: two consecutive candles with approximately the same high
        high_diff = abs(high[-1] - high[-2])
        high_within_tolerance = high_diff <= (self.tolerance_pct / 100.0) * avg_price

        # Tweezer bottom: bullish confirmation (second candle closes higher)
        is_tweezer_bottom = low_within_tolerance and (close[-1] > close[-2])
        # Tweezer top: bearish confirmation (second candle closes lower)
        is_tweezer_top = high_within_tolerance and (close[-1] < close[-2])

        if is_tweezer_bottom and ema_f_val > ema_s_val:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema_fast": float(ema_f_val),
                    "ema_slow": float(ema_s_val),
                    "low_diff": float(low_diff),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif is_tweezer_top and ema_f_val < ema_s_val:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema_fast": float(ema_f_val),
                    "ema_slow": float(ema_s_val),
                    "high_diff": float(high_diff),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
