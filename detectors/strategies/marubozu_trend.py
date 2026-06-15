"""
detectors/strategies/marubozu_trend.py
---------------------------------------
Marubozu candle pattern + EMA 9/21 trend (M5 only).

Marubozu = body is a large portion of the total range (little/no wicks).

Rules:
  LONG:  Bullish marubozu (body > 80% of range) + EMA9 > EMA21
  SHORT: Bearish marubozu (body > 80% of range) + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. MarubozuTrend strategy signals disabled.")


class MarubozuTrendStrategy(BaseStrategy):
    """Marubozu candle pattern confirmed by EMA 9/21 trend."""

    name = "marubozu_trend"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        body_ratio: float = 0.8,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.body_ratio = body_ratio

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.ema_slow + 1:
            return detected

        close = window["close"].values.astype(np.float64)
        open_ = window["open"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Indicators
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # Current candle analysis
        body = abs(close[-1] - open_[-1])
        range_ = high[-1] - low[-1]

        if range_ == 0:
            return detected

        body_pct = body / range_

        ema_f_val = ema_f[-1]
        ema_s_val = ema_s[-1]

        if body_pct >= self.body_ratio:
            if close[-1] > open_[-1] and ema_f_val > ema_s_val:
                # Bullish marubozu
                detected.append(PatternSignal(
                    name=f"{self.name}_LONG",
                    start_time=window.index[-1],
                    end_time=window.index[-1],
                    confidence=body_pct,
                    metadata={
                        "strategy": self.name,
                        "direction": "LONG",
                        "ema_fast": float(ema_f_val),
                        "ema_slow": float(ema_s_val),
                        "body_ratio": float(body_pct),
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

            elif close[-1] < open_[-1] and ema_f_val < ema_s_val:
                # Bearish marubozu
                detected.append(PatternSignal(
                    name=f"{self.name}_SHORT",
                    start_time=window.index[-1],
                    end_time=window.index[-1],
                    confidence=body_pct,
                    metadata={
                        "strategy": self.name,
                        "direction": "SHORT",
                        "ema_fast": float(ema_f_val),
                        "ema_slow": float(ema_s_val),
                        "body_ratio": float(body_pct),
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
