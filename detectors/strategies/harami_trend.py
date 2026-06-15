"""
detectors/strategies/harami_trend.py
-------------------------------------
Harami candlestick pattern + EMA 9/21 + RSI (M5 only).

Rules:
  LONG:  Bullish harami + EMA9 > EMA21 + RSI < 40
  SHORT: Bearish harami + EMA9 < EMA21 + RSI > 60
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
    logger.warning("TA-Lib not installed. HaramiTrend strategy signals disabled.")


class HaramiTrendStrategy(BaseStrategy):
    """Harami pattern confirmed by EMA 9/21 and RSI."""

    name = "harami_trend"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < max(self.ema_slow, self.rsi_period) + 2:
            return detected

        close = window["close"].values.astype(np.float64)

        # Indicators
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        # TA-Lib harami pattern (+100 bullish, -100 bearish)
        harami = talib.CDLHARAMI(
            window["open"].values.astype(np.float64),
            window["high"].values.astype(np.float64),
            window["low"].values.astype(np.float64),
            close,
        )

        last_harami = harami[-1]
        ema_f_val = ema_f[-1]
        ema_s_val = ema_s[-1]
        rsi_val = rsi[-1]

        if last_harami > 0 and ema_f_val > ema_s_val and rsi_val < 40:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema_fast": float(ema_f_val),
                    "ema_slow": float(ema_s_val),
                    "rsi": float(rsi_val),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif last_harami < 0 and ema_f_val < ema_s_val and rsi_val > 60:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema_fast": float(ema_f_val),
                    "ema_slow": float(ema_s_val),
                    "rsi": float(rsi_val),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
