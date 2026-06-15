"""
detectors/strategies/engulfing_ema.py
--------------------------------------
Engulfing candlestick pattern + EMA 9/21 trend (M5 only).

Rules:
  LONG:  Bullish engulfing + EMA9 > EMA21
  SHORT: Bearish engulfing + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. EngulfingEma strategy signals disabled.")


class EngulfingEmaStrategy(BaseStrategy):
    """Engulfing pattern confirmed by EMA 9/21 trend."""

    name = "engulfing_ema"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

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

        # Indicators
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # TA-Lib engulfing pattern (returns +100 bullish, -100 bearish, 0 none)
        engulfing = talib.CDLENGULFING(
            window["open"].values.astype(np.float64),
            window["high"].values.astype(np.float64),
            window["low"].values.astype(np.float64),
            close,
        )

        last_engulf = engulfing[-1]
        ema_f_val = ema_f[-1]
        ema_s_val = ema_s[-1]

        if last_engulf > 0 and ema_f_val > ema_s_val:
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
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif last_engulf < 0 and ema_f_val < ema_s_val:
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
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
