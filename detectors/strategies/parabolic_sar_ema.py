"""
detectors/strategies/parabolic_sar_ema.py
-----------------------------------------
Parabolic SAR + EMA 9/21 trend following strategy (M5 only).

Rules:
  LONG:  SAR flips below price + EMA9 > EMA21
  SHORT: SAR flips above price + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. ParabolicSarEma strategy signals disabled.")


class ParabolicSarEmaStrategy(BaseStrategy):
    """Parabolic SAR flip + EMA 9/21 trend filter (M5 only)."""

    name = "parabolic_sar_ema"

    def __init__(
        self,
        sar_af: float = 0.02,
        sar_max: float = 0.2,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.sar_af = sar_af
        self.sar_max = sar_max
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
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Parabolic SAR
        sar = talib.SAR(high, low, acceleration=self.sar_af, maximum=self.sar_max)

        # EMAs
        ema_fast = talib.EMA(close, timeperiod=self.ema_fast)
        ema_slow = talib.EMA(close, timeperiod=self.ema_slow)

        # Need at least 2 bars to detect flip
        if len(sar) < 2 or np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
            return detected

        current_close = close[-1]
        prev_close = close[-2]

        # SAR flip detection:
        # Bullish flip: SAR was above price (bearish), now SAR is below price (bullish)
        sar_flip_bullish = (sar[-2] > prev_close) and (sar[-1] < current_close)
        # Bearish flip: SAR was below price (bullish), now SAR is above price (bearish)
        sar_flip_bearish = (sar[-2] < prev_close) and (sar[-1] > current_close)

        ema_uptrend = ema_fast[-1] > ema_slow[-1]
        ema_downtrend = ema_fast[-1] < ema_slow[-1]

        # --- LONG ---
        if sar_flip_bullish and ema_uptrend:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "sar": sar[-1],
                    "ema_fast": ema_fast[-1],
                    "ema_slow": ema_slow[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # --- SHORT ---
        elif sar_flip_bearish and ema_downtrend:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "sar": sar[-1],
                    "ema_fast": ema_fast[-1],
                    "ema_slow": ema_slow[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
