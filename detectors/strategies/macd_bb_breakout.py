"""
detectors/strategies/macd_bb_breakout.py
-----------------------------------------
BB squeeze + MACD crossover + breakout (M5 only).

Setup: BB width < 20-period average (squeeze).
Rules:
  LONG:  MACD crosses above signal line + break above upper BB
  SHORT: MACD crosses below signal line + break below lower BB
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
    logger.warning("TA-Lib not installed. MacdBbBreakout strategy signals disabled.")


class MacdBbBreakoutStrategy(BaseStrategy):
    """BB squeeze + MACD crossover + price breakout."""

    name = "macd_bb_breakout"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_width_avg_period: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_avg_period = bb_width_avg_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = self.bb_period + self.bb_width_avg_period + 1
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
        )
        macd, signal, _ = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        # BB width = (upper - lower) / middle
        bb_width = np.where(middle != 0, (upper - lower) / middle, 0.0)
        bb_width_avg = talib.SMA(bb_width, timeperiod=self.bb_width_avg_period)

        # --- Squeeze detection ---
        was_squeeze = bb_width[-2] < bb_width_avg[-2]

        # --- MACD crossover detection ---
        macd_cross_above = (macd[-2] <= signal[-2]) and (macd[-1] > signal[-1])
        macd_cross_below = (macd[-2] >= signal[-2]) and (macd[-1] < signal[-1])

        # --- Breakout detection ---
        broke_above_upper = close[-1] > upper[-1]
        broke_below_lower = close[-1] < lower[-1]

        # --- Generate signals ---
        if was_squeeze and macd_cross_above and broke_above_upper:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.9,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width[-1]),
                    "bb_width_avg": float(bb_width_avg[-1]),
                    "was_squeeze": was_squeeze,
                    "macd": float(macd[-1]),
                    "macd_signal": float(signal[-1]),
                    "close": float(close[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s, squeeze→breakout)", current_timestamp, self.name)

        elif was_squeeze and macd_cross_below and broke_below_lower:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.9,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width[-1]),
                    "bb_width_avg": float(bb_width_avg[-1]),
                    "was_squeeze": was_squeeze,
                    "macd": float(macd[-1]),
                    "macd_signal": float(signal[-1]),
                    "close": float(close[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s, squeeze→breakout)", current_timestamp, self.name)

        return detected
