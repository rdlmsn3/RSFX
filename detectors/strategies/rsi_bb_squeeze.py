"""
detectors/strategies/rsi_bb_squeeze.py
---------------------------------------
RSI bounce + Bollinger Band squeeze expansion (M5 only).

Setup: BB width < 20-period average (squeeze detected).
Rules:
  LONG:  RSI bounces from 35 + BB expands + close > middle BB
  SHORT: RSI from 65 + BB expands + close < middle BB
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
    logger.warning("TA-Lib not installed. RsiBbSqueeze strategy signals disabled.")


class RsiBbSqueezeStrategy(BaseStrategy):
    """RSI mean-reversion bounce during BB squeeze expansion."""

    name = "rsi_bb_squeeze"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_width_avg_period: int = 20,
        rsi_period: int = 14,
        rsi_long_threshold: float = 35.0,
        rsi_short_threshold: float = 65.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_avg_period = bb_width_avg_period
        self.rsi_period = rsi_period
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold

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
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        # BB width = (upper - lower) / middle
        bb_width = np.where(middle != 0, (upper - lower) / middle, 0.0)
        bb_width_avg = talib.SMA(bb_width, timeperiod=self.bb_width_avg_period)

        # --- Squeeze + expansion detection ---
        # Previous bar was in squeeze, current bar shows expansion
        was_squeeze = bb_width[-2] < bb_width_avg[-2]
        is_expanding = bb_width[-1] > bb_width[-2]

        # --- RSI bounce detection ---
        # Long: RSI was below 35 and now bouncing above
        rsi_bounce_long = (rsi[-2] < self.rsi_long_threshold) and (rsi[-1] >= self.rsi_long_threshold)
        # Short: RSI was above 65 and now dropping below
        rsi_bounce_short = (rsi[-2] > self.rsi_short_threshold) and (rsi[-1] <= self.rsi_short_threshold)

        # --- Generate signals ---
        if was_squeeze and is_expanding and rsi_bounce_long and close[-1] > middle[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.85,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width[-1]),
                    "bb_width_avg": float(bb_width_avg[-1]),
                    "was_squeeze": was_squeeze,
                    "rsi": float(rsi[-1]),
                    "close": float(close[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s, squeeze→expansion)", current_timestamp, self.name)

        elif was_squeeze and is_expanding and rsi_bounce_short and close[-1] < middle[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.85,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width[-1]),
                    "bb_width_avg": float(bb_width_avg[-1]),
                    "was_squeeze": was_squeeze,
                    "rsi": float(rsi[-1]),
                    "close": float(close[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s, squeeze→expansion)", current_timestamp, self.name)

        return detected
