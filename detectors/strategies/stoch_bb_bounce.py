"""
detectors/strategies/stoch_bb_bounce.py
---------------------------------------
Bollinger Band bounce + Stochastic crossover (M5 only).

Rules:
  LONG:  Price touches lower BB + Stochastic crosses above 20
  SHORT: Price touches upper BB + Stochastic crosses below 80
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
    logger.warning("TA-Lib not installed. StochBbBounce strategy signals disabled.")


class StochBbBounceStrategy(BaseStrategy):
    """BB band touch + Stochastic crossover for mean-reversion entries."""

    name = "stoch_bb_bounce"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 5,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_smooth = stoch_smooth
        self.oversold = oversold
        self.overbought = overbought

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.bb_period + 1:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
        )
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
        )

        # --- Touch detection ---
        touch_lower = close[-1] <= lower[-1]
        touch_upper = close[-1] >= upper[-1]

        # --- Stochastic crossover detection ---
        stoch_cross_above = (slowk[-2] <= self.oversold) and (slowk[-1] > self.oversold)
        stoch_cross_below = (slowk[-2] >= self.overbought) and (slowk[-1] < self.overbought)

        # --- Generate signals ---
        if touch_lower and stoch_cross_above:
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
                    "close": float(close[-1]),
                    "stoch_k": float(slowk[-1]),
                    "stoch_d": float(slowd[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif touch_upper and stoch_cross_below:
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
                    "close": float(close[-1]),
                    "stoch_k": float(slowk[-1]),
                    "stoch_d": float(slowd[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
