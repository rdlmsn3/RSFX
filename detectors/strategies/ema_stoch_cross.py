"""
detectors/strategies/ema_stoch_cross.py
----------------------------------------
EMA 9/21 trend + Stochastic 5,3,3 crossover (M5 only).

Rules:
  LONG:  EMA9 > EMA21 (uptrend) + Stochastic crosses above 20 (exit oversold)
  SHORT: EMA9 < EMA21 (downtrend) + Stochastic crosses below 80 (exit overbought)
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
    logger.warning("TA-Lib not installed. EmaStochCross strategy signals disabled.")


class EmaStochCrossStrategy(BaseStrategy):
    """EMA 9/21 trend filter + Stochastic crossover trigger."""

    name = "ema_stoch_cross"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        stoch_k: int = 5,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
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
        if window is None or not TA_AVAILABLE or len(window) < self.ema_slow + 1:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
        )

        # --- Trend filter ---
        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # --- Stochastic crossover ---
        # Cross above oversold: prev K <= oversold AND current K > oversold
        stoch_cross_above = (slowk[-2] <= self.oversold) and (slowk[-1] > self.oversold)
        # Cross below overbought: prev K >= overbought AND current K < overbought
        stoch_cross_below = (slowk[-2] >= self.overbought) and (slowk[-1] < self.overbought)

        # --- Generate signals ---
        if uptrend and stoch_cross_above:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                    "stoch_k": float(slowk[-1]),
                    "stoch_d": float(slowd[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif downtrend and stoch_cross_below:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                    "stoch_k": float(slowk[-1]),
                    "stoch_d": float(slowd[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
