"""
detectors/strategies/stoch_ema_trend.py
---------------------------------------
Stochastic cross + EMA 9/21 trend strategy (M5 only).

Rules:
  LONG:  Stochastic crosses above 20 + EMA9 > EMA21
  SHORT: Stochastic crosses below 80 + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. Stoch EMA Trend signals disabled.")


class StochEMATrendStrategy(BaseStrategy):
    """Stochastic crossover + EMA 9/21 trend confirmation (M5)."""

    name = "stoch_ema_trend"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        stoch_k: int = 5,
        stoch_d: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        oscillator_lookback: int = 5,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.oversold = oversold
        self.overbought = overbought
        self.oscillator_lookback = oscillator_lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ema_slow, self.stoch_k + self.stoch_d) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # EMAs
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # Stochastic
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_d,
            slowd_period=3,
        )
        # Cross above 20 (within lookback window)
        lb = self.oscillator_lookback
        stoch_cross_above_20 = bool(np.any(
            (slowk[-lb - 1 : -1] <= self.oversold) &
            (slowk[-lb:] > self.oversold)
        ))
        # Cross below 80 (within lookback window)
        stoch_cross_below_80 = bool(np.any(
            (slowk[-lb - 1 : -1] >= self.overbought) &
            (slowk[-lb:] < self.overbought)
        ))

        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # Long: Stoch crosses above 20 + uptrend
        if stoch_cross_above_20 and uptrend:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "stoch_prev": slowk[-2],
                    "stoch_now": slowk[-1],
                    "ema_fast": ema_f[-1],
                    "ema_slow": ema_s[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: Stoch crosses below 80 + downtrend
        elif stoch_cross_below_80 and downtrend:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "stoch_prev": slowk[-2],
                    "stoch_now": slowk[-1],
                    "ema_fast": ema_f[-1],
                    "ema_slow": ema_s[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
