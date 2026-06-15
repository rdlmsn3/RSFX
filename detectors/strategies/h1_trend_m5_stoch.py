"""
detectors/strategies/h1_trend_m5_stoch.py
------------------------------------------
H1 EMA trend + M5 Stochastic cross strategy.

Layer 1 (H1): Trend bias — EMA9/EMA21 direction
Layer 2 (M5): Entry trigger — Stochastic crosses key levels

LONG:  H1 EMA9 > EMA21 + M5 Stoch crosses above 20
SHORT: H1 EMA9 < EMA21 + M5 Stoch crosses below 80
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
    logger.warning("TA-Lib not installed. H1TrendM5Stoch strategy signals disabled.")


class H1TrendM5StochStrategy(BaseStrategy):
    """H1 EMA trend filter + M5 Stochastic crossover entry."""

    name = "h1_trend_m5_stoch"

    def __init__(
        self,
        h1_ema_fast: int = 9,
        h1_ema_slow: int = 21,
        m5_stoch_k: int = 14,
        m5_stoch_d: int = 3,
        m5_stoch_slowing: int = 3,
        m5_stoch_oversold: float = 20.0,
        m5_stoch_overbought: float = 80.0,
    ) -> None:
        self.h1_ema_fast = h1_ema_fast
        self.h1_ema_slow = h1_ema_slow
        self.m5_stoch_k = m5_stoch_k
        self.m5_stoch_d = m5_stoch_d
        self.m5_stoch_slowing = m5_stoch_slowing
        self.m5_stoch_oversold = m5_stoch_oversold
        self.m5_stoch_overbought = m5_stoch_overbought

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []

        if not TA_AVAILABLE:
            return detected

        # --- Check required TFs ---
        if "M5" not in windows or windows["M5"] is None or len(windows["M5"]) < 5:
            return detected
        if "H1" not in windows or windows["H1"] is None or len(windows["H1"]) < self.h1_ema_slow + 1:
            return detected

        h1 = windows["H1"]
        m5 = windows["M5"]

        # ==============================================================
        # Layer 1: H1 Trend Bias
        # ==============================================================
        h1_close = h1["close"].values.astype(np.float64)
        h1_ema_f = talib.EMA(h1_close, timeperiod=self.h1_ema_fast)
        h1_ema_s = talib.EMA(h1_close, timeperiod=self.h1_ema_slow)

        if np.isnan(h1_ema_f[-1]) or np.isnan(h1_ema_s[-1]):
            return detected

        h1_uptrend = h1_ema_f[-1] > h1_ema_s[-1]
        h1_downtrend = h1_ema_f[-1] < h1_ema_s[-1]

        if not h1_uptrend and not h1_downtrend:
            return detected

        # ==============================================================
        # Layer 2: M5 Stochastic Cross
        # ==============================================================
        m5_high = m5["high"].values.astype(np.float64)
        m5_low = m5["low"].values.astype(np.float64)
        m5_close = m5["close"].values.astype(np.float64)

        m5_slowk, m5_slowd = talib.STOCH(
            m5_high, m5_low, m5_close,
            fastk_period=self.m5_stoch_k,
            slowk_period=self.m5_stoch_slowing,
            slowk_matype=0,
            slowd_period=self.m5_stoch_d,
            slowd_matype=0,
        )

        if len(m5_slowk) < 2 or np.isnan(m5_slowk[-1]) or np.isnan(m5_slowd[-1]):
            return detected

        # Stoch crosses above oversold level (20): K crosses above D near oversold
        stoch_cross_up = (
            m5_slowk[-2] < m5_slowd[-2]
            and m5_slowk[-1] > m5_slowd[-1]
            and m5_slowk[-1] < self.m5_stoch_oversold + 10  # near oversold zone
        )
        # Stoch crosses below overbought level (80): K crosses below D near overbought
        stoch_cross_down = (
            m5_slowk[-2] > m5_slowd[-2]
            and m5_slowk[-1] < m5_slowd[-1]
            and m5_slowk[-1] > self.m5_stoch_overbought - 10  # near overbought zone
        )

        # ==============================================================
        # Confluence
        # ==============================================================
        if h1_uptrend and stoch_cross_up:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=m5.index[-1],
                end_time=m5.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "h1_ema_fast": h1_ema_f[-1],
                    "h1_ema_slow": h1_ema_s[-1],
                    "m5_stoch_k": m5_slowk[-1],
                    "m5_stoch_d": m5_slowd[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif h1_downtrend and stoch_cross_down:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=m5.index[-1],
                end_time=m5.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "h1_ema_fast": h1_ema_f[-1],
                    "h1_ema_slow": h1_ema_s[-1],
                    "m5_stoch_k": m5_slowk[-1],
                    "m5_stoch_d": m5_slowd[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
