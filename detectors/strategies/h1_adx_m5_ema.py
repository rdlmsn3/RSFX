"""
detectors/strategies/h1_adx_m5_ema.py
--------------------------------------
H1 ADX trend + M5 EMA cross strategy.

Layer 1 (H1): ADX trend strength + DI direction
Layer 2 (M5): Entry trigger — EMA9/EMA21 crossover

LONG:  H1 ADX > 25 + H1 +DI > -DI + M5 EMA9 crosses above EMA21
SHORT: H1 ADX > 25 + H1 -DI > +DI + M5 EMA9 crosses below EMA21
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
    logger.warning("TA-Lib not installed. H1AdxM5Ema strategy signals disabled.")


class H1AdxM5EmaStrategy(BaseStrategy):
    """H1 ADX trend filter + M5 EMA crossover entry."""

    name = "h1_adx_m5_ema"

    def __init__(
        self,
        h1_adx_period: int = 14,
        h1_adx_threshold: float = 25.0,
        m5_ema_fast: int = 9,
        m5_ema_slow: int = 21,
    ) -> None:
        self.h1_adx_period = h1_adx_period
        self.h1_adx_threshold = h1_adx_threshold
        self.m5_ema_fast = m5_ema_fast
        self.m5_ema_slow = m5_ema_slow

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
        if "H1" not in windows or windows["H1"] is None or len(windows["H1"]) < self.h1_adx_period + 2:
            return detected

        h1 = windows["H1"]
        m5 = windows["M5"]

        # ==============================================================
        # Layer 1: H1 ADX + DI Trend
        # ==============================================================
        h1_high = h1["high"].values.astype(np.float64)
        h1_low = h1["low"].values.astype(np.float64)
        h1_close = h1["close"].values.astype(np.float64)

        h1_adx = talib.ADX(h1_high, h1_low, h1_close, timeperiod=self.h1_adx_period)
        h1_plus_di = talib.PLUS_DI(h1_high, h1_low, h1_close, timeperiod=self.h1_adx_period)
        h1_minus_di = talib.MINUS_DI(h1_high, h1_low, h1_close, timeperiod=self.h1_adx_period)

        if np.isnan(h1_adx[-1]) or np.isnan(h1_plus_di[-1]) or np.isnan(h1_minus_di[-1]):
            return detected

        # ADX must be above threshold (trending market)
        if h1_adx[-1] < self.h1_adx_threshold:
            return detected

        # +DI > -DI means bullish trend, -DI > +DI means bearish trend
        h1_bullish = h1_plus_di[-1] > h1_minus_di[-1]
        h1_bearish = h1_minus_di[-1] > h1_plus_di[-1]

        if not h1_bullish and not h1_bearish:
            return detected

        # ==============================================================
        # Layer 2: M5 EMA Crossover
        # ==============================================================
        m5_close = m5["close"].values.astype(np.float64)
        m5_ema_f = talib.EMA(m5_close, timeperiod=self.m5_ema_fast)
        m5_ema_s = talib.EMA(m5_close, timeperiod=self.m5_ema_slow)

        if len(m5_ema_f) < 2 or np.isnan(m5_ema_f[-1]) or np.isnan(m5_ema_s[-1]):
            return detected

        # EMA9 crosses above EMA21
        ema_cross_up = (m5_ema_f[-2] < m5_ema_s[-2]) and (m5_ema_f[-1] > m5_ema_s[-1])
        # EMA9 crosses below EMA21
        ema_cross_down = (m5_ema_f[-2] > m5_ema_s[-2]) and (m5_ema_f[-1] < m5_ema_s[-1])

        # ==============================================================
        # Confluence
        # ==============================================================
        if h1_bullish and ema_cross_up:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=m5.index[-1],
                end_time=m5.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "h1_adx": h1_adx[-1],
                    "h1_plus_di": h1_plus_di[-1],
                    "h1_minus_di": h1_minus_di[-1],
                    "m5_ema_fast": m5_ema_f[-1],
                    "m5_ema_slow": m5_ema_s[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif h1_bearish and ema_cross_down:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=m5.index[-1],
                end_time=m5.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "h1_adx": h1_adx[-1],
                    "h1_plus_di": h1_plus_di[-1],
                    "h1_minus_di": h1_minus_di[-1],
                    "m5_ema_fast": m5_ema_f[-1],
                    "m5_ema_slow": m5_ema_s[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
