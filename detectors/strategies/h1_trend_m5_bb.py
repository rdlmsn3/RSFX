"""
detectors/strategies/h1_trend_m5_bb.py
---------------------------------------
H1 EMA trend + M5 Bollinger bounce strategy.

Layer 1 (H1): Trend bias — EMA9/EMA21 direction
Layer 2 (M5): Entry trigger — price touches Bollinger Band + RSI confirmation

LONG:  H1 EMA9 > EMA21 + M5 price touches lower BB + RSI < 35
SHORT: H1 EMA9 < EMA21 + M5 price touches upper BB + RSI > 65
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
    logger.warning("TA-Lib not installed. H1TrendM5Bb strategy signals disabled.")


class H1TrendM5BbStrategy(BaseStrategy):
    """H1 EMA trend filter + M5 Bollinger Band bounce entry."""

    name = "h1_trend_m5_bb"

    def __init__(
        self,
        h1_ema_fast: int = 9,
        h1_ema_slow: int = 21,
        m5_bb_period: int = 20,
        m5_bb_std: float = 2.0,
        m5_rsi_period: int = 14,
        m5_rsi_long_thresh: float = 35.0,
        m5_rsi_short_thresh: float = 65.0,
    ) -> None:
        self.h1_ema_fast = h1_ema_fast
        self.h1_ema_slow = h1_ema_slow
        self.m5_bb_period = m5_bb_period
        self.m5_bb_std = m5_bb_std
        self.m5_rsi_period = m5_rsi_period
        self.m5_rsi_long_thresh = m5_rsi_long_thresh
        self.m5_rsi_short_thresh = m5_rsi_short_thresh

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
        # Layer 2: M5 Bollinger Band Bounce + RSI
        # ==============================================================
        m5_close = m5["close"].values.astype(np.float64)
        m5_low = m5["low"].values.astype(np.float64)
        m5_high = m5["high"].values.astype(np.float64)

        m5_upper, m5_middle, m5_lower = talib.BBANDS(
            m5_close,
            timeperiod=self.m5_bb_period,
            nbdevup=self.m5_bb_std,
            nbdevdn=self.m5_bb_std,
            matype=0,
        )
        m5_rsi = talib.RSI(m5_close, timeperiod=self.m5_rsi_period)

        if np.isnan(m5_lower[-1]) or np.isnan(m5_upper[-1]) or np.isnan(m5_rsi[-1]):
            return detected

        # Touch lower BB: low went to or below lower band
        touches_lower_bb = m5_low[-1] <= m5_lower[-1]
        # Touch upper BB: high went to or above upper band
        touches_upper_bb = m5_high[-1] >= m5_upper[-1]

        # ==============================================================
        # Confluence
        # ==============================================================
        if h1_uptrend and touches_lower_bb and m5_rsi[-1] < self.m5_rsi_long_thresh:
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
                    "m5_bb_lower": m5_lower[-1],
                    "m5_bb_upper": m5_upper[-1],
                    "m5_low": m5_low[-1],
                    "m5_rsi": m5_rsi[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif h1_downtrend and touches_upper_bb and m5_rsi[-1] > self.m5_rsi_short_thresh:
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
                    "m5_bb_lower": m5_lower[-1],
                    "m5_bb_upper": m5_upper[-1],
                    "m5_high": m5_high[-1],
                    "m5_rsi": m5_rsi[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
