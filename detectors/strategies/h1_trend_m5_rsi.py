"""
detectors/strategies/h1_trend_m5_rsi.py
----------------------------------------
H1 EMA trend + M5 RSI bounce strategy.

Layer 1 (H1): Trend bias — EMA9/EMA21 direction
Layer 2 (M5): Entry trigger — RSI bounces from key levels

LONG:  H1 EMA9 > EMA21 + M5 RSI bounces from 40
SHORT: H1 EMA9 < EMA21 + M5 RSI bounces from 60
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
    logger.warning("TA-Lib not installed. H1TrendM5Rsi strategy signals disabled.")


class H1TrendM5RsiStrategy(BaseStrategy):
    """H1 EMA trend filter + M5 RSI bounce entry."""

    name = "h1_trend_m5_rsi"

    def __init__(
        self,
        h1_ema_fast: int = 9,
        h1_ema_slow: int = 21,
        m5_rsi_period: int = 14,
        m5_rsi_bounce_long: float = 40.0,
        m5_rsi_bounce_short: float = 60.0,
        m5_rsi_tolerance: float = 5.0,
    ) -> None:
        self.h1_ema_fast = h1_ema_fast
        self.h1_ema_slow = h1_ema_slow
        self.m5_rsi_period = m5_rsi_period
        self.m5_rsi_bounce_long = m5_rsi_bounce_long
        self.m5_rsi_bounce_short = m5_rsi_bounce_short
        self.m5_rsi_tolerance = m5_rsi_tolerance

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
        # Layer 2: M5 RSI Bounce
        # ==============================================================
        m5_close = m5["close"].values.astype(np.float64)
        m5_rsi = talib.RSI(m5_close, timeperiod=self.m5_rsi_period)

        if len(m5_rsi) < 2 or np.isnan(m5_rsi[-1]):
            return detected

        # Bounce from 40: RSI was at or below 40 then bounced above
        rsi_bounce_long = (
            m5_rsi[-2] <= self.m5_rsi_bounce_long + self.m5_rsi_tolerance
            and m5_rsi[-1] > m5_rsi[-2]
        )
        # Bounce from 60: RSI was at or above 60 then dropped below
        rsi_bounce_short = (
            m5_rsi[-2] >= self.m5_rsi_bounce_short - self.m5_rsi_tolerance
            and m5_rsi[-1] < m5_rsi[-2]
        )

        # ==============================================================
        # Confluence
        # ==============================================================
        if h1_uptrend and rsi_bounce_long:
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
                    "m5_rsi": m5_rsi[-1],
                    "m5_rsi_prev": m5_rsi[-2],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif h1_downtrend and rsi_bounce_short:
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
                    "m5_rsi": m5_rsi[-1],
                    "m5_rsi_prev": m5_rsi[-2],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
