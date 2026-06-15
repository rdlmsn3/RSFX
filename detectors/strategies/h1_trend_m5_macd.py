"""
detectors/strategies/h1_trend_m5_macd.py
-----------------------------------------
H1 EMA trend + M5 MACD cross strategy.

Layer 1 (H1): Trend bias — EMA9/EMA21 direction
Layer 2 (M5): Entry trigger — MACD crosses signal line

LONG:  H1 EMA9 > EMA21 + M5 MACD crosses above signal
SHORT: H1 EMA9 < EMA21 + M5 MACD crosses below signal
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
    logger.warning("TA-Lib not installed. H1TrendM5Macd strategy signals disabled.")


class H1TrendM5MacdStrategy(BaseStrategy):
    """H1 EMA trend filter + M5 MACD crossover entry."""

    name = "h1_trend_m5_macd"

    def __init__(
        self,
        h1_ema_fast: int = 9,
        h1_ema_slow: int = 21,
        m5_macd_fast: int = 12,
        m5_macd_slow: int = 26,
        m5_macd_signal: int = 9,
    ) -> None:
        self.h1_ema_fast = h1_ema_fast
        self.h1_ema_slow = h1_ema_slow
        self.m5_macd_fast = m5_macd_fast
        self.m5_macd_slow = m5_macd_slow
        self.m5_macd_signal = m5_macd_signal

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
        # Layer 2: M5 MACD Cross
        # ==============================================================
        m5_close = m5["close"].values.astype(np.float64)
        m5_macd, m5_signal, m5_hist = talib.MACD(
            m5_close,
            fastperiod=self.m5_macd_fast,
            slowperiod=self.m5_macd_slow,
            signalperiod=self.m5_macd_signal,
        )

        if len(m5_macd) < 2 or np.isnan(m5_macd[-1]) or np.isnan(m5_signal[-1]):
            return detected

        # MACD crosses above signal line
        macd_cross_up = (m5_macd[-2] < m5_signal[-2]) and (m5_macd[-1] > m5_signal[-1])
        # MACD crosses below signal line
        macd_cross_down = (m5_macd[-2] > m5_signal[-2]) and (m5_macd[-1] < m5_signal[-1])

        # ==============================================================
        # Confluence
        # ==============================================================
        if h1_uptrend and macd_cross_up:
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
                    "m5_macd": m5_macd[-1],
                    "m5_signal": m5_signal[-1],
                    "m5_hist": m5_hist[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif h1_downtrend and macd_cross_down:
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
                    "m5_macd": m5_macd[-1],
                    "m5_signal": m5_signal[-1],
                    "m5_hist": m5_hist[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
