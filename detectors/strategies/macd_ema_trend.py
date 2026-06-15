"""
detectors/strategies/macd_ema_trend.py
---------------------------------------
MACD crossover + EMA 50 trend filter (M5 only).

Rules:
  LONG:  MACD line crosses above signal line + price > EMA50
  SHORT: MACD line crosses below signal line + price < EMA50
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
    logger.warning("TA-Lib not installed. MacdEmaTrend strategy signals disabled.")


class MacdEmaTrendStrategy(BaseStrategy):
    """MACD cross confirmed by EMA 50 trend direction."""

    name = "macd_ema_trend"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ema_trend: int = 50,
        oscillator_lookback: int = 5,
    ) -> None:
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_trend = ema_trend
        self.oscillator_lookback = oscillator_lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = max(self.macd_slow + self.macd_signal, self.ema_trend) + 1
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        macd_line, signal_line, _ = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        ema50 = talib.EMA(close, timeperiod=self.ema_trend)

        # --- Crossover detection (within lookback window) ---
        lb = self.oscillator_lookback
        cross_above_recent = bool(np.any(
            (macd_line[-lb - 1 : -1] <= signal_line[-lb - 1 : -1]) &
            (macd_line[-lb:] > signal_line[-lb:])
        ))
        cross_below_recent = bool(np.any(
            (macd_line[-lb - 1 : -1] >= signal_line[-lb - 1 : -1]) &
            (macd_line[-lb:] < signal_line[-lb:])
        ))

        # --- Trend filter ---
        above_ema50 = close[-1] > ema50[-1]
        below_ema50 = close[-1] < ema50[-1]

        # --- Generate signals ---
        if cross_above_recent and above_ema50:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "macd": float(macd_line[-1]),
                    "macd_signal": float(signal_line[-1]),
                    "ema50": float(ema50[-1]),
                    "close": float(close[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif cross_below_recent and below_ema50:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "macd": float(macd_line[-1]),
                    "macd_signal": float(signal_line[-1]),
                    "ema50": float(ema50[-1]),
                    "close": float(close[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
