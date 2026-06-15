"""
detectors/strategies/triple_confirm.py
---------------------------------------
EMA + RSI + MACD triple confirmation (M5 only).

Rules:
  LONG:  EMA9 > EMA21 + RSI > 50 + MACD > signal line
  SHORT: EMA9 < EMA21 + RSI < 50 + MACD < signal line
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
    logger.warning("TA-Lib not installed. TripleConfirm strategy signals disabled.")


class TripleConfirmStrategy(BaseStrategy):
    """Triple confluence: EMA trend + RSI momentum + MACD direction."""

    name = "triple_confirm"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = max(self.ema_slow, self.macd_slow + self.macd_signal) + 1
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        macd, signal, _ = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        # --- Trend conditions ---
        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # --- RSI conditions ---
        rsi_bullish = rsi[-1] > 50
        rsi_bearish = rsi[-1] < 50

        # --- MACD conditions ---
        macd_bullish = macd[-1] > signal[-1]
        macd_bearish = macd[-1] < signal[-1]

        # --- Generate signals ---
        if uptrend and rsi_bullish and macd_bullish:
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
                    "rsi": float(rsi[-1]),
                    "macd": float(macd[-1]),
                    "macd_signal": float(signal[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s, triple confirm)", current_timestamp, self.name)

        elif downtrend and rsi_bearish and macd_bearish:
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
                    "rsi": float(rsi[-1]),
                    "macd": float(macd[-1]),
                    "macd_signal": float(signal[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s, triple confirm)", current_timestamp, self.name)

        return detected
