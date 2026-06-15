"""
detectors/strategies/ema_rsi_cross.py
--------------------------------------
EMA 9/21 crossover + RSI 14 confirmation (M5 only).

Rules:
  LONG:  EMA9 crosses above EMA21 + RSI was < 40 within last 5 candles
  SHORT: EMA9 crosses below EMA21 + RSI was > 60 within last 5 candles
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
    logger.warning("TA-Lib not installed. EmaRsiCross strategy signals disabled.")


class EmaRsiCrossStrategy(BaseStrategy):
    """EMA 9/21 crossover confirmed by RSI 14."""

    name = "ema_rsi_cross"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_long_threshold: float = 40.0,
        rsi_short_threshold: float = 60.0,
        rsi_lookback: int = 5,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.0,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.rsi_lookback = rsi_lookback
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        m5 = windows.get("M5")
        m1 = windows.get("M1")
        window = m5 if m5 is not None and not m5.empty else m1
        if window is None or window.empty or not TA_AVAILABLE or len(window) < self.ema_slow + self.rsi_lookback + 1:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        # --- Crossover detection ---
        cross_above = (ema_f[-2] <= ema_s[-2]) and (ema_f[-1] > ema_s[-1])
        cross_below = (ema_f[-2] >= ema_s[-2]) and (ema_f[-1] < ema_s[-1])

        if not cross_above and not cross_below:
            return detected

        # --- RSI condition: was RSI in zone within last N candles? ---
        rsi_window = rsi[-(self.rsi_lookback + 1):]
        rsi_was_oversold = bool(np.any(rsi_window < self.rsi_long_threshold))
        rsi_was_overbought = bool(np.any(rsi_window > self.rsi_short_threshold))

        # --- Generate signals ---
        if cross_above and rsi_was_oversold:
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
                },
            ))
            self.compute_tp_sl(detected[-1], window, self.atr_period, self.sl_atr_mult, self.tp_atr_mult)
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif cross_below and rsi_was_overbought:
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
                },
            ))
            self.compute_tp_sl(detected[-1], window, self.atr_period, self.sl_atr_mult, self.tp_atr_mult)
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
