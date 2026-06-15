"""
detectors/strategies/keltner_rsi.py
------------------------------------
Keltner channel breakout + RSI confirmation (M5 only).

Rules:
  LONG:  close breaks above upper KC + RSI > 50
  SHORT: close breaks below lower KC + RSI < 50

KC = EMA20 +/- 2 * ATR10
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
    logger.warning("TA-Lib not installed. KeltnerRsi strategy signals disabled.")


class KeltnerRsiStrategy(BaseStrategy):
    """Keltner channel breakout with RSI momentum confirmation (M5 only)."""

    name = "keltner_rsi"

    def __init__(
        self,
        kc_ema: int = 20,
        atr_period: int = 10,
        atr_mult: float = 2.0,
        rsi_period: int = 14,
    ) -> None:
        self.kc_ema = kc_ema
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.rsi_period = rsi_period

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < max(self.kc_ema + 1, self.atr_period + 1, 21):
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # Keltner Channel = EMA +/- atr_mult * ATR
        kc_mid = talib.EMA(close, timeperiod=self.kc_ema)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

        upper_kc = kc_mid + self.atr_mult * atr
        lower_kc = kc_mid - self.atr_mult * atr

        # RSI
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        if np.isnan(upper_kc[-1]) or np.isnan(lower_kc[-1]) or np.isnan(rsi[-1]):
            return detected

        current_close = close[-1]
        prev_close = close[-2]

        # Breakout detection (price crosses channel boundary)
        break_above_upper = current_close > upper_kc[-1] and prev_close <= upper_kc[-1]
        break_below_lower = current_close < lower_kc[-1] and prev_close >= lower_kc[-1]

        # --- LONG ---
        if break_above_upper and rsi[-1] > 50.0:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "upper_kc": upper_kc[-1],
                    "lower_kc": lower_kc[-1],
                    "kc_mid": kc_mid[-1],
                    "close": float(current_close),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # --- SHORT ---
        elif break_below_lower and rsi[-1] < 50.0:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "upper_kc": upper_kc[-1],
                    "lower_kc": lower_kc[-1],
                    "kc_mid": kc_mid[-1],
                    "close": float(current_close),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
