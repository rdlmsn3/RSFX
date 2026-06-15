"""
detectors/strategies/keltner_breakout.py
-----------------------------------------
Keltner channel breakout with volume (M5 only).

Rules:
  LONG:  close breaks above upper KC + volume spike
  SHORT: close breaks below lower KC + volume spike

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
    logger.warning("TA-Lib not installed. KeltnerBreakout strategy signals disabled.")


class KeltnerBreakoutStrategy(BaseStrategy):
    """Keltner channel breakout with volume confirmation (M5 only)."""

    name = "keltner_breakout"

    def __init__(
        self,
        kc_ema: int = 20,
        atr_period: int = 10,
        atr_mult: float = 2.0,
        volume_mult: float = 1.5,
    ) -> None:
        self.kc_ema = kc_ema
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.volume_mult = volume_mult

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
        volume = window["volume"].values.astype(np.float64)

        # Keltner Channel = EMA +/- ATR_mult * ATR
        kc_mid = talib.EMA(close, timeperiod=self.kc_ema)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

        upper_kc = kc_mid + self.atr_mult * atr
        lower_kc = kc_mid - self.atr_mult * atr

        # Volume spike: current volume > volume_mult * 20-period SMA
        vol_sma = talib.SMA(volume, timeperiod=20)

        if np.isnan(upper_kc[-1]) or np.isnan(lower_kc[-1]) or np.isnan(vol_sma[-1]) or vol_sma[-1] == 0:
            return detected

        current_close = close[-1]
        prev_close = close[-2]
        is_volume_spike = volume[-1] > self.volume_mult * vol_sma[-1]

        # Breakout detection (price crosses channel boundary)
        break_above_upper = current_close > upper_kc[-1] and prev_close <= upper_kc[-1]
        break_below_lower = current_close < lower_kc[-1] and prev_close >= lower_kc[-1]

        # --- LONG ---
        if break_above_upper and is_volume_spike:
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
                    "volume_ratio": volume[-1] / vol_sma[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # --- SHORT ---
        elif break_below_lower and is_volume_spike:
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
                    "volume_ratio": volume[-1] / vol_sma[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
