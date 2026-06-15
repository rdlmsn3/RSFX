"""
detectors/strategies/bb_squeeze_breakout.py
--------------------------------------------
Bollinger Band squeeze breakout with volume confirmation (M5 only).

Setup: BB width is below its 20-period moving average (squeeze / low volatility).
Rules:
  LONG:  Price breaks above upper BB + volume spike (vol > 2x avg volume)
  SHORT: Price breaks below lower BB + volume spike (vol > 2x avg volume)
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
    logger.warning("TA-Lib not installed. BbSqueezeBreakout strategy signals disabled.")


class BbSqueezeBreakoutStrategy(BaseStrategy):
    """BB squeeze detection + breakout with volume spike confirmation."""

    name = "bb_squeeze_breakout"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_width_avg_period: int = 20,
        volume_spike_mult: float = 2.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_avg_period = bb_width_avg_period
        self.volume_spike_mult = volume_spike_mult

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = self.bb_period + self.bb_width_avg_period + 1
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        # --- Indicators ---
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
        )

        # BB width = (upper - lower) / middle
        bb_width = np.where(middle != 0, (upper - lower) / middle, 0.0)
        bb_width_avg = talib.SMA(bb_width, timeperiod=self.bb_width_avg_period)

        # Volume average (20-period)
        vol_avg = talib.SMA(volume, timeperiod=self.bb_width_avg_period)

        # --- Squeeze detection ---
        # Squeeze: current BB width < its moving average
        is_squeeze = bb_width[-1] < bb_width_avg[-1]

        # --- Breakout detection ---
        # Previous bar was inside (or at) bands, current bar broke out
        broke_above_upper = (close[-2] <= upper[-2]) and (close[-1] > upper[-1])
        broke_below_lower = (close[-2] >= lower[-2]) and (close[-1] < lower[-1])

        # --- Volume spike ---
        volume_spike = vol_avg[-1] > 0 and volume[-1] > (vol_avg[-1] * self.volume_spike_mult)

        # --- Generate signals ---
        if broke_above_upper and volume_spike:
            confidence = 0.8 if is_squeeze else 0.6
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width[-1]),
                    "bb_width_avg": float(bb_width_avg[-1]),
                    "was_squeeze": is_squeeze,
                    "volume": float(volume[-1]),
                    "volume_avg": float(vol_avg[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s, squeeze=%s)", current_timestamp, self.name, is_squeeze)

        elif broke_below_lower and volume_spike:
            confidence = 0.8 if is_squeeze else 0.6
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width[-1]),
                    "bb_width_avg": float(bb_width_avg[-1]),
                    "was_squeeze": is_squeeze,
                    "volume": float(volume[-1]),
                    "volume_avg": float(vol_avg[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s, squeeze=%s)", current_timestamp, self.name, is_squeeze)

        return detected
