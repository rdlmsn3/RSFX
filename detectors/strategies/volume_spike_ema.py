"""
detectors/strategies/volume_spike_ema.py
-----------------------------------------
Volume spike + bullish/bearish close + EMA trend strategy (M5 only).

Rules:
  LONG:  Volume > 2x 20-bar average + bullish close + EMA9 > EMA21
  SHORT: Volume > 2x 20-bar average + bearish close + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. Volume Spike EMA signals disabled.")


class VolumeSpikeEMAStrategy(BaseStrategy):
    """Volume spike + close direction + EMA trend filter (M5)."""

    name = "volume_spike_ema"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        volume_avg_period: int = 20,
        volume_spike_mult: float = 2.0,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.volume_avg_period = volume_avg_period
        self.volume_spike_mult = volume_spike_mult

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ema_slow, self.volume_avg_period) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        ema_fast = talib.EMA(close, timeperiod=self.ema_fast)
        ema_slow = talib.EMA(close, timeperiod=self.ema_slow)

        # Average volume over the lookback period
        avg_volume = np.mean(volume[-self.volume_avg_period - 1:-1])

        vol_now = volume[-1]
        close_now = close[-1]
        open_now = window["open"].values[-1]

        # Bullish / bearish close
        bullish_close = close_now > open_now
        bearish_close = close_now < open_now

        # Volume spike
        volume_spike = vol_now > self.volume_spike_mult * avg_volume

        ema_fast_now = ema_fast[-1]
        ema_slow_now = ema_slow[-1]

        if not volume_spike:
            return detected

        # Long: volume spike + bullish close + EMA9 > EMA21
        if bullish_close and ema_fast_now > ema_slow_now:
            confidence = min(0.90, 0.70 + (vol_now / (self.volume_spike_mult * avg_volume) - 1) * 0.1)
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=round(confidence, 2),
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "volume": vol_now,
                    "avg_volume": round(avg_volume, 2),
                    "ema9": round(ema_fast_now, 5),
                    "ema21": round(ema_slow_now, 5),
                    "price": close_now,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: volume spike + bearish close + EMA9 < EMA21
        elif bearish_close and ema_fast_now < ema_slow_now:
            confidence = min(0.90, 0.70 + (vol_now / (self.volume_spike_mult * avg_volume) - 1) * 0.1)
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=round(confidence, 2),
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "volume": vol_now,
                    "avg_volume": round(avg_volume, 2),
                    "ema9": round(ema_fast_now, 5),
                    "ema21": round(ema_slow_now, 5),
                    "price": close_now,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
