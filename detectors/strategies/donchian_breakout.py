"""
detectors/strategies/donchian_breakout.py
------------------------------------------
Donchian channel breakout with volume (M5 only).

Rules:
  LONG:  close breaks above 20-period high + volume spike
  SHORT: close breaks below 20-period low + volume spike
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
    logger.warning("TA-Lib not installed. DonchianBreakout strategy signals disabled.")


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian channel breakout with volume confirmation (M5 only)."""

    name = "donchian_breakout"

    def __init__(
        self,
        donchian_period: int = 20,
        consolidation_bars: int = 5,
        volume_mult: float = 1.5,
    ) -> None:
        self.donchian_period = donchian_period
        self.consolidation_bars = consolidation_bars
        self.volume_mult = volume_mult

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.donchian_period + 2:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        # Donchian channels: N-period high and low (excluding current bar)
        don_high = talib.MAX(high, timeperiod=self.donchian_period)
        don_low = talib.MIN(low, timeperiod=self.donchian_period)

        # Volume spike: current volume > volume_mult * 20-period SMA
        vol_sma = talib.SMA(volume, timeperiod=20)

        if np.isnan(don_high[-2]) or np.isnan(don_low[-2]) or np.isnan(vol_sma[-1]) or vol_sma[-1] == 0:
            return detected

        current_close = close[-1]
        is_volume_spike = volume[-1] > self.volume_mult * vol_sma[-1]

        # Use previous bar's channel levels for breakout detection
        prev_don_high = don_high[-2]
        prev_don_low = don_low[-2]

        # Breakout: close exceeds the prior N-period extreme
        break_above_upper = current_close > prev_don_high
        break_below_lower = current_close < prev_don_low

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
                    "don_high": prev_don_high,
                    "don_low": prev_don_low,
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
                    "don_high": prev_don_high,
                    "don_low": prev_don_low,
                    "volume_ratio": volume[-1] / vol_sma[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
