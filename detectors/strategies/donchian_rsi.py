"""
detectors/strategies/donchian_rsi.py
-------------------------------------
Donchian channel breakout + RSI confirmation (M5 only).

Rules:
  LONG:  close breaks above 20-period high + RSI > 50
  SHORT: close breaks below 20-period low  + RSI < 50
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
    logger.warning("TA-Lib not installed. DonchianRsi strategy signals disabled.")


class DonchianRsiStrategy(BaseStrategy):
    """Donchian channel breakout with RSI momentum confirmation (M5 only)."""

    name = "donchian_rsi"

    def __init__(
        self,
        donchian_period: int = 20,
        rsi_period: int = 14,
    ) -> None:
        self.donchian_period = donchian_period
        self.rsi_period = rsi_period

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

        # Donchian channels: N-period high and low (excluding current bar for breakout)
        don_high = talib.MAX(high, timeperiod=self.donchian_period)
        don_low = talib.MIN(low, timeperiod=self.donchian_period)

        # RSI
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        if np.isnan(don_high[-2]) or np.isnan(don_low[-2]) or np.isnan(rsi[-1]):
            return detected

        current_close = close[-1]
        prev_don_high = don_high[-2]
        prev_don_low = don_low[-2]

        # Breakout: close exceeds the prior N-period extreme
        break_above_upper = current_close > prev_don_high
        break_below_lower = current_close < prev_don_low

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
                    "don_high": prev_don_high,
                    "don_low": prev_don_low,
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
                    "don_high": prev_don_high,
                    "don_low": prev_don_low,
                    "close": float(current_close),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
