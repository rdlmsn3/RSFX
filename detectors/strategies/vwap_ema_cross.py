"""
detectors/strategies/vwap_ema_cross.py
---------------------------------------
VWAP + EMA cross strategy (M5).

VWAP calculated manually as cumulative (typical price * volume) / cumulative volume.
Rules:
  LONG:  Price crosses above VWAP + EMA9 > EMA21
  SHORT: Price crosses below VWAP + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. VWAP EMA Cross signals disabled.")


class VwapEmaCrossStrategy(BaseStrategy):
    """VWAP crossover confirmed by EMA alignment (M5)."""

    name = "vwap_ema_cross"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ema_slow, 52) + 10
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        # Calculate VWAP manually: cumulative(typical_price * volume) / cumulative(volume)
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        # Avoid division by zero
        cum_vol = np.where(cum_vol == 0, 1.0, cum_vol)
        vwap = cum_tp_vol / cum_vol

        # Compute EMAs
        ema_fast = talib.EMA(close, timeperiod=self.ema_fast)
        ema_slow = talib.EMA(close, timeperiod=self.ema_slow)

        # Cross detection: current bar vs previous bar relative to VWAP
        price_curr = close[-1]
        price_prev = close[-2]
        vwap_curr = vwap[-1]
        vwap_prev = vwap[-2]

        cross_above = price_prev <= vwap_prev and price_curr > vwap_curr
        cross_below = price_prev >= vwap_prev and price_curr < vwap_curr

        # EMA alignment
        ema_bullish = ema_fast[-1] > ema_slow[-1]
        ema_bearish = ema_fast[-1] < ema_slow[-1]

        # Long: crosses above VWAP + EMA9 > EMA21
        if cross_above and ema_bullish:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "vwap": float(vwap_curr),
                    "ema_fast": float(ema_fast[-1]),
                    "ema_slow": float(ema_slow[-1]),
                    "price": float(price_curr),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: crosses below VWAP + EMA9 < EMA21
        elif cross_below and ema_bearish:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "vwap": float(vwap_curr),
                    "ema_fast": float(ema_fast[-1]),
                    "ema_slow": float(ema_slow[-1]),
                    "price": float(price_curr),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
