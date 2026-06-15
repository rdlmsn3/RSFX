"""
detectors/strategies/ad_ema.py
-------------------------------
A/D line crosses its EMA + price trend strategy (M5 only).

Rules:
  LONG:  AD crosses above AD_EMA + price > EMA50
  SHORT: AD crosses below AD_EMA + price < EMA50
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
    logger.warning("TA-Lib not installed. A/D EMA signals disabled.")


class ADEMStrategy(BaseStrategy):
    """A/D line / A/D-EMA crossover + price trend filter (M5)."""

    name = "ad_ema"

    def __init__(
        self,
        ad_ema_period: int = 20,
        price_ema_period: int = 50,
    ) -> None:
        self.ad_ema_period = ad_ema_period
        self.price_ema_period = price_ema_period

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ad_ema_period, self.price_ema_period) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        ad = talib.AD(high, low, close, volume)
        ad_ema = talib.EMA(ad, timeperiod=self.ad_ema_period)
        price_ema = talib.EMA(close, timeperiod=self.price_ema_period)

        # Crossover detection: compare current vs previous bar
        ad_now = ad[-1]
        ad_ema_now = ad_ema[-1]
        ad_prev = ad[-2]
        ad_ema_prev = ad_ema[-2]

        price_now = close[-1]
        price_ema_now = price_ema[-1]

        # AD crosses above its EMA
        ad_cross_up = ad_prev <= ad_ema_prev and ad_now > ad_ema_now
        # AD crosses below its EMA
        ad_cross_down = ad_prev >= ad_ema_prev and ad_now < ad_ema_now

        # Long: AD crosses above AD_EMA + price > EMA50
        if ad_cross_up and price_now > price_ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.76,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ad": round(ad_now, 2),
                    "ad_ema": round(ad_ema_now, 2),
                    "ema50": round(price_ema_now, 5),
                    "price": price_now,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: AD crosses below AD_EMA + price < EMA50
        elif ad_cross_down and price_now < price_ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.76,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ad": round(ad_now, 2),
                    "ad_ema": round(ad_ema_now, 2),
                    "ema50": round(price_ema_now, 5),
                    "price": price_now,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
