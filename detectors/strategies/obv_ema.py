"""
detectors/strategies/obv_ema.py
--------------------------------
OBV crosses its EMA + price trend strategy (M5 only).

Rules:
  LONG:  OBV crosses above OBV_EMA + price > EMA50
  SHORT: OBV crosses below OBV_EMA + price < EMA50
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
    logger.warning("TA-Lib not installed. OBV EMA signals disabled.")


class OBVEMAStrategy(BaseStrategy):
    """OBV / OBV-EMA crossover + price trend filter (M5)."""

    name = "obv_ema"

    def __init__(
        self,
        obv_ema_period: int = 20,
        price_ema_period: int = 50,
    ) -> None:
        self.obv_ema_period = obv_ema_period
        self.price_ema_period = price_ema_period

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.obv_ema_period, self.price_ema_period) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        obv = talib.OBV(close, volume)
        obv_ema = talib.EMA(obv, timeperiod=self.obv_ema_period)
        price_ema = talib.EMA(close, timeperiod=self.price_ema_period)

        # Crossover detection: compare current vs previous bar
        obv_now = obv[-1]
        obv_ema_now = obv_ema[-1]
        obv_prev = obv[-2]
        obv_ema_prev = obv_ema[-2]

        price_now = close[-1]
        price_ema_now = price_ema[-1]

        # OBV crosses above its EMA
        obv_cross_up = obv_prev <= obv_ema_prev and obv_now > obv_ema_now
        # OBV crosses below its EMA
        obv_cross_down = obv_prev >= obv_ema_prev and obv_now < obv_ema_now

        # Long: OBV crosses above OBV_EMA + price > EMA50
        if obv_cross_up and price_now > price_ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.76,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "obv": round(obv_now, 2),
                    "obv_ema": round(obv_ema_now, 2),
                    "ema50": round(price_ema_now, 5),
                    "price": price_now,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: OBV crosses below OBV_EMA + price < EMA50
        elif obv_cross_down and price_now < price_ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.76,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "obv": round(obv_now, 2),
                    "obv_ema": round(obv_ema_now, 2),
                    "ema50": round(price_ema_now, 5),
                    "price": price_now,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
