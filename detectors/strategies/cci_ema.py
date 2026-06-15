"""
detectors/strategies/cci_ema.py
-------------------------------
CCI crossover + EMA 50 trend strategy (M5 only).

Rules:
  LONG:  CCI crosses above -100 + price > EMA 50
  SHORT: CCI crosses below +100 + price < EMA 50
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
    logger.warning("TA-Lib not installed. CCI EMA signals disabled.")


class CCIERMATrendStrategy(BaseStrategy):
    """CCI crossover + EMA 50 trend filter (M5)."""

    name = "cci_ema"

    def __init__(
        self,
        cci_period: int = 20,
        ema_period: int = 50,
        oscillator_lookback: int = 5,
    ) -> None:
        self.cci_period = cci_period
        self.ema_period = ema_period
        self.oscillator_lookback = oscillator_lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.cci_period, self.ema_period) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        cci = talib.CCI(high, low, close, timeperiod=self.cci_period)
        ema50 = talib.EMA(close, timeperiod=self.ema_period)

        # CCI crosses above -100 (within lookback window)
        lb = self.oscillator_lookback
        cci_cross_above_neg100 = bool(np.any(
            (cci[-lb - 1 : -1] <= -100.0) &
            (cci[-lb:] > -100.0)
        ))
        # CCI crosses below +100 (within lookback window)
        cci_cross_below_pos100 = bool(np.any(
            (cci[-lb - 1 : -1] >= 100.0) &
            (cci[-lb:] < 100.0)
        ))

        price_now = close[-1]
        ema_now = ema50[-1]

        # Long: CCI crosses above -100 + price > EMA50
        if cci_cross_above_neg100 and price_now > ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "cci_prev": cci[-2],
                    "cci_now": cci[-1],
                    "ema50": ema_now,
                    "price": price_now,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: CCI crosses below +100 + price < EMA50
        elif cci_cross_below_pos100 and price_now < ema_now:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "cci_prev": cci[-2],
                    "cci_now": cci[-1],
                    "ema50": ema_now,
                    "price": price_now,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
