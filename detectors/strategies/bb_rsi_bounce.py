"""
detectors/strategies/bb_rsi_bounce.py
--------------------------------------
Bollinger Band bounce + RSI confirmation (M5 only).

Rules:
  LONG:  Price touches lower Bollinger Band + RSI < 35 (oversold bounce)
  SHORT: Price touches upper Bollinger Band + RSI > 65 (overbought reversal)
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
    logger.warning("TA-Lib not installed. BbRsiBounce strategy signals disabled.")


class BbRsiBounceStrategy(BaseStrategy):
    """Bollinger Band touch + RSI filter for mean-reversion entries."""

    name = "bb_rsi_bounce"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_long_threshold: float = 35.0,
        rsi_short_threshold: float = 65.0,
        oscillator_lookback: int = 5,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.oscillator_lookback = oscillator_lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.bb_period + 1:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
        )
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        # --- Touch detection ---
        # Lower band touch: current close <= lower band (within tolerance)
        touch_lower = close[-1] <= lower[-1]
        # Upper band touch: current close >= upper band
        touch_upper = close[-1] >= upper[-1]

        # --- Generate signals ---
        rsi_oversold_recent = bool(np.any(rsi[-self.oscillator_lookback:] < self.rsi_long_threshold))
        rsi_overbought_recent = bool(np.any(rsi[-self.oscillator_lookback:] > self.rsi_short_threshold))

        if touch_lower and rsi_oversold_recent:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "close": float(close[-1]),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif touch_upper and rsi_overbought_recent:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "close": float(close[-1]),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
