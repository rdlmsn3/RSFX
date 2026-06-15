"""
detectors/strategies/vwap_bounce.py
------------------------------------
VWAP bounce + RSI strategy (M5).

VWAP calculated manually as cumulative (typical price * volume) / cumulative volume.
Rules:
  LONG:  Price touches VWAP from above + RSI > 40
  SHORT: Price touches VWAP from below + RSI < 60
  Exit target: 1x ATR from VWAP
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
    logger.warning("TA-Lib not installed. VWAP Bounce signals disabled.")


class VwapBounceStrategy(BaseStrategy):
    """VWAP bounce with RSI filter and ATR-based exit (M5)."""

    name = "vwap_bounce"

    def __init__(
        self,
        rsi_period: int = 14,
        atr_period: int = 14,
        touch_tolerance: float = 0.0005,
    ) -> None:
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.touch_tolerance = touch_tolerance

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.atr_period, self.rsi_period, 52) + 10
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        # Calculate VWAP manually
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        cum_vol = np.where(cum_vol == 0, 1.0, cum_vol)
        vwap = cum_tp_vol / cum_vol

        # RSI and ATR
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

        price_curr = close[-1]
        price_prev = close[-2]
        vwap_curr = vwap[-1]

        # Touch detection: price low came near VWAP from above
        tolerance = vwap_curr * self.touch_tolerance
        touched_from_above = (
            low[-1] <= vwap_curr + tolerance
            and price_prev > vwap_curr
        )
        # Touch detection: price high came near VWAP from below
        touched_from_below = (
            high[-1] >= vwap_curr - tolerance
            and price_prev < vwap_curr
        )

        # ATR-based exit target
        atr_val = atr[-1]
        exit_long = vwap_curr + atr_val
        exit_short = vwap_curr - atr_val

        # Long: touches VWAP from above + RSI > 40
        if touched_from_above and rsi[-1] > 40:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "vwap": float(vwap_curr),
                    "rsi": float(rsi[-1]),
                    "atr": float(atr_val),
                    "exit_target": float(exit_long),
                    "price": float(price_curr),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: touches VWAP from below + RSI < 60
        elif touched_from_below and rsi[-1] < 60:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "vwap": float(vwap_curr),
                    "rsi": float(rsi[-1]),
                    "atr": float(atr_val),
                    "exit_target": float(exit_short),
                    "price": float(price_curr),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
