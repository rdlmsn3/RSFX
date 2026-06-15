"""
detectors/strategies/ema_ribbon_pullback.py
--------------------------------------------
EMA ribbon aligned (5,8,13,21) + pullback to EMA13 with candle confirmation (M5).

Rules:
  LONG:  Bullish alignment (5>8>13>21) + pullback to EMA13 + bullish candle
  SHORT: Bearish alignment (5<8<13<21) + pullback to EMA13 + bearish candle
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
    logger.warning("TA-Lib not installed. EMA Ribbon Pullback signals disabled.")


class EmaRibbonPullbackStrategy(BaseStrategy):
    """EMA ribbon alignment with pullback to EMA13 and candle confirmation (M5)."""

    name = "ema_ribbon_pullback"

    def __init__(
        self,
        periods: list[int] | None = None,
        pullback_tolerance: float = 0.001,
    ) -> None:
        self.periods = periods or [5, 8, 13, 21]
        self.pullback_tolerance = pullback_tolerance

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.periods) + 10
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        open_ = window["open"].values.astype(np.float64)

        # Compute EMAs
        emas = [talib.EMA(close, timeperiod=p) for p in self.periods]

        # Check ribbon alignment (fastest first: 5, 8, 13, 21)
        ema5, ema8, ema13, ema21 = [e[-1] for e in emas]

        bullish_alignment = ema5 > ema8 > ema13 > ema21
        bearish_alignment = ema5 < ema8 < ema13 < ema21

        # Pullback: price low touched or came within tolerance of EMA13
        price_low = low[-1]
        price_close = close[-1]
        price_open = open_[-1]
        ema13_val = emas[2][-1]

        tolerance = ema13_val * self.pullback_tolerance
        touched_ema13 = abs(price_low - ema13_val) <= tolerance

        # Also check previous bars for the pullback
        prev_touched = False
        for i in range(-3, -1):
            if abs(low[i] - emas[2][i]) <= tolerance:
                prev_touched = True
                break

        pullback = touched_ema13 or prev_touched

        # Candle confirmation
        bullish_candle = price_close > price_open
        bearish_candle = price_close < price_open

        # Long: bullish alignment + pullback to EMA13 + bullish candle
        if bullish_alignment and pullback and price_close >= ema13_val and bullish_candle:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema5": float(ema5),
                    "ema8": float(ema8),
                    "ema13": float(ema13_val),
                    "ema21": float(ema21),
                    "price": float(price_close),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: bearish alignment + pullback to EMA13 + bearish candle
        elif bearish_alignment and pullback and price_close <= ema13_val and bearish_candle:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema5": float(ema5),
                    "ema8": float(ema8),
                    "ema13": float(ema13_val),
                    "ema21": float(ema21),
                    "price": float(price_close),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
