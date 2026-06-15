"""
detectors/strategies/rsi_divergence.py
--------------------------------------
RSI divergence from price (M5 only).

Rules:
  LONG:  price makes lower low + RSI makes higher low  (bullish divergence)
  SHORT: price makes higher high + RSI makes lower high (bearish divergence)
  Look back 20 bars for swing points.
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
    logger.warning("TA-Lib not installed. RSI Divergence signals disabled.")


class RSIDivergenceStrategy(BaseStrategy):
    """RSI divergence from price on M5."""

    name = "rsi_divergence"

    def __init__(
        self,
        rsi_period: int = 14,
        lookback: int = 20,
    ) -> None:
        self.rsi_period = rsi_period
        self.lookback = lookback

    @staticmethod
    def _find_swing_lows(arr: np.ndarray) -> list[tuple[int, float]]:
        """Find swing low points (local minima) as (index, value)."""
        lows: list[tuple[int, float]] = []
        for i in range(2, len(arr) - 1):
            if arr[i] < arr[i - 1] and arr[i] < arr[i - 2] and arr[i] < arr[i + 1]:
                lows.append((i, arr[i]))
        return lows

    @staticmethod
    def _find_swing_highs(arr: np.ndarray) -> list[tuple[int, float]]:
        """Find swing high points (local maxima) as (index, value)."""
        highs: list[tuple[int, float]] = []
        for i in range(2, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i - 2] and arr[i] > arr[i + 1]:
                highs.append((i, arr[i]))
        return highs

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = self.rsi_period + self.lookback + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        lb = self.lookback
        price_lb = low[-lb:]  # use low for bullish swing detection
        price_hb = high[-lb:]  # use high for bearish swing detection
        rsi_lb = rsi[-lb:]

        # Bullish divergence: price lower low + RSI higher low
        price_lows = self._find_swing_lows(price_lb)
        rsi_lows = self._find_swing_lows(rsi_lb)

        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            pl1, pl2 = price_lows[-2], price_lows[-1]
            rl1, rl2 = rsi_lows[-2], rsi_lows[-1]
            if pl2[1] < pl1[1] and rl2[1] > rl1[1]:
                detected.append(PatternSignal(
                    name=f"{self.name}_LONG",
                    start_time=window.index[-(lb - pl1[0])],
                    end_time=window.index[-1],
                    confidence=0.80,
                    metadata={
                        "strategy": self.name,
                        "direction": "LONG",
                        "price_low1": float(pl1[1]),
                        "price_low2": float(pl2[1]),
                        "rsi_low1": float(rl1[1]),
                        "rsi_low2": float(rl2[1]),
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Bearish divergence: price higher high + RSI lower high
        price_highs = self._find_swing_highs(price_hb)
        rsi_highs = self._find_swing_highs(rsi_lb)

        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            ph1, ph2 = price_highs[-2], price_highs[-1]
            rh1, rh2 = rsi_highs[-2], rsi_highs[-1]
            if ph2[1] > ph1[1] and rh2[1] < rh1[1]:
                detected.append(PatternSignal(
                    name=f"{self.name}_SHORT",
                    start_time=window.index[-(lb - ph1[0])],
                    end_time=window.index[-1],
                    confidence=0.80,
                    metadata={
                        "strategy": self.name,
                        "direction": "SHORT",
                        "price_high1": float(ph1[1]),
                        "price_high2": float(ph2[1]),
                        "rsi_high1": float(rh1[1]),
                        "rsi_high2": float(rh2[1]),
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
