"""
detectors/strategies/macd_divergence.py
---------------------------------------
MACD histogram divergence from price (M5 only).

Rules:
  LONG:  price makes lower low + MACD histogram makes higher low  (bullish divergence)
  SHORT: price makes higher high + MACD histogram makes lower high (bearish divergence)
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
    logger.warning("TA-Lib not installed. MACD Divergence signals disabled.")


class MACDDivergenceStrategy(BaseStrategy):
    """MACD histogram divergence from price on M5."""

    name = "macd_divergence"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        lookback: int = 20,
    ) -> None:
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
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
        min_bars = self.macd_slow + self.macd_signal + self.lookback + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        _macd, _signal, hist = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        lb = self.lookback
        price_lb = low[-lb:]
        price_hb = high[-lb:]
        hist_lb = hist[-lb:]

        # Bullish divergence: price lower low + MACD histogram higher low
        price_lows = self._find_swing_lows(price_lb)
        hist_lows = self._find_swing_lows(hist_lb)

        if len(price_lows) >= 2 and len(hist_lows) >= 2:
            pl1, pl2 = price_lows[-2], price_lows[-1]
            hl1, hl2 = hist_lows[-2], hist_lows[-1]
            if pl2[1] < pl1[1] and hl2[1] > hl1[1]:
                detected.append(PatternSignal(
                    name=f"{self.name}_LONG",
                    start_time=window.index[-(lb - pl1[0])],
                    end_time=window.index[-1],
                    confidence=0.83,
                    metadata={
                        "strategy": self.name,
                        "direction": "LONG",
                        "price_low1": float(pl1[1]),
                        "price_low2": float(pl2[1]),
                        "hist_low1": float(hl1[1]),
                        "hist_low2": float(hl2[1]),
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Bearish divergence: price higher high + MACD histogram lower high
        price_highs = self._find_swing_highs(price_hb)
        hist_highs = self._find_swing_highs(hist_lb)

        if len(price_highs) >= 2 and len(hist_highs) >= 2:
            ph1, ph2 = price_highs[-2], price_highs[-1]
            hh1, hh2 = hist_highs[-2], hist_highs[-1]
            if ph2[1] > ph1[1] and hh2[1] < hh1[1]:
                detected.append(PatternSignal(
                    name=f"{self.name}_SHORT",
                    start_time=window.index[-(lb - ph1[0])],
                    end_time=window.index[-1],
                    confidence=0.83,
                    metadata={
                        "strategy": self.name,
                        "direction": "SHORT",
                        "price_high1": float(ph1[1]),
                        "price_high2": float(ph2[1]),
                        "hist_high1": float(hh1[1]),
                        "hist_high2": float(hh2[1]),
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
