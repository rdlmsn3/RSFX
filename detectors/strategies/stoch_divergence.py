"""
detectors/strategies/stoch_divergence.py
----------------------------------------
Stochastic divergence from price (M5 only).

Rules:
  LONG:  price makes lower low + Stoch makes higher low  (bullish divergence)
  SHORT: price makes higher high + Stoch makes lower high (bearish divergence)
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
    logger.warning("TA-Lib not installed. Stochastic Divergence signals disabled.")


class StochDivergenceStrategy(BaseStrategy):
    """Stochastic divergence from price on M5."""

    name = "stoch_divergence"

    def __init__(
        self,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        lookback: int = 20,
    ) -> None:
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_smooth = stoch_smooth
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
        min_bars = self.stoch_k + self.stoch_smooth + self.lookback + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        slowk, _slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
        )

        lb = self.lookback
        price_lb = low[-lb:]
        price_hb = high[-lb:]
        stoch_lb = slowk[-lb:]

        # Bullish divergence: price lower low + Stoch higher low
        price_lows = self._find_swing_lows(price_lb)
        stoch_lows = self._find_swing_lows(stoch_lb)

        if len(price_lows) >= 2 and len(stoch_lows) >= 2:
            pl1, pl2 = price_lows[-2], price_lows[-1]
            sl1, sl2 = stoch_lows[-2], stoch_lows[-1]
            if pl2[1] < pl1[1] and sl2[1] > sl1[1]:
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
                        "stoch_low1": float(sl1[1]),
                        "stoch_low2": float(sl2[1]),
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Bearish divergence: price higher high + Stoch lower high
        price_highs = self._find_swing_highs(price_hb)
        stoch_highs = self._find_swing_highs(stoch_lb)

        if len(price_highs) >= 2 and len(stoch_highs) >= 2:
            ph1, ph2 = price_highs[-2], price_highs[-1]
            sh1, sh2 = stoch_highs[-2], stoch_highs[-1]
            if ph2[1] > ph1[1] and sh2[1] < sh1[1]:
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
                        "stoch_high1": float(sh1[1]),
                        "stoch_high2": float(sh2[1]),
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
