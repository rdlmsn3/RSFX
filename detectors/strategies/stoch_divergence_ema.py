"""
detectors/strategies/stoch_divergence_ema.py
---------------------------------------------
Stochastic divergence + EMA 9/21 trend confirmation (M5 only).

Rules:
  LONG:  Price makes lower low + Stochastic makes higher low (bullish div)
         + EMA9 > EMA21
  SHORT: Price makes higher high + Stochastic makes lower high (bearish div)
         + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. Stoch Divergence EMA signals disabled.")


class StochDivergenceEMAStrategy(BaseStrategy):
    """Stochastic divergence with EMA 9/21 trend filter (M5)."""

    name = "stoch_divergence_ema"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        lookback: int = 30,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
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
        min_bars = max(self.ema_slow, self.stoch_k + self.stoch_smooth) + self.lookback + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # EMAs
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # Stochastic
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            fastd_period=self.stoch_d,
            slowk_period=self.stoch_smooth,
            slowd_period=self.stoch_smooth,
        )

        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # Lookback window
        lb = self.lookback
        price_lb = close[-lb:]
        stoch_lb = slowk[-lb:]

        # Bullish divergence: price lower low + Stoch higher low
        price_lows = self._find_swing_lows(price_lb)
        stoch_lows = self._find_swing_lows(stoch_lb)

        if len(price_lows) >= 2 and len(stoch_lows) >= 2:
            pl1, pl2 = price_lows[-2], price_lows[-1]
            sl1, sl2 = stoch_lows[-2], stoch_lows[-1]

            if pl2[1] < pl1[1] and sl2[1] > sl1[1] and uptrend:
                detected.append(PatternSignal(
                    name=f"{self.name}_LONG",
                    start_time=window.index[-(lb - pl1[0])],
                    end_time=window.index[-1],
                    confidence=0.82,
                    metadata={
                        "strategy": self.name,
                        "direction": "LONG",
                        "price_low1": float(pl1[1]),
                        "price_low2": float(pl2[1]),
                        "stoch_low1": float(sl1[1]),
                        "stoch_low2": float(sl2[1]),
                        "ema_fast": float(ema_f[-1]),
                        "ema_slow": float(ema_s[-1]),
                    },
                ))
                logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Bearish divergence: price higher high + Stoch lower high
        price_highs = self._find_swing_highs(price_lb)
        stoch_highs = self._find_swing_highs(stoch_lb)

        if len(price_highs) >= 2 and len(stoch_highs) >= 2:
            ph1, ph2 = price_highs[-2], price_highs[-1]
            sh1, sh2 = stoch_highs[-2], stoch_highs[-1]

            if ph2[1] > ph1[1] and sh2[1] < sh1[1] and downtrend:
                detected.append(PatternSignal(
                    name=f"{self.name}_SHORT",
                    start_time=window.index[-(lb - ph1[0])],
                    end_time=window.index[-1],
                    confidence=0.82,
                    metadata={
                        "strategy": self.name,
                        "direction": "SHORT",
                        "price_high1": float(ph1[1]),
                        "price_high2": float(ph2[1]),
                        "stoch_high1": float(sh1[1]),
                        "stoch_high2": float(sh2[1]),
                        "ema_fast": float(ema_f[-1]),
                        "ema_slow": float(ema_s[-1]),
                    },
                ))
                logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
