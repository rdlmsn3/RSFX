"""
detectors/strategies/macd_histogram_div.py
------------------------------------------
MACD Histogram Divergence strategy (M5 only).

Rules:
  LONG:  price makes lower low  + MACD histogram makes higher low  (bullish divergence)
  SHORT: price makes higher high + MACD histogram makes lower high (bearish divergence)
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
    logger.warning("TA-Lib not installed. MACD Histogram Divergence signals disabled.")


class MACDHistogramDivStrategy(BaseStrategy):
    """MACD histogram divergence: detect price/indicator divergence on M5."""

    name = "macd_histogram_div"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        lookback: int = 30,
    ) -> None:
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.lookback = lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = self.macd_slow + self.macd_signal + self.lookback + 1
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)

        # Compute MACD histogram
        macd, signal_line, hist = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        lb = self.lookback
        recent_low = low[-lb:]
        recent_high = high[-lb:]
        recent_hist = hist[-lb:]

        # Find two swing lows in the lookback window for bullish divergence
        # Simple approach: split lookback into two halves, find min of each
        half = lb // 2
        price_ll1 = np.nanmin(recent_low[:half])
        price_ll2 = np.nanmin(recent_low[half:])
        hist_ll1 = np.nanmin(recent_hist[:half])
        hist_ll2 = np.nanmin(recent_hist[half:])

        # Bearish divergence: two swing highs
        price_hh1 = np.nanmax(recent_high[:half])
        price_hh2 = np.nanmax(recent_high[half:])
        hist_hh1 = np.nanmax(recent_hist[:half])
        hist_hh2 = np.nanmax(recent_hist[half:])

        # Bullish divergence: price lower low + histogram higher low
        if price_ll2 < price_ll1 and hist_ll2 > hist_ll1:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-lb],
                end_time=window.index[-1],
                confidence=0.85,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "type": "bullish_divergence",
                    "price_low1": price_ll1,
                    "price_low2": price_ll2,
                    "hist_low1": hist_ll1,
                    "hist_low2": hist_ll2,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Bearish divergence: price higher high + histogram lower high
        elif price_hh2 > price_hh1 and hist_hh2 < hist_hh1:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-lb],
                end_time=window.index[-1],
                confidence=0.85,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "type": "bearish_divergence",
                    "price_high1": price_hh1,
                    "price_high2": price_hh2,
                    "hist_high1": hist_hh1,
                    "hist_high2": hist_hh2,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
