"""
detectors/strategies/gap_fill.py
--------------------------------
Gap at session open + fade (M5 only) — Group 12 Hybrid.

Rules:
  Detect gap between previous bar's close and current bar's open.
  LONG:  gap down (current open < previous close) + RSI < 35 (fade up expected)
  SHORT: gap up (current open > previous close) + RSI > 65 (fade down expected)
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
    logger.warning("TA-Lib not installed. Gap Fill signals disabled.")


class GapFillStrategy(BaseStrategy):
    """Gap at session open + fade on M5."""

    name = "gap_fill"

    def __init__(
        self,
        rsi_period: int = 14,
        gap_threshold_pct: float = 0.0005,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
    ) -> None:
        self.rsi_period = rsi_period
        self.gap_threshold_pct = gap_threshold_pct
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = self.rsi_period + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        open_ = window["open"].values.astype(np.float64)

        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        prev_close = close[-2]
        current_open = open_[-1]
        current_rsi = rsi[-1]

        if prev_close == 0:
            return detected

        gap_pct = (current_open - prev_close) / prev_close

        # Gap down + RSI oversold → fade up (LONG)
        if gap_pct < -self.gap_threshold_pct and current_rsi < self.rsi_oversold:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "prev_close": float(prev_close),
                    "current_open": float(current_open),
                    "gap_pct": float(gap_pct),
                    "rsi": float(current_rsi),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s) gap_down=%.4f rsi=%.1f",
                        current_timestamp, self.name, gap_pct, current_rsi)

        # Gap up + RSI overbought → fade down (SHORT)
        if gap_pct > self.gap_threshold_pct and current_rsi > self.rsi_overbought:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "prev_close": float(prev_close),
                    "current_open": float(current_open),
                    "gap_pct": float(gap_pct),
                    "rsi": float(current_rsi),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s) gap_up=%.4f rsi=%.1f",
                        current_timestamp, self.name, gap_pct, current_rsi)

        return detected
