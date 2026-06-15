"""
detectors/strategies/pivot_rsi_bounce.py
-----------------------------------------
Pivot point bounce + RSI extreme confirmation.

Rules:
  Calculate daily pivots from H1 data.
  LONG:  price touches S1 + RSI < 35
  SHORT: price touches R1 + RSI > 65
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
    logger.warning("TA-Lib not installed. PivotRsiBounce strategy signals disabled.")


class PivotRsiBounceStrategy(BaseStrategy):
    """Pivot point (S1/R1) bounce with RSI extreme confirmation."""

    name = "pivot_rsi_bounce"

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_long: float = 35.0,
        rsi_short: float = 65.0,
        tolerance_pct: float = 0.05,
    ) -> None:
        self.rsi_period = rsi_period
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.tolerance_pct = tolerance_pct

    def _calc_pivots(self, prev_high: float, prev_low: float, prev_close: float) -> dict:
        """Calculate classic pivot points from previous day's H/L/C."""
        pp = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2.0 * pp - prev_low
        s1 = 2.0 * pp - prev_high
        r2 = pp + (prev_high - prev_low)
        s2 = pp - (prev_high - prev_low)
        return {"PP": pp, "R1": r1, "R2": r2, "S1": s1, "S2": s2}

    def _get_previous_day_hlc(self, h1_window: pd.DataFrame, current_timestamp: pd.Timestamp) -> tuple[float, float, float] | None:
        """Get previous trading day's high, low, close from H1 data."""
        prev_day = current_timestamp.normalize() - pd.Timedelta(days=1)
        day_data = h1_window[h1_window.index.normalize() == prev_day]
        if len(day_data) < 4:
            dates = h1_window.index.normalize().unique()
            dates = dates[dates < current_timestamp.normalize()]
            if len(dates) < 1:
                return None
            prev_day = dates[-1]
            day_data = h1_window[h1_window.index.normalize() == prev_day]
            if len(day_data) < 4:
                return None
        return float(day_data["high"].max()), float(day_data["low"].min()), float(day_data["close"].iloc[-1])

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        h1_window = windows.get("H1")
        if window is None or not TA_AVAILABLE or len(window) < self.rsi_period + 5:
            return detected
        if h1_window is None or len(h1_window) < 24:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        rsi = talib.RSI(close, timeperiod=self.rsi_period)

        if np.isnan(rsi[-1]):
            return detected

        # --- Pivot points from previous day ---
        prev_hlc = self._get_previous_day_hlc(h1_window, current_timestamp)
        if prev_hlc is None:
            return detected

        prev_high, prev_low, prev_close = prev_hlc
        pivots = self._calc_pivots(prev_high, prev_low, prev_close)

        current_close = close[-1]
        current_low = window["low"].values[-1]
        current_high = window["high"].values[-1]

        # --- Bounce detection ---
        s1 = pivots["S1"]
        r1 = pivots["R1"]

        # Price touches S1 zone
        s1_touch = abs(current_low - s1) / s1 * 100 <= self.tolerance_pct
        # Price touches R1 zone
        r1_touch = abs(current_high - r1) / r1 * 100 <= self.tolerance_pct

        # --- Generate signals ---
        if s1_touch and rsi[-1] < self.rsi_long:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "pivot_pp": float(pivots["PP"]),
                    "pivot_s1": float(s1),
                    "pivot_r1": float(r1),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif r1_touch and rsi[-1] > self.rsi_short:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "pivot_pp": float(pivots["PP"]),
                    "pivot_s1": float(s1),
                    "pivot_r1": float(r1),
                    "rsi": float(rsi[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
