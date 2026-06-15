"""
detectors/strategies/pivot_ema_bounce.py
-----------------------------------------
Pivot point bounce + EMA confirmation.

Rules:
  Calculate daily pivots from H1 data (previous day's H/L/C).
  LONG:  price bounces from S1 + EMA9 > EMA21
  SHORT: price rejects from R1 + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. PivotEmaBounce strategy signals disabled.")


class PivotEmaBounceStrategy(BaseStrategy):
    """Pivot point (S1/R1) bounce with EMA trend confirmation."""

    name = "pivot_ema_bounce"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        tolerance_pct: float = 0.05,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
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
            # Try last available complete day before current
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
        if window is None or not TA_AVAILABLE or len(window) < self.ema_slow + 5:
            return detected
        if h1_window is None or len(h1_window) < 24:
            return detected

        close = window["close"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]):
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
        # Check if price touched S1 zone and bounced
        s1 = pivots["S1"]
        s1_zone = abs(current_low - s1) / s1 * 100 <= self.tolerance_pct
        bounce_from_s1 = s1_zone and current_close > current_low

        # Check if price touched R1 zone and rejected
        r1 = pivots["R1"]
        r1_zone = abs(current_high - r1) / r1 * 100 <= self.tolerance_pct
        reject_from_r1 = r1_zone and current_close < current_high

        # --- Generate signals ---
        if bounce_from_s1 and ema_f[-1] > ema_s[-1]:
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
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif reject_from_r1 and ema_f[-1] < ema_s[-1]:
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
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
