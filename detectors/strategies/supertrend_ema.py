"""
detectors/strategies/supertrend_ema.py
--------------------------------------
Supertrend + EMA 50 trend following strategy (M5 only).

Rules:
  LONG:  Supertrend flips bullish + price > EMA50
  SHORT: Supertrend flips bearish + price < EMA50

Supertrend is calculated manually using ATR.
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
    logger.warning("TA-Lib not installed. SupertrendEma strategy signals disabled.")


class SupertrendEmaStrategy(BaseStrategy):
    """Supertrend flip + EMA 50 trend filter (M5 only)."""

    name = "supertrend_ema"

    def __init__(
        self,
        atr_period: int = 10,
        atr_mult: float = 3.0,
        ema_trend: int = 50,
    ) -> None:
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.ema_trend = ema_trend

    @staticmethod
    def _compute_supertrend(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr_period: int,
        atr_mult: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Supertrend and direction arrays.

        Returns
        -------
        supertrend : np.ndarray
            Supertrend line values.
        direction : np.ndarray
            +1 for bullish, -1 for bearish.
        """
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        # ATR via Wilder's smoothing
        atr = np.empty(len(close), dtype=np.float64)
        atr[:] = np.nan
        atr[1] = np.mean(tr[:atr_period]) if atr_period <= len(tr) else tr[0]
        for i in range(2, len(close)):
            if i - 1 < len(tr):
                atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i - 1]) / atr_period

        hl2 = (high + low) / 2.0
        upper_band = hl2 + atr_mult * atr
        lower_band = hl2 - atr_mult * atr

        supertrend = np.empty(len(close), dtype=np.float64)
        direction = np.zeros(len(close), dtype=np.float64)

        supertrend[0] = upper_band[0]
        direction[0] = -1.0

        for i in range(1, len(close)):
            # Adjust bands
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                pass
            else:
                lower_band[i] = lower_band[i - 1]

            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                pass
            else:
                upper_band[i] = upper_band[i - 1]

            # Direction logic
            if direction[i - 1] == 1.0:
                if close[i] < lower_band[i]:
                    direction[i] = -1.0
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1.0
                    supertrend[i] = lower_band[i]
            else:
                if close[i] > upper_band[i]:
                    direction[i] = 1.0
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1.0
                    supertrend[i] = upper_band[i]

        return supertrend, direction

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < max(self.atr_period + 2, self.ema_trend + 1):
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # EMA 50 for trend filter
        ema50 = talib.EMA(close, timeperiod=self.ema_trend)

        # Supertrend
        supertrend, direction = self._compute_supertrend(high, low, close, self.atr_period, self.atr_mult)

        # Check for flip on the latest bar
        if len(direction) < 2 or np.isnan(ema50[-1]):
            return detected

        flipped_bullish = direction[-1] == 1.0 and direction[-2] == -1.0
        flipped_bearish = direction[-1] == -1.0 and direction[-2] == 1.0

        current_close = close[-1]

        # --- LONG ---
        if flipped_bullish and current_close > ema50[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "supertrend": supertrend[-1],
                    "ema50": ema50[-1],
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # --- SHORT ---
        elif flipped_bearish and current_close < ema50[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "supertrend": supertrend[-1],
                    "ema50": ema50[-1],
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
