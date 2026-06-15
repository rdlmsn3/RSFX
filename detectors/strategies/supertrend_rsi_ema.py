"""
detectors/strategies/supertrend_rsi_ema.py
------------------------------------------
Supertrend (ATR-based) + RSI momentum + EMA alignment (M5 only).

Rules:
  LONG:  Supertrend bullish + RSI > 50 + EMA9 > EMA21
  SHORT: Supertrend bearish + RSI < 50 + EMA9 < EMA21

Supertrend is implemented manually using ATR:
  basic_upper = (high + low) / 2 + multiplier * ATR
  basic_lower = (high + low) / 2 - multiplier * ATR
  Final bands trail with closing price.
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
    logger.warning("TA-Lib not installed. SupertrendRsiEma strategy signals disabled.")


def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Supertrend indicator.

    Returns
    -------
    supertrend_line : np.ndarray
        The Supertrend line value at each bar.
    direction : np.ndarray
        +1 for bullish (uptrend), -1 for bearish (downtrend).
    """
    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2.0

    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    direction = np.ones(n, dtype=np.float64)

    # Final band values that trail
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    st_line = np.full(n, np.nan)

    for i in range(period, n):
        # Basic bands
        basic_upper = hl2[i] + multiplier * atr[i]
        basic_lower = hl2[i] - multiplier * atr[i]

        # Final upper: use current basic unless previous close was above it
        if np.isnan(final_upper[i - 1]):
            final_upper[i] = basic_upper
        else:
            if close[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper
            else:
                final_upper[i] = max(basic_upper, final_upper[i - 1])

        # Final lower: use current basic unless previous close was below it
        if np.isnan(final_lower[i - 1]):
            final_lower[i] = basic_lower
        else:
            if close[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower
            else:
                final_lower[i] = min(basic_lower, final_lower[i - 1])

        # Direction logic
        if i == period:
            direction[i] = 1.0 if close[i] > final_upper[i] else -1.0
        else:
            prev_dir = direction[i - 1]
            if prev_dir == -1.0:  # was bearish
                direction[i] = 1.0 if close[i] > final_upper[i] else -1.0
            else:  # was bullish
                direction[i] = -1.0 if close[i] < final_lower[i] else 1.0

        st_line[i] = final_lower[i] if direction[i] == 1.0 else final_upper[i]

    return st_line, direction


class SupertrendRsiEmaStrategy(BaseStrategy):
    """Supertrend (ATR-based) + RSI + EMA alignment confluence."""

    name = "supertrend_rsi_ema"

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        rsi_period: int = 14,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = max(self.st_period * 2, self.ema_slow, self.rsi_period) + 5
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        st_line, st_dir = supertrend(high, low, close, period=self.st_period, multiplier=self.st_multiplier)
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        if np.isnan(st_dir[-1]) or np.isnan(rsi[-1]):
            return detected

        # --- Current conditions ---
        st_bull = st_dir[-1] == 1.0
        st_bear = st_dir[-1] == -1.0
        rsi_bull = rsi[-1] > 50
        rsi_bear = rsi[-1] < 50
        ema_bull = ema_f[-1] > ema_s[-1]
        ema_bear = ema_f[-1] < ema_s[-1]

        # --- Generate signals ---
        if st_bull and rsi_bull and ema_bull:
            confidence = self._calc_confidence(st_dir[-1], rsi[-1], ema_f[-1], ema_s[-1])
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "supertrend_value": float(st_line[-1]),
                    "supertrend_dir": "BULLISH",
                    "rsi": float(rsi[-1]),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info(
                "LONG signal at %s (strategy=%s, RSI=%.1f, ST=BULLISH)",
                current_timestamp, self.name, rsi[-1],
            )

        elif st_bear and rsi_bear and ema_bear:
            confidence = self._calc_confidence(st_dir[-1], rsi[-1], ema_f[-1], ema_s[-1])
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "supertrend_value": float(st_line[-1]),
                    "supertrend_dir": "BEARISH",
                    "rsi": float(rsi[-1]),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info(
                "SHORT signal at %s (strategy=%s, RSI=%.1f, ST=BEARISH)",
                current_timestamp, self.name, rsi[-1],
            )

        return detected

    @staticmethod
    def _calc_confidence(st_dir: float, rsi: float, ema_f: float, ema_s: float) -> float:
        """Confidence based on RSI distance from 50 and EMA separation."""
        rsi_dist = abs(rsi - 50) / 50.0
        ema_sep = abs(ema_f - ema_s) / ema_s if ema_s != 0 else 0
        raw = 0.5 + min(rsi_dist * 0.25, 0.25) + min(ema_sep, 0.25)
        return round(min(raw, 1.0), 2)
