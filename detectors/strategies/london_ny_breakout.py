"""
detectors/strategies/london_ny_breakout.py
-------------------------------------------
Session range breakout + EMA confirmation.

Rules:
  Session determined from timestamp hour (London: 7-15, NY: 12-20).
  LONG:  price breaks above session high + EMA9 > EMA21
  SHORT: price breaks below session low  + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. LondonNyBreakout strategy signals disabled.")


class LondonNyBreakoutStrategy(BaseStrategy):
    """London/NY session range breakout with EMA trend confirmation."""

    name = "london_ny_breakout"

    def __init__(
        self,
        session_lookback_minutes: int = 15,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.session_lookback_minutes = session_lookback_minutes
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def _get_session_mask(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Return boolean mask for bars within the active session window.

        London: hours 7-15, NY: hours 12-20.  The active session is the
        union (hours 7-20).
        """
        hours = idx.hour
        return (hours >= 7) & (hours <= 20)

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        if window is None or not TA_AVAILABLE or len(window) < self.ema_slow + 5:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        if np.isnan(ema_f[-1]) or np.isnan(ema_s[-1]):
            return detected

        # --- Determine session range ---
        # Use bars within session hours (7-20) from recent history
        session_mask = self._get_session_mask(window.index)
        lookback_bars = min(self.session_lookback_minutes // 5, len(window) - 1)
        if lookback_bars < 1:
            return detected

        recent_mask = pd.Series(False, index=window.index)
        recent_mask.iloc[-lookback_bars:] = True
        combined_mask = session_mask & recent_mask

        if combined_mask.sum() < 2:
            return detected

        session_high = high[combined_mask.values].max()
        session_low = low[combined_mask.values].min()

        # --- Breakout detection ---
        current_close = close[-1]
        current_high = high[-1]
        current_low = low[-1]

        # Break above session high
        break_above = current_high > session_high and current_close > session_high
        # Break below session low
        break_below = current_low < session_low and current_close < session_low

        # --- Generate signals ---
        if break_above and ema_f[-1] > ema_s[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "session_high": float(session_high),
                    "session_low": float(session_low),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif break_below and ema_f[-1] < ema_s[-1]:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "session_high": float(session_high),
                    "session_low": float(session_low),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
