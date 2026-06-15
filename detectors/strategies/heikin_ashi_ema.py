"""
detectors/strategies/heikin_ashi_ema.py
----------------------------------------
Heikin Ashi color change + EMA 9/21 trend strategy (M5 only).

Rules:
  LONG:  Heikin Ashi turns green (HA close > HA open) + EMA9 > EMA21
  SHORT: Heikin Ashi turns red   (HA close < HA open) + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. Heikin Ashi EMA signals disabled.")


class HeikinAshiEMAStrategy(BaseStrategy):
    """Heikin Ashi candle color change + EMA 9/21 trend confirmation (M5)."""

    name = "heikin_ashi_ema"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    @staticmethod
    def _compute_heikin_ashi(open_: np.ndarray, high: np.ndarray,
                             low: np.ndarray, close: np.ndarray):
        """Calculate Heikin Ashi OHLC from standard candles."""
        ha_close = (open_ + high + low + close) / 4.0

        ha_open = np.empty_like(close)
        ha_open[0] = (open_[0] + close[0]) / 2.0
        for i in range(1, len(close)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

        ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

        return ha_open, ha_high, ha_low, ha_close

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ema_slow, 30) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        open_ = window["open"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        close = window["close"].values.astype(np.float64)

        # Heikin Ashi
        ha_open, ha_high, ha_low, ha_close = self._compute_heikin_ashi(
            open_, high, low, close
        )

        # EMA 9 / 21
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)

        # Current bar HA color
        ha_green = ha_close[-1] > ha_open[-1]
        ha_red = ha_close[-1] < ha_open[-1]

        # Previous bar HA color (ensure a color change)
        ha_prev_green = ha_close[-2] > ha_open[-2]
        ha_prev_red = ha_close[-2] < ha_open[-2]

        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # Long: HA turns green + EMA9 > EMA21
        if ha_green and not ha_prev_green and uptrend:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ha_close": float(ha_close[-1]),
                    "ha_open": float(ha_open[-1]),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: HA turns red + EMA9 < EMA21
        elif ha_red and not ha_prev_red and downtrend:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ha_close": float(ha_close[-1]),
                    "ha_open": float(ha_open[-1]),
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
