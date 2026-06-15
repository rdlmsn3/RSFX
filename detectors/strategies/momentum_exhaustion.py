"""
detectors/strategies/momentum_exhaustion.py
-------------------------------------------
Strong momentum candle + Stochastic extreme (M5 only) — Group 12 Hybrid.

Rules:
  LONG:  large bullish candle (body > 70% of range) + Stoch > 80 (extreme overbought exhaustion)
  SHORT: large bearish candle (body > 70% of range) + Stoch < 20 (extreme oversold exhaustion)
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
    logger.warning("TA-Lib not installed. Momentum Exhaustion signals disabled.")


class MomentumExhaustionStrategy(BaseStrategy):
    """Strong momentum candle + Stoch extreme on M5."""

    name = "momentum_exhaustion"

    def __init__(
        self,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
        body_range_pct: float = 0.70,
        stoch_overbought: float = 80.0,
        stoch_oversold: float = 20.0,
    ) -> None:
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_smooth = stoch_smooth
        self.body_range_pct = body_range_pct
        self.stoch_overbought = stoch_overbought
        self.stoch_oversold = stoch_oversold

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = self.stoch_k + self.stoch_smooth + 5
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        open_ = window["open"].values.astype(np.float64)
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

        current_close = close[-1]
        current_open = open_[-1]
        current_high = high[-1]
        current_low = low[-1]

        body = abs(current_close - current_open)
        candle_range = current_high - current_low

        if candle_range == 0:
            return detected

        body_pct = body / candle_range
        stoch_val = slowk[-1]

        is_bullish = current_close > current_open
        is_bearish = current_close < current_open

        # Long: large bullish candle + Stoch > overbought (exhaustion after strong move)
        if is_bullish and body_pct > self.body_range_pct and stoch_val > self.stoch_overbought:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "body_pct": float(body_pct),
                    "stoch": float(stoch_val),
                    "candle_range": float(candle_range),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: large bearish candle + Stoch < oversold (exhaustion after strong drop)
        if is_bearish and body_pct > self.body_range_pct and stoch_val < self.stoch_oversold:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.75,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "body_pct": float(body_pct),
                    "stoch": float(stoch_val),
                    "candle_range": float(candle_range),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
