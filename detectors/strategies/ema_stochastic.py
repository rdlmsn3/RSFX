"""
detectors/strategies/ema_stochastic.py
---------------------------------------
Trend + Momentum + Candlestick trigger strategy (M1 only).

Rules:
  LONG:  uptrend (EMA9 > EMA21) + stochastic oversold + bullish candle trigger
  SHORT: downtrend (EMA9 < EMA21) + stochastic overbought + bearish candle trigger

Note: This strategy uses only M1 data. For multi-timeframe confirmation,
use EMAStochasticMTFStrategy instead.
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
    logger.warning("TA-Lib not installed. EMAStochastic strategy signals disabled.")


class EMAStochasticStrategy(BaseStrategy):
    """9 EMA / 21 EMA trend + Fast Stochastic + candlestick trigger (M1 only)."""

    name = "ema_stochastic"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        stoch_k: int = 5,
        stoch_d: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        oscillator_lookback: int = 5,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.oversold = oversold
        self.overbought = overbought
        self.oscillator_lookback = oscillator_lookback

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M1")
        if window is None or not TA_AVAILABLE or len(window) < self.ema_slow + 1:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        open_p = window["open"].values.astype(np.float64)

        # --- Trend: EMA crossover ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # --- Momentum: Fast Stochastic ---
        fastk, _ = talib.STOCHF(high, low, close, fastk_period=self.stoch_k, fastd_period=self.stoch_d)
        oversold_recent = bool(np.any(fastk[-self.oscillator_lookback:] < self.oversold))
        overbought_recent = bool(np.any(fastk[-self.oscillator_lookback:] > self.overbought))

        # --- Candlestick triggers ---
        engulfing = talib.CDLENGULFING(open_p, high, low, close)
        hammer = talib.CDLHAMMER(open_p, high, low, close)
        shooting_star = talib.CDLSHOOTINGSTAR(open_p, high, low, close)

        is_bullish = (engulfing[-1] == 100) or (hammer[-1] == 100)
        is_bearish = (engulfing[-1] == -100) or (shooting_star[-1] == -100)

        # --- Generate signal ---
        if uptrend and oversold_recent and is_bullish:
            pattern = "bullish_engulfing" if engulfing[-1] == 100 else "hammer"
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "trend": "up",
                    "ema_fast": ema_f[-1],
                    "ema_slow": ema_s[-1],
                    "stoch_fastk": fastk[-1],
                    "candle_trigger_idx": window.index[-1],
                    "pattern": pattern,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        elif downtrend and overbought_recent and is_bearish:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "trend": "down",
                    "ema_fast": ema_f[-1],
                    "ema_slow": ema_s[-1],
                    "stoch_fastk": fastk[-1],
                    "candle_trigger_idx": window.index[-1],
                    "pattern": "bearish_candle",
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
