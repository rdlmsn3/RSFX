"""
detectors/strategies/ema_stochastic_mtf.py
-------------------------------------------
Multi-timeframe version: 3-layer confirmation strategy.

Layer 1 (H1): Trend bias — EMA9/EMA21 crossover determines direction
Layer 2 (M5): Momentum — Fast Stochastic oversold/overbought confirms setup
Layer 3 (M1): Entry trigger — candlestick pattern fires the signal

Signal fires ONLY when all 3 layers agree.
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
    logger.warning("TA-Lib not installed. EMAStochasticMTF strategy signals disabled.")


class EMAStochasticMTFStrategy(BaseStrategy):
    """
    3-layer MTF confirmation:
      H1 = trend direction
      M5 = momentum confirmation
      M1 = entry trigger (candlestick pattern)
    """

    name = "ema_stochastic_mtf"

    def __init__(
        self,
        # H1 trend
        h1_ema_fast: int = 9,
        h1_ema_slow: int = 21,
        # M5 momentum
        m5_stoch_k: int = 5,
        m5_stoch_d: int = 3,
        m5_oversold: float = 20.0,
        m5_overbought: float = 80.0,
        # Required TFs
        required_tfs: tuple[str, ...] = ("M1", "M5", "H1"),
    ) -> None:
        self.h1_ema_fast = h1_ema_fast
        self.h1_ema_slow = h1_ema_slow
        self.m5_stoch_k = m5_stoch_k
        self.m5_stoch_d = m5_stoch_d
        self.m5_oversold = m5_oversold
        self.m5_overbought = m5_overbought
        self.required_tfs = required_tfs

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []

        if not TA_AVAILABLE:
            return detected

        # --- Check all required TFs are present ---
        for tf in self.required_tfs:
            if tf not in windows or windows[tf] is None or len(windows[tf]) < 3:
                return detected

        m1 = windows["M1"]
        m5 = windows["M5"]
        h1 = windows["H1"]

        # ==============================================================
        # Layer 1: H1 Trend Bias
        # ==============================================================
        h1_close = h1["close"].values.astype(np.float64)
        h1_ema_f = talib.EMA(h1_close, timeperiod=self.h1_ema_fast)
        h1_ema_s = talib.EMA(h1_close, timeperiod=self.h1_ema_slow)

        h1_uptrend = h1_ema_f[-1] > h1_ema_s[-1]
        h1_downtrend = h1_ema_f[-1] < h1_ema_s[-1]

        if not h1_uptrend and not h1_downtrend:
            return detected  # H1 in no-man's land

        h1_bias = "LONG" if h1_uptrend else "SHORT"

        # ==============================================================
        # Layer 2: M5 Momentum Confirmation
        # ==============================================================
        m5_high = m5["high"].values.astype(np.float64)
        m5_low = m5["low"].values.astype(np.float64)
        m5_close = m5["close"].values.astype(np.float64)

        m5_fastk, _ = talib.STOCHF(
            m5_high, m5_low, m5_close,
            fastk_period=self.m5_stoch_k,
            fastd_period=self.m5_stoch_d,
        )

        # Check last 2 M5 candles for oversold/overbought
        m5_oversold = (m5_fastk[-1] < self.m5_oversold) or (m5_fastk[-2] < self.m5_oversold)
        m5_overbought = (m5_fastk[-1] > self.m5_overbought) or (m5_fastk[-2] > self.m5_overbought)

        # M5 must confirm H1 direction:
        #   H1 LONG  → M5 oversold (buying dip in uptrend)
        #   H1 SHORT → M5 overbought (selling rally in downtrend)
        if h1_bias == "LONG" and not m5_oversold:
            return detected
        if h1_bias == "SHORT" and not m5_overbought:
            return detected

        m5_confirms = True

        # ==============================================================
        # Layer 3: M1 Entry Trigger (candlestick pattern)
        # ==============================================================
        m1_close = m1["close"].values.astype(np.float64)
        m1_high = m1["high"].values.astype(np.float64)
        m1_low = m1["low"].values.astype(np.float64)
        m1_open = m1["open"].values.astype(np.float64)

        engulfing = talib.CDLENGULFING(m1_open, m1_high, m1_low, m1_close)
        hammer = talib.CDLHAMMER(m1_open, m1_high, m1_low, m1_close)
        shooting_star = talib.CDLSHOOTINGSTAR(m1_open, m1_high, m1_low, m1_close)

        is_bullish_trigger = (engulfing[-1] == 100) or (hammer[-1] == 100)
        is_bearish_trigger = (engulfing[-1] == -100) or (shooting_star[-1] == -100)

        # ==============================================================
        # Confluence: all 3 layers must agree
        # ==============================================================
        if h1_bias == "LONG" and m5_confirms and is_bullish_trigger:
            pattern = "bullish_engulfing" if engulfing[-1] == 100 else "hammer"
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=m1.index[-1],
                end_time=m1.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    # Layer 1: H1
                    "h1_trend": "up",
                    "h1_ema_fast": h1_ema_f[-1],
                    "h1_ema_slow": h1_ema_s[-1],
                    # Layer 2: M5
                    "m5_stoch_fastk": m5_fastk[-1],
                    "m5_oversold": m5_oversold,
                    # Layer 3: M1
                    "m1_pattern": pattern,
                    "candle_trigger_idx": m1.index[-1],
                },
            ))
            logger.info(
                "MTF LONG at %s | H1=up M5_stoch=%.1f M1=%s",
                current_timestamp, m5_fastk[-1], pattern,
            )

        elif h1_bias == "SHORT" and m5_confirms and is_bearish_trigger:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=m1.index[-1],
                end_time=m1.index[-1],
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    # Layer 1: H1
                    "h1_trend": "down",
                    "h1_ema_fast": h1_ema_f[-1],
                    "h1_ema_slow": h1_ema_s[-1],
                    # Layer 2: M5
                    "m5_stoch_fastk": m5_fastk[-1],
                    "m5_overbought": m5_overbought,
                    # Layer 3: M1
                    "m1_pattern": "bearish_candle",
                    "candle_trigger_idx": m1.index[-1],
                },
            ))
            logger.info(
                "MTF SHORT at %s | H1=down M5_stoch=%.1f M1=bearish",
                current_timestamp, m5_fastk[-1],
            )

        return detected
