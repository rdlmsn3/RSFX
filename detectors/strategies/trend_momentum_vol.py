"""
detectors/strategies/trend_momentum_vol.py
--------------------------------------------
EMA trend + Stochastic momentum + BB expansion (M5 only).

Rules:
  LONG:  EMA9 > EMA21 + Stochastic > 20 + BB expanding
  SHORT: EMA9 < EMA21 + Stochastic < 80 + BB expanding
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
    logger.warning("TA-Lib not installed. TrendMomentumVol strategy signals disabled.")


class TrendMomentumVolStrategy(BaseStrategy):
    """EMA trend + Stochastic momentum + Bollinger Band expansion confluence."""

    name = "trend_momentum_vol"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 5,
        stoch_d: int = 3,
        stoch_smooth: int = 3,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_smooth = stoch_smooth

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = max(self.ema_slow, self.bb_period) + 2
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        # --- Indicators ---
        ema_f = talib.EMA(close, timeperiod=self.ema_fast)
        ema_s = talib.EMA(close, timeperiod=self.ema_slow)
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
        )
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
        )

        # --- Trend conditions ---
        uptrend = ema_f[-1] > ema_s[-1]
        downtrend = ema_f[-1] < ema_s[-1]

        # --- BB expansion detection ---
        bb_width_now = upper[-1] - lower[-1]
        bb_width_prev = upper[-2] - lower[-2]
        bb_expanding = bb_width_now > bb_width_prev

        # --- Stochastic conditions ---
        stoch_above_20 = slowk[-1] > 20
        stoch_below_80 = slowk[-1] < 80

        # --- Generate signals ---
        if uptrend and stoch_above_20 and bb_expanding:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.9,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                    "stoch_k": float(slowk[-1]),
                    "stoch_d": float(slowd[-1]),
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width_now),
                    "bb_expanding": bb_expanding,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s, trend+momentum+expansion)", current_timestamp, self.name)

        elif downtrend and stoch_below_80 and bb_expanding:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=0.9,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "ema_fast": float(ema_f[-1]),
                    "ema_slow": float(ema_s[-1]),
                    "stoch_k": float(slowk[-1]),
                    "stoch_d": float(slowd[-1]),
                    "bb_lower": float(lower[-1]),
                    "bb_middle": float(middle[-1]),
                    "bb_upper": float(upper[-1]),
                    "bb_width": float(bb_width_now),
                    "bb_expanding": bb_expanding,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s, trend+momentum+expansion)", current_timestamp, self.name)

        return detected
