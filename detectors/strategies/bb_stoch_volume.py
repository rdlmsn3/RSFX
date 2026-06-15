"""
detectors/strategies/bb_stoch_volume.py
---------------------------------------
Bollinger Bands + Stochastic oscillator + Volume spike (M5 only).

Rules:
  LONG:  price touches/crosses lower BB + Stoch %K < 20 + volume spike
  SHORT: price touches/crosses upper BB + Stoch %K > 80 + volume spike
  Volume spike = volume > SMA(volume, 20) * 2.0
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
    logger.warning("TA-Lib not installed. BbStochVolume strategy signals disabled.")


class BbStochVolumeStrategy(BaseStrategy):
    """Bollinger Bands + Stochastic + Volume spike confluence."""

    name = "bb_stoch_volume"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 14,
        stoch_d: int = 3,
        vol_period: int = 20,
        vol_multiplier: float = 2.0,
        stoch_oversold: float = 20.0,
        stoch_overbought: float = 80.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.vol_period = vol_period
        self.vol_multiplier = vol_multiplier
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_len = max(self.bb_period, self.stoch_k + self.stoch_d, self.vol_period) + 5
        if window is None or not TA_AVAILABLE or len(window) < min_len:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)
        volume = window["volume"].values.astype(np.float64)

        # --- Bollinger Bands ---
        upper_bb, middle_bb, lower_bb = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
        )

        # --- Stochastic ---
        stoch_k, stoch_d = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_d,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
        )

        # --- Volume spike detection ---
        vol_sma = talib.SMA(volume, timeperiod=self.vol_period)
        vol_spike = volume[-1] > vol_sma[-1] * self.vol_multiplier

        if np.isnan(vol_sma[-1]) or np.isnan(upper_bb[-1]) or np.isnan(stoch_k[-1]):
            return detected

        # --- Conditions ---
        price = close[-1]
        at_lower_bb = price <= lower_bb[-1]
        at_upper_bb = price >= upper_bb[-1]
        stoch_oversold = stoch_k[-1] < self.stoch_oversold
        stoch_overbought = stoch_k[-1] > self.stoch_overbought

        # --- LONG: lower BB + Stoch oversold + volume spike ---
        if at_lower_bb and stoch_oversold and vol_spike:
            confidence = self._calc_confidence(price, lower_bb[-1], upper_bb[-1], stoch_k[-1], 20)
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "close": float(price),
                    "lower_bb": float(lower_bb[-1]),
                    "upper_bb": float(upper_bb[-1]),
                    "middle_bb": float(middle_bb[-1]),
                    "stoch_k": float(stoch_k[-1]),
                    "stoch_d": float(stoch_d[-1]),
                    "volume": float(volume[-1]),
                    "vol_sma": float(vol_sma[-1]),
                    "vol_spike_ratio": float(volume[-1] / vol_sma[-1]),
                },
            ))
            logger.info(
                "LONG signal at %s (strategy=%s, BB_lower=%.4f, StochK=%.1f)",
                current_timestamp, self.name, lower_bb[-1], stoch_k[-1],
            )

        # --- SHORT: upper BB + Stoch overbought + volume spike ---
        elif at_upper_bb and stoch_overbought and vol_spike:
            confidence = self._calc_confidence(price, lower_bb[-1], upper_bb[-1], stoch_k[-1], 80)
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-1],
                end_time=window.index[-1],
                confidence=confidence,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "close": float(price),
                    "lower_bb": float(lower_bb[-1]),
                    "upper_bb": float(upper_bb[-1]),
                    "middle_bb": float(middle_bb[-1]),
                    "stoch_k": float(stoch_k[-1]),
                    "stoch_d": float(stoch_d[-1]),
                    "volume": float(volume[-1]),
                    "vol_sma": float(vol_sma[-1]),
                    "vol_spike_ratio": float(volume[-1] / vol_sma[-1]),
                },
            ))
            logger.info(
                "SHORT signal at %s (strategy=%s, BB_upper=%.4f, StochK=%.1f)",
                current_timestamp, self.name, upper_bb[-1], stoch_k[-1],
            )

        return detected

    @staticmethod
    def _calc_confidence(price, lower_bb, upper_bb, stoch_k, stoch_extreme) -> float:
        """Confidence scales with how deep into the band and how extreme the stochastic."""
        bb_range = upper_bb - lower_bb if upper_bb != lower_bb else 1.0
        if stoch_extreme < 50:  # oversold
            bb_depth = (price - lower_bb) / bb_range  # negative or zero at lower band
            stoch_ext = (stoch_extreme - stoch_k) / stoch_extreme if stoch_extreme != 0 else 0
        else:  # overbought
            bb_depth = (upper_bb - price) / bb_range  # negative or zero at upper band
            stoch_ext = (stoch_k - stoch_extreme) / (100 - stoch_extreme) if stoch_extreme != 100 else 0

        raw = 0.5 + max(0, -bb_depth) * 0.25 + max(0, stoch_ext) * 0.25
        return round(min(max(raw, 0.0), 1.0), 2)
