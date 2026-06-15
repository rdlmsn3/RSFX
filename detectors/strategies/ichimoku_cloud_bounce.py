"""
detectors/strategies/ichimoku_cloud_bounce.py
----------------------------------------------
Ichimoku cloud bounce + Tenkan/Kijun alignment (M5).

Ichimoku components calculated manually:
  Tenkan-sen = (9-period high + 9-period low) / 2
  Kijun-sen  = (26-period high + 26-period low) / 2
  Senkou A   = (Tenkan + Kijun) / 2  (displaced forward, uses current calc)
  Senkou B   = (52-period high + 52-period low) / 2

Rules:
  LONG:  Price bounces from cloud top + Tenkan > Kijun
  SHORT: Price rejects from cloud bottom + Tenkan < Kijun
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
    logger.warning("TA-Lib not installed. Ichimoku Cloud Bounce signals disabled.")


class IchimokuCloudBounceStrategy(BaseStrategy):
    """Ichimoku cloud bounce with Tenkan/Kijun confirmation (M5)."""

    name = "ichimoku_cloud_bounce"

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        touch_tolerance: float = 0.001,
    ) -> None:
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.touch_tolerance = touch_tolerance

    def _calc_ichimoku(self, high: np.ndarray, low: np.ndarray, close: np.ndarray):
        """Calculate Ichimoku components."""
        # Tenkan-sen: (9-period high + 9-period low) / 2
        tenkan_high = pd.Series(high).rolling(self.tenkan_period).max().values
        tenkan_low = pd.Series(low).rolling(self.tenkan_period).min().values
        tenkan = (tenkan_high + tenkan_low) / 2.0

        # Kijun-sen: (26-period high + 26-period low) / 2
        kijun_high = pd.Series(high).rolling(self.kijun_period).max().values
        kijun_low = pd.Series(low).rolling(self.kijun_period).min().values
        kijun = (kijun_high + kijun_low) / 2.0

        # Senkou A: (Tenkan + Kijun) / 2 (current calculation)
        senkou_a = (tenkan + kijun) / 2.0

        # Senkou B: (52-period high + 52-period low) / 2
        senkou_b_high = pd.Series(high).rolling(self.senkou_b_period).max().values
        senkou_b_low = pd.Series(low).rolling(self.senkou_b_period).min().values
        senkou_b = (senkou_b_high + senkou_b_low) / 2.0

        return tenkan, kijun, senkou_a, senkou_b

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = self.senkou_b_period + 10
        if window is None or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)
        high = window["high"].values.astype(np.float64)
        low = window["low"].values.astype(np.float64)

        tenkan, kijun, senkou_a, senkou_b = self._calc_ichimoku(high, low, close)

        # Cloud boundaries (use latest values)
        cloud_top = max(senkou_a[-1], senkou_b[-1])
        cloud_bottom = min(senkou_a[-1], senkou_b[-1])

        price_curr = close[-1]
        price_prev = close[-2]
        tolerance = cloud_top * self.touch_tolerance

        # Tenkan/Kijun alignment
        tenkan_above_kijun = tenkan[-1] > kijun[-1]
        tenkan_below_kijun = tenkan[-1] < kijun[-1]

        # Long: price bounces from cloud top (was at/above cloud, dipped to cloud top)
        touched_cloud_top = (
            low[-1] <= cloud_top + tolerance
            and price_curr >= cloud_top
        )
        if touched_cloud_top and tenkan_above_kijun:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "tenkan": float(tenkan[-1]),
                    "kijun": float(kijun[-1]),
                    "cloud_top": float(cloud_top),
                    "cloud_bottom": float(cloud_bottom),
                    "price": float(price_curr),
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: price rejects from cloud bottom (was at/below cloud, rose to cloud bottom)
        touched_cloud_bottom = (
            high[-1] >= cloud_bottom - tolerance
            and price_curr <= cloud_bottom
        )
        if touched_cloud_bottom and tenkan_below_kijun:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-2],
                end_time=window.index[-1],
                confidence=0.80,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "tenkan": float(tenkan[-1]),
                    "kijun": float(kijun[-1]),
                    "cloud_top": float(cloud_top),
                    "cloud_bottom": float(cloud_bottom),
                    "price": float(price_curr),
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
