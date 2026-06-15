"""
detectors/strategies/volume_profile_ema.py
-------------------------------------------
Price bounce from high-volume node + EMA trend strategy (M5 only).

Calculates a simple volume profile by binning recent price range and
identifying high-volume nodes. A "bounce" occurs when price touches a
high-volume node area and reverses.

Rules:
  LONG:  Price near high-volume node + bounces + EMA9 > EMA21
  SHORT: Price near high-volume node + bounces + EMA9 < EMA21
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
    logger.warning("TA-Lib not installed. Volume Profile EMA signals disabled.")


class VolumeProfileEMAStrategy(BaseStrategy):
    """Volume profile high-volume-node bounce + EMA trend filter (M5)."""

    name = "volume_profile_ema"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        profile_lookback: int = 100,
        num_bins: int = 20,
        node_pct_threshold: float = 0.15,
        bounce_tolerance_pct: float = 0.3,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.profile_lookback = profile_lookback
        self.num_bins = num_bins
        self.node_pct_threshold = node_pct_threshold
        self.bounce_tolerance_pct = bounce_tolerance_pct

    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        detected: list[PatternSignal] = []
        window = windows.get("M5")
        min_bars = max(self.ema_slow, self.profile_lookback) + 3
        if window is None or not TA_AVAILABLE or len(window) < min_bars:
            return detected

        close = window["close"].values.astype(np.float64)

        ema_fast = talib.EMA(close, timeperiod=self.ema_fast)
        ema_slow = talib.EMA(close, timeperiod=self.ema_slow)

        # Build volume profile from lookback window
        lookback_start = max(0, len(window) - self.profile_lookback)
        lb_close = window["close"].values[lookback_start:].astype(np.float64)
        lb_volume = window["volume"].values[lookback_start:].astype(np.float64)

        price_min = lb_close.min()
        price_max = lb_close.max()
        if price_max == price_min:
            return detected

        # Bin prices and accumulate volume
        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_indices = np.digitize(lb_close, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        vol_profile = np.zeros(self.num_bins)
        for i, bi in enumerate(bin_indices):
            vol_profile[bi] += lb_volume[i]

        total_volume = vol_profile.sum()
        if total_volume == 0:
            return detected

        # Normalise and find high-volume nodes (bins with > threshold of total volume)
        vol_pct = vol_profile / total_volume
        hvn_indices = np.where(vol_pct > self.node_pct_threshold)[0]
        if len(hvn_indices) == 0:
            return detected

        # Find the high-volume node closest to current price
        hvn_prices = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in hvn_indices]
        price_now = close[-1]
        closest_hvn = min(hvn_prices, key=lambda p: abs(p - price_now))

        tolerance = price_now * (self.bounce_tolerance_pct / 100.0)
        near_hvn = abs(price_now - closest_hvn) <= tolerance

        if not near_hvn:
            return detected

        # Check bounce: price touched near HVN and is reversing
        # Look at last 3 bars: low was near HVN, current close is moving away
        recent_lows = window["low"].values[-3:].astype(np.float64)
        recent_highs = window["high"].values[-3:].astype(np.float64)

        # For long: recent low touched HVN area, now closing above it
        touched_from_below = any(
            abs(low - closest_hvn) <= tolerance or low <= closest_hvn
            for low in recent_lows[:-1]
        )
        closing_above = price_now > closest_hvn

        # For short: recent high touched HVN area, now closing below it
        touched_from_above = any(
            abs(high - closest_hvn) <= tolerance or high >= closest_hvn
            for high in recent_highs[:-1]
        )
        closing_below = price_now < closest_hvn

        ema_fast_now = ema_fast[-1]
        ema_slow_now = ema_slow[-1]

        # Long: bounce from HVN + EMA9 > EMA21
        if touched_from_below and closing_above and ema_fast_now > ema_slow_now:
            detected.append(PatternSignal(
                name=f"{self.name}_LONG",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "LONG",
                    "hvn_price": round(closest_hvn, 5),
                    "ema9": round(ema_fast_now, 5),
                    "ema21": round(ema_slow_now, 5),
                    "price": price_now,
                },
            ))
            logger.info("LONG signal at %s (strategy=%s)", current_timestamp, self.name)

        # Short: bounce from HVN + EMA9 < EMA21
        elif touched_from_above and closing_below and ema_fast_now < ema_slow_now:
            detected.append(PatternSignal(
                name=f"{self.name}_SHORT",
                start_time=window.index[-3],
                end_time=window.index[-1],
                confidence=0.78,
                metadata={
                    "strategy": self.name,
                    "direction": "SHORT",
                    "hvn_price": round(closest_hvn, 5),
                    "ema9": round(ema_fast_now, 5),
                    "ema21": round(ema_slow_now, 5),
                    "price": price_now,
                },
            ))
            logger.info("SHORT signal at %s (strategy=%s)", current_timestamp, self.name)

        return detected
