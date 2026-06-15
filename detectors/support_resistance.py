"""
support_resistance.py
---------------------
Support/Resistance detection via pivot point clustering.

Methods:
  1. Pivot detection — find swing highs/lows (local extrema)
  2. Clustering — merge nearby pivots into S/R zones
  3. Strength scoring — touches, recency, proximity, cleanliness

Usage:
    sr = SupportResistance(df)
    levels = sr.find_levels()

    # Query nearest levels
    support = sr.nearest_support(150.250)
    resistance = sr.nearest_resistance(150.250)

    # Get TP/SL from S/R
    tp, sl = sr.get_tp_sl(150.250, "LONG", atr_sl=0.30)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SRLevel:
    """A single support or resistance level."""
    price: float
    level_type: str    # "support" or "resistance"
    strength: float    # 0-1 score (higher = stronger)
    touches: int       # how many pivots clustered here
    recency: int       # candles since last touch (0 = current)
    zone_low: float    # lower bound of the zone
    zone_high: float   # upper bound of the zone

    @property
    def zone_mid(self) -> float:
        return (self.zone_low + self.zone_high) / 2


class SupportResistance:
    """
    Pivot-based S/R detection with clustering.

    Algorithm:
      1. Find all swing highs/lows using a rolling window
      2. Cluster nearby pivots (within pip_tolerance)
      3. Score each cluster by touch count, recency, and cleanliness
      4. Classify as support (below price) or resistance (above price)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pivot_lookback: int = 5,
        pip_tolerance: float = 0.015,
        min_touches: int = 3,
    ):
        """
        Parameters
        ----------
        df : DataFrame with OHLC columns
        pivot_lookback : candles each side to confirm a swing point
        pip_tolerance : max distance (in price units) to merge pivots
        min_touches : minimum touches to keep a level
        """
        self._df = df
        self._pivot_lookback = pivot_lookback
        self._pip_tolerance = pip_tolerance
        self._min_touches = min_touches

        self._closes = df["close"].values
        self._highs = df["high"].values
        self._lows = df["low"].values
        self._n = len(df)

    def _find_pivots(self) -> list[tuple[float, str, int]]:
        """
        Find swing highs and lows.

        Returns list of (price, type, candle_index).
        """
        pivots = []
        lb = self._pivot_lookback

        for i in range(lb, self._n - lb):
            # Swing high: high is the max in the window
            window_highs = self._highs[i - lb: i + lb + 1]
            if self._highs[i] == window_highs.max():
                pivots.append((float(self._highs[i]), "high", i))

            # Swing low: low is the min in the window
            window_lows = self._lows[i - lb: i + lb + 1]
            if self._lows[i] == window_lows.min():
                pivots.append((float(self._lows[i]), "low", i))

        return pivots

    def _cluster_pivots(
        self, pivots: list[tuple[float, str, int]]
    ) -> list[dict]:
        """
        Cluster nearby pivots into S/R zones.

        Returns list of dicts with aggregated info.
        """
        if not pivots:
            return []

        # Sort by price
        pivots.sort(key=lambda x: x[0])

        clusters = []
        current_cluster = [pivots[0]]

        for p in pivots[1:]:
            # Check if close enough to merge
            if abs(p[0] - current_cluster[-1][0]) <= self._pip_tolerance:
                current_cluster.append(p)
            else:
                clusters.append(current_cluster)
                current_cluster = [p]
        clusters.append(current_cluster)

        # Build cluster info
        result = []
        for cluster in clusters:
            prices = [p[0] for p in cluster]
            avg_price = sum(prices) / len(prices)
            touches = len(cluster)
            indices = [p[2] for p in cluster]
            recency = self._n - 1 - max(indices)  # candles since last touch

            # Determine if support or resistance
            # (will be classified relative to current price later)
            highs = sum(1 for p in cluster if p[1] == "high")
            lows = sum(1 for p in cluster if p[1] == "low")

            # Zone bounds
            zone_low = min(prices) - self._pip_tolerance * 0.2
            zone_high = max(prices) + self._pip_tolerance * 0.2

            result.append({
                "price": avg_price,
                "touches": touches,
                "recency": recency,
                "indices": indices,
                "zone_low": zone_low,
                "zone_high": zone_high,
                "highs": highs,
                "lows": lows,
            })

        return result

    def _score_level(
        self, cluster: dict, current_price: float
    ) -> float:
        """
        Score a cluster's strength (0-1).

        Factors:
          - Touch count (more = stronger)
          - Recency (recent = stronger)
          - Proximity to price (closer = more relevant)
          - Cleanliness (all bounces vs mix of high/low)
        """
        # Touch score: logarithmic scaling, max around 5-6 touches
        touch_score = min(1.0, np.log1p(cluster["touches"]) / np.log1p(6))

        # Recency score: exponential decay over 500 candles
        recency_score = np.exp(-cluster["recency"] / 500)

        # Proximity score: closer to current price = more relevant
        distance = abs(cluster["price"] - current_price)
        proximity_score = np.exp(-distance / (current_price * 0.01))  # 1% decay

        # Cleanliness: ratio of dominant type (all bounces = clean)
        total = cluster["touches"]
        dominant = max(cluster["highs"], cluster["lows"])
        cleanliness = dominant / total if total > 0 else 0.5

        # Weighted combination
        score = (
            0.35 * touch_score +
            0.25 * recency_score +
            0.25 * proximity_score +
            0.15 * cleanliness
        )

        return round(min(1.0, max(0.0, score)), 3)

    def find_levels(
        self,
        current_price: Optional[float] = None,
    ) -> list[SRLevel]:
        """
        Find all S/R levels.

        Returns sorted list of SRLevel (strongest first).
        """
        if current_price is None:
            current_price = float(self._closes[-1])

        pivots = self._find_pivots()
        clusters = self._cluster_pivots(pivots)

        levels = []
        for cluster in clusters:
            if cluster["touches"] < self._min_touches:
                continue

            # Classify as support or resistance
            if cluster["price"] < current_price:
                level_type = "support"
            elif cluster["price"] > current_price:
                level_type = "resistance"
            else:
                # At current price — classify by dominant type
                level_type = "support" if cluster["lows"] >= cluster["highs"] else "resistance"

            strength = self._score_level(cluster, current_price)

            levels.append(SRLevel(
                price=cluster["price"],
                level_type=level_type,
                strength=strength,
                touches=cluster["touches"],
                recency=cluster["recency"],
                zone_low=cluster["zone_low"],
                zone_high=cluster["zone_high"],
            ))

        # Sort by strength (strongest first)
        levels.sort(key=lambda x: x.strength, reverse=True)

        return levels

    def nearest_support(self, price: float) -> Optional[SRLevel]:
        """Find the nearest support level below the given price."""
        levels = self.find_levels(price)
        supports = [l for l in levels if l.level_type == "support" and l.price < price]
        if not supports:
            return None
        # Nearest = closest price
        supports.sort(key=lambda x: price - x.price)
        return supports[0]

    def nearest_resistance(self, price: float) -> Optional[SRLevel]:
        """Find the nearest resistance level above the given price."""
        levels = self.find_levels(price)
        resistances = [l for l in levels if l.level_type == "resistance" and l.price > price]
        if not resistances:
            return None
        # Nearest = closest price
        resistances.sort(key=lambda x: x.price - price)
        return resistances[0]

    def get_tp_sl(
        self,
        entry_price: float,
        direction: str,
        atr_sl: float,
        max_tp_ratio: float = 3.0,
        min_sl_pips: float = 0.05,
    ) -> tuple[float, float]:
        """
        Compute TP/SL using S/R levels with ATR fallback.

        Parameters
        ----------
        entry_price : current price
        direction : "LONG" or "SHORT"
        atr_sl : ATR-based stop loss distance (used as fallback or base SL)
        max_tp_ratio : max TP/SL ratio (risk management cap)
        min_sl_pips : minimum SL distance

        Returns
        -------
        (take_profit, stop_loss)
        """
        if direction == "LONG":
            resistance = self.nearest_resistance(entry_price)
            support = self.nearest_support(entry_price)

            # SL: use support level if close, otherwise ATR
            if support and (entry_price - support.zone_mid) <= atr_sl * 1.5:
                sl = support.zone_mid - self._pip_tolerance * 0.5
            else:
                sl = entry_price - atr_sl

            # TP: target next resistance
            if resistance:
                tp = resistance.zone_mid
            else:
                tp = entry_price + atr_sl * 2.0  # fallback: 2x ATR

        else:  # SHORT
            resistance = self.nearest_resistance(entry_price)
            support = self.nearest_support(entry_price)

            # SL: use resistance level if close, otherwise ATR
            if resistance and (resistance.zone_mid - entry_price) <= atr_sl * 1.5:
                sl = resistance.zone_mid + self._pip_tolerance * 0.5
            else:
                sl = entry_price + atr_sl

            # TP: target next support
            if support:
                tp = support.zone_mid
            else:
                tp = entry_price - atr_sl * 2.0  # fallback: 2x ATR

        # Sanity: enforce minimum SL
        risk = abs(entry_price - sl)
        if risk < min_sl_pips:
            if direction == "LONG":
                sl = entry_price - min_sl_pips
            else:
                sl = entry_price + min_sl_pips

        # Sanity: cap TP/SL ratio
        reward = abs(tp - entry_price)
        risk = abs(entry_price - sl)
        if risk > 0 and reward / risk > max_tp_ratio:
            if direction == "LONG":
                tp = entry_price + risk * max_tp_ratio
            else:
                tp = entry_price - risk * max_tp_ratio

        return tp, sl
