"""
detectors/strategies/base.py
-----------------------------
Base class for all trading strategies.

Any strategy must implement evaluate() returning a list of PatternSignal.
The signal's metadata MUST include "direction" (LONG or SHORT).

Multi-timeframe support:
  windows dict keys are timeframe labels ("M1", "M5", "H1", "D1").
  Strategies use whichever TFs they need — M1-only strategies just
  ignore extra keys.

TP/SL:
  Use compute_tp_sl() to add entry/take_profit/stop_loss to signal metadata.
  ATR-based by default. Override _compute_sl()/_compute_tp() for custom logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from detectors.signal import PatternSignal

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class BaseStrategy(ABC):
    """Abstract base for pluggable strategies."""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        """
        Evaluate strategy across multiple timeframes.

        Parameters
        ----------
        windows : dict[str, pd.DataFrame]
            Keys: timeframe labels ("M1", "M5", "H1", …).
            Values: OHLCV DataFrames up to current_timestamp.
        current_timestamp : pd.Timestamp
            Current playback cursor timestamp.

        Returns
        -------
        list[PatternSignal]
            0 or more signals. metadata MUST include "direction" key
            ("LONG" or "SHORT").
        """
        ...

    # ------------------------------------------------------------------
    # TP/SL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tp_sl(
        signal: PatternSignal,
        window: pd.DataFrame,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.0,
    ) -> PatternSignal:
        """
        Add entry_price, stop_loss, take_profit to signal metadata.

        ATR-based calculation:
          SL = entry ± ATR * sl_atr_mult
          TP = entry ± ATR * tp_atr_mult

        For LONG:  SL below entry, TP above entry
        For SHORT: SL above entry, TP below entry

        Returns the same signal object (mutated in place).
        """
        direction = signal.metadata.get("direction", "")
        if direction not in ("LONG", "SHORT"):
            return signal

        # Get entry price (close of trigger candle)
        entry_price = float(window["close"].iloc[-1])

        # Compute ATR
        if HAS_TALIB and len(window) >= atr_period + 1:
            high = window["high"].values.astype(np.float64)
            low = window["low"].values.astype(np.float64)
            close = window["close"].values.astype(np.float64)
            atr = talib.ATR(high, low, close, timeperiod=atr_period)
            atr_value = float(atr[-1]) if not np.isnan(atr[-1]) else float(atr[~np.isnan(atr)][-1]) if any(~np.isnan(atr)) else 0.001
        else:
            # Fallback: average range of last 14 candles
            ranges = window["high"].iloc[-atr_period:] - window["low"].iloc[-atr_period:]
            atr_value = float(ranges.mean()) if len(ranges) > 0 else 0.001

        # Calculate SL and TP
        sl_distance = atr_value * sl_atr_mult
        tp_distance = atr_value * tp_atr_mult

        if direction == "LONG":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        signal.metadata["entry_price"] = round(entry_price, 5)
        signal.metadata["stop_loss"] = round(stop_loss, 5)
        signal.metadata["take_profit"] = round(take_profit, 5)
        signal.metadata["atr"] = round(atr_value, 5)
        signal.metadata["sl_distance"] = round(sl_distance, 5)
        signal.metadata["tp_distance"] = round(tp_distance, 5)

        return signal

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
