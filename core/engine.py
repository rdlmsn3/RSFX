"""
core/engine.py
--------------
Shared data primitives for the RSFX platform.

CandleArrays is the canonical container for OHLCV NumPy arrays
used by strategies, the backtester, and the live engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandleArrays:
    """
    Full-series NumPy arrays extracted once from the DataFrame.

    Strategies receive these arrays + the current index i.
    Indexing a NumPy array is ~100x faster than pandas .iloc[i]['col'].
    """
    timestamps: np.ndarray   # dtype=datetime64[ns]
    opens:      np.ndarray
    highs:      np.ndarray
    lows:       np.ndarray
    closes:     np.ndarray
    volumes:    np.ndarray
    n:          int          # total length

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CandleArrays":
        return cls(
            timestamps = df.index.values,
            opens      = df["open"].values,
            highs      = df["high"].values,
            lows       = df["low"].values,
            closes     = df["close"].values,
            volumes    = df["volume"].values if "volume" in df.columns else np.zeros(len(df)),
            n          = len(df),
        )
