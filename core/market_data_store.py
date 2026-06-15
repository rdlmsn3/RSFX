"""
core/market_data_store.py
-------------------------
Centralised, multi-symbol, multi-timeframe market data store.

Responsibilities
----------------
- Hold raw M1 data for one or more symbols
- Pre-compute higher timeframes (M5, H1, D1) at load time
- Provide efficient windowed lookups without copying full frames
- Never re-resample during playback

Design for scale
----------------
- Pre-computation is O(n) once; lookups are O(log n) via .searchsorted()
- get_window() avoids slicing large DataFrames when only the tail is needed

Future extensions (add symbols / timeframes without touching other modules)
---------------------------------------------------------------------------
    store.load_symbol("DXY",  dxy_df)
    store.load_symbol("XAUUSD", gold_df)
    store.load_symbol("US10Y", bond_df)
"""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Supported timeframes and their Pandas resample rules
TIMEFRAME_RULES: dict[str, str] = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
    "W1":  "1W",
}


class MarketDataStore:
    """
    Multi-symbol, multi-timeframe OHLCV store.

    Attributes
    ----------
    symbols : dict[str, dict[str, pd.DataFrame]]
        Top-level key  : symbol name ("EURUSD", "DXY", …)
        Second-level   : timeframe label ("M1", "M5", "H1", "D1")
        Value          : pre-computed OHLCV DataFrame with DatetimeIndex

    Usage
    -----
    store = MarketDataStore()
    store.load_symbol("EURUSD", m1_dataframe)
    df = store.get_data("EURUSD", "H1")
    window = store.get_window("EURUSD", "M5", current_ts, lookback=200)
    """

    # Timeframes to pre-compute from M1 (extend as needed)
    PRECOMPUTED_TIMEFRAMES: list[str] = ["M1", "M5", "H1", "D1"]

    def __init__(self) -> None:
        # { symbol: { timeframe: DataFrame } }
        self.symbols: dict[str, dict[str, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_symbol(
        self,
        symbol: str,
        m1_data: pd.DataFrame,
        extra_timeframes: Optional[list[str]] = None,
    ) -> None:
        """
        Ingest M1 data for *symbol* and pre-compute higher timeframes.

        Parameters
        ----------
        symbol : str
            Instrument identifier, e.g. "EURUSD".
        m1_data : pd.DataFrame
            Clean M1 OHLCV frame with DatetimeIndex (from DataAdapter.load()).
        extra_timeframes : list[str], optional
            Additional timeframes beyond the defaults to pre-compute.
            Must be keys in TIMEFRAME_RULES.
        """
        if m1_data.empty:
            raise ValueError(f"Cannot load empty DataFrame for symbol '{symbol}'.")

        timeframes_to_build = list(self.PRECOMPUTED_TIMEFRAMES)
        if extra_timeframes:
            for tf in extra_timeframes:
                if tf not in TIMEFRAME_RULES:
                    raise ValueError(f"Unknown timeframe '{tf}'. Valid: {list(TIMEFRAME_RULES)}")
                if tf not in timeframes_to_build:
                    timeframes_to_build.append(tf)

        logger.info(
            "Loading %s: %d M1 candles (%s → %s)",
            symbol, len(m1_data), m1_data.index[0], m1_data.index[-1],
        )

        self.symbols[symbol] = {}
        self.symbols[symbol]["M1"] = m1_data.copy()

        for tf in timeframes_to_build:
            if tf == "M1":
                continue
            self.symbols[symbol][tf] = self._resample(m1_data, tf)
            logger.info("  Pre-computed %s %s: %d candles", symbol, tf, len(self.symbols[symbol][tf]))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Return the full pre-computed DataFrame for *symbol* / *timeframe*.

        Returns
        -------
        pd.DataFrame
            Read-only reference (do NOT mutate the returned frame).
        """
        self._assert_available(symbol, timeframe)
        return self.symbols[symbol][timeframe]

    def get_window(
        self,
        symbol: str,
        timeframe: str,
        current_timestamp: pd.Timestamp,
        lookback: int = 500,
    ) -> pd.DataFrame:
        """
        Return the most recent *lookback* candles up to and including
        *current_timestamp* for *symbol* / *timeframe*.

        This is the primary method used during playback to avoid repeatedly
        slicing large DataFrames from the beginning.

        Parameters
        ----------
        symbol : str
        timeframe : str
        current_timestamp : pd.Timestamp
            The playback cursor.  Only candles ≤ this value are returned.
        lookback : int
            Maximum number of candles to return (default 500).

        Returns
        -------
        pd.DataFrame
            Slice of the store; index is a DatetimeIndex.
        """
        self._assert_available(symbol, timeframe)
        df = self.symbols[symbol][timeframe]

        # Find position of current_timestamp using binary search – O(log n)
        pos = df.index.searchsorted(current_timestamp, side="right")
        start = max(0, pos - lookback)
        return df.iloc[start:pos]

    def get_timestamp_at_index(self, symbol: str, timeframe: str, index: int) -> pd.Timestamp:
        """Return the timestamp at a given integer position."""
        self._assert_available(symbol, timeframe)
        df = self.symbols[symbol][timeframe]
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} out of range for {symbol}/{timeframe} (len={len(df)})")
        return df.index[index]

    def length(self, symbol: str, timeframe: str = "M1") -> int:
        """Return the total number of candles for *symbol* / *timeframe*."""
        self._assert_available(symbol, timeframe)
        return len(self.symbols[symbol][timeframe])

    def available_symbols(self) -> list[str]:
        return list(self.symbols.keys())

    def available_timeframes(self, symbol: str) -> list[str]:
        if symbol not in self.symbols:
            return []
        return list(self.symbols[symbol].keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(m1_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample M1 data to a higher timeframe using standard OHLCV logic."""
        rule = TIMEFRAME_RULES[timeframe]
        agg: dict[str, str] = {
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last",
        }
        if "volume" in m1_df.columns:
            agg["volume"] = "sum"

        resampled = m1_df.resample(rule).agg(agg).dropna(subset=["open", "close"])
        return resampled

    def _assert_available(self, symbol: str, timeframe: str) -> None:
        if symbol not in self.symbols:
            raise KeyError(
                f"Symbol '{symbol}' not loaded. "
                f"Available: {self.available_symbols()}"
            )
        if timeframe not in self.symbols[symbol]:
            raise KeyError(
                f"Timeframe '{timeframe}' not pre-computed for '{symbol}'. "
                f"Available: {self.available_timeframes(symbol)}"
            )