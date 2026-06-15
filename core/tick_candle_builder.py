"""
core/tick_candle_builder.py
---------------------------
Aggregate tick data into OHLCV candles.

Input: DataFrame with columns [timestamp, bid, ask, volume]
Output: Standard OHLCV DataFrame compatible with MarketDataStore

Uses midprice = (bid + ask) / 2 for candle construction.
"""

from __future__ import annotations
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Valid pandas resample offsets
_TF_MAP = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


class TickCandleBuilder:
    """
    Build OHLCV candles from raw tick data.

    Usage::

        builder = TickCandleBuilder()
        m1 = builder.build_m1(ticks_df)
        m5 = builder.resample(m1, "M5")
    """

    @staticmethod
    def build_m1(
        ticks: pd.DataFrame,
        timestamp_col: str = "timestamp",
        bid_col: str = "bid",
        ask_col: str = "ask",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """
        Build M1 (1-minute) OHLCV candles from raw ticks.

        Parameters
        ----------
        ticks : DataFrame
            Must contain timestamp, bid, ask, volume columns.
        timestamp_col, bid_col, ask_col, volume_col : str
            Column names.

        Returns
        -------
        DataFrame with columns [open, high, low, close, volume]
        and DatetimeIndex (1-minute bars).
        """
        df = ticks.copy()

        # Parse timestamps
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        df = df.set_index(timestamp_col).sort_index()

        # Midprice
        df["mid"] = (df[bid_col].astype(float) + df[ask_col].astype(float)) / 2.0

        # Group into 1-minute windows
        ohlcv = df["mid"].resample("1min").ohlc()
        vol = df[volume_col].resample("1min").sum()

        # Combine
        result = pd.DataFrame({
            "open":   ohlcv["open"],
            "high":   ohlcv["high"],
            "low":    ohlcv["low"],
            "close":  ohlcv["close"],
            "volume": vol,
        })

        # Drop empty candles (no ticks in that minute)
        result = result.dropna(subset=["open"])

        logger.info(
            "Built %d M1 candles from %d ticks (%s → %s)",
            len(result), len(df), result.index[0], result.index[-1],
        )
        return result

    @staticmethod
    def build_m1_with_ticks(
        ticks: pd.DataFrame,
        timestamp_col: str = "timestamp",
        bid_col: str = "bid",
        ask_col: str = "ask",
        volume_col: str = "volume",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build M1 candles AND return the cleaned tick DataFrame.

        Returns
        -------
        (m1_df, ticks_df)
            m1_df   : standard OHLCV M1 bars (same as build_m1)
            ticks_df: cleaned ticks with DatetimeIndex, columns [bid, ask, volume]
        """
        df = ticks.copy()

        # Parse timestamps
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        df = df.set_index(timestamp_col).sort_index()

        # Midprice
        df["mid"] = (df[bid_col].astype(float) + df[ask_col].astype(float)) / 2.0

        # Group into 1-minute windows
        ohlcv = df["mid"].resample("1min").ohlc()
        vol = df[volume_col].resample("1min").sum()

        # Combine
        m1 = pd.DataFrame({
            "open":   ohlcv["open"],
            "high":   ohlcv["high"],
            "low":    ohlcv["low"],
            "close":  ohlcv["close"],
            "volume": vol,
        })
        m1 = m1.dropna(subset=["open"])

        # Prepare ticks: keep bid, ask, volume with clean DatetimeIndex
        tick_out = df[[bid_col, ask_col, volume_col]].copy()
        tick_out.columns = ["bid", "ask", "volume"]

        logger.info(
            "Built %d M1 candles from %d ticks (%s → %s)",
            len(m1), len(tick_out), m1.index[0], m1.index[-1],
        )
        return m1, tick_out

    @staticmethod
    def resample(
        m1: pd.DataFrame,
        timeframe: str = "M5",
    ) -> pd.DataFrame:
        """
        Resample M1 candles into higher timeframe.

        Parameters
        ----------
        m1 : DataFrame
            M1 OHLCV candles (output of build_m1).
        timeframe : str
            Target timeframe: "M5", "M15", "H1", "H4", "D1".

        Returns
        -------
        Resampled OHLCV DataFrame.
        """
        offset = _TF_MAP.get(timeframe.upper())
        if offset is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(_TF_MAP.keys())}")

        ohlcv = m1[["open", "high", "low", "close"]].resample(offset).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        vol = m1["volume"].resample(offset).sum()

        result = pd.DataFrame({
            "open":   ohlcv["open"],
            "high":   ohlcv["high"],
            "low":    ohlcv["low"],
            "close":  ohlcv["close"],
            "volume": vol,
        }).dropna(subset=["open"])

        logger.info("Resampled M1 → %s: %d candles", timeframe, len(result))
        return result

    @classmethod
    def build_all_timeframes(
        cls,
        ticks: pd.DataFrame,
        timestamp_col: str = "timestamp",
        bid_col: str = "bid",
        ask_col: str = "ask",
        volume_col: str = "volume",
    ) -> dict[str, pd.DataFrame]:
        """
        Build M1, M5, H1 candles from ticks in one call.

        Returns dict with keys "M1", "M5", "H1".
        """
        m1 = cls.build_m1(ticks, timestamp_col, bid_col, ask_col, volume_col)
        return {
            "M1": m1,
            "M5": cls.resample(m1, "M5"),
            "H1": cls.resample(m1, "H1"),
        }
