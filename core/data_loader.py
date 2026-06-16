"""
core/data_loader.py
-------------------
Flexible, adapter-pattern data loading system.

The DataAdapter abstract base class defines the contract.
Concrete adapters handle provider-specific formats without
touching the rest of the system.

Existing adapters
-----------------
    HistDataAdapter   – HistData.com  (Date,Time,Open,High,Low,Close,Volume)
    ParquetAdapter    – Parquet files (auto-detects bar vs tick columns)

Future adapters (add here, zero changes elsewhere)
---------------------------------------------------
    MT5Adapter        – MetaTrader 5 export
    DukascopyAdapter  – Dukascopy tick/bar CSV
    CustomAdapter     – User-defined column mapping
    DatabaseAdapter   – SQL / TimescaleDB source
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that every adapter must deliver after normalisation
REQUIRED_COLUMNS = {"open", "high", "low", "close"}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataAdapter(ABC):
    """
    Contract for all data source adapters.

    Subclass this and implement *load()* to add a new data source.
    The rest of the system only ever calls *load()*.
    """

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """
        Load and normalise OHLC data from *path*.

        Returns
        -------
        pd.DataFrame
            Index  : pd.DatetimeIndex named "timestamp", UTC-naive, sorted ASC
            Columns: open, high, low, close, volume (volume optional)
            Dtypes : float64 for all price/volume columns
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HistData.com adapter
# ---------------------------------------------------------------------------

class HistDataAdapter(DataAdapter):
    """
    Adapter for HistData.com ASCII bar data.

    Expected raw format (no header row in the file itself – HistData ships
    files both with and without headers; we handle both):

        20030505 000100,1.1234,1.1240,1.1230,1.1238,0

    OR with a header:

        Date,Time,Open,High,Low,Close,Volume
        20030505,000100,1.1234,1.1240,1.1230,1.1238,0
    """

    # HistData uses separate Date + Time columns; combined format also seen
    _COLUMN_MAP = {
        "date":   "date",
        "time":   "time",
        "open":   "open",
        "high":   "high",
        "low":    "low",
        "close":  "close",
        "volume": "volume",
        "vol":    "volume",
    }

    def __init__(self) -> None:
        self.raw_ticks: pd.DataFrame | None = None

    def load(self, path: str) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info("Loading HistData file: %s", path)

        # ---- Detect delimiter (comma or semicolon) ----
        with open(file_path, 'r') as f:
            first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
        logger.info(f"Detected delimiter: '{delimiter}'")

        # ---- Peek at first line to detect header presence ----
        raw = pd.read_csv(
            file_path,
            header=None,
            nrows=1,
            dtype=str,
            sep=delimiter,          # use detected delimiter
        )
        first_cell = str(raw.iloc[0, 0]).strip().lower()
        has_header = not first_cell[:4].isdigit()

        # ---- Load full file with the detected delimiter ----
        df = pd.read_csv(
            file_path,
            header=0 if has_header else None,
            dtype=str,
            low_memory=False,
            sep=delimiter,          # use same delimiter
        )

        # ---- Normalise columns ----
        is_tick_format = False
        if not has_header:
            col_count = len(df.columns)
            # Detect combined datetime column (space in first value)
            first_val = str(df.iloc[0, 0])
            if ' ' in first_val and first_val[:4].isdigit():
                if col_count == 4:
                    # Tick data: datetime, bid, ask, volume
                    df.columns = ["datetime", "bid", "ask", "volume"]
                    is_tick_format = True
                else:
                    # Bar data: datetime, open, high, low, close, volume
                    expected_cols = ["datetime", "open", "high", "low", "close", "volume"]
                    df.columns = expected_cols[:col_count]
            else:
                # Separate date + time
                positional = ["date", "time", "open", "high", "low", "close", "volume"]
                df.columns = positional[:col_count]
        else:
            df.columns = [c.strip().lower() for c in df.columns]
            df.rename(columns=self._COLUMN_MAP, inplace=True)

        # ---- Tick data: route through TickCandleBuilder --------------------
        if is_tick_format:
            return self._load_tick_data(df)

        # ---- Build timestamp index ----
        df = self._build_timestamp_index(df)

        # ---- Validate required columns ------------------------------------
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"HistData file is missing required columns after parsing: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # ---- Cast to numeric ----------------------------------------------
        price_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            price_cols.append("volume")

        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---- Drop rows with NaN OHLC values --------------------------------
        before = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with NaN OHLC values.", dropped)

        # ---- Sort and deduplicate -----------------------------------------
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            "Loaded %d candles from %s to %s",
            len(df),
            df.index[0],
            df.index[-1],
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine Date + Time columns (or a single datetime column) into a
        DatetimeIndex named 'timestamp'.
        """
        if "date" in df.columns and "time" in df.columns:
            # Pad time to 6 digits: "10000" → "010000"
            df["time"] = df["time"].str.strip()
            df["date"] = df["date"].str.strip()
            combined = df["date"] + " " + df["time"]
            # Try two common HistData datetime formats
            for fmt in ("%Y%m%d %H%M%S", "%Y%m%d %H%M%S%f",
                        "%Y.%m.%d %H%M%S", "%m/%d/%Y %H%M%S"):
                try:
                    ts = pd.to_datetime(combined, format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
            else:
                ts = pd.to_datetime(combined, infer_datetime_format=True)

            df = df.drop(columns=["date", "time"])
        elif "datetime" in df.columns:
            for fmt in ("%Y%m%d %H%M%S", "%Y%m%d %H%M%S%f",
                        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    ts = pd.to_datetime(df["datetime"], format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
            else:
                ts = pd.to_datetime(df["datetime"], infer_datetime_format=True)
            df = df.drop(columns=["datetime"])
        else:
            raise ValueError(
                "Cannot find date/time columns. Expected 'date'+'time' or 'datetime'."
            )

        df.index = ts
        df.index.name = "timestamp"
        return df

    # ------------------------------------------------------------------
    # Tick data handling (HistData.com tick files: bid/ask/volume)
    # ------------------------------------------------------------------

    def _load_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a raw tick DataFrame (datetime, bid, ask, volume) into
        M1 OHLCV candles via TickCandleBuilder.

        The input df has columns: datetime, bid, ask, volume
        with the datetime column still as a string (not yet parsed).
        """
        from core.tick_candle_builder import TickCandleBuilder

        # Parse timestamps
        for fmt in ("%Y%m%d %H%M%S%f", "%Y%m%d %H%M%S",
                     "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                df["timestamp"] = pd.to_datetime(df["datetime"], format=fmt)
                break
            except (ValueError, TypeError):
                continue
        else:
            df["timestamp"] = pd.to_datetime(df["datetime"], infer_datetime_format=True)

        df = df.drop(columns=["datetime"])

        # Cast to numeric
        for col in ["bid", "ask", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        before = len(df)
        df.dropna(subset=["bid", "ask"], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with NaN tick prices.", dropped)

        logger.info(
            "Loaded %d ticks from %s to %s",
            len(df), df["timestamp"].min(), df["timestamp"].max(),
        )

        # Build M1 candles from ticks (keep raw ticks for tick-level backtest)
        builder = TickCandleBuilder()
        m1, tick_out = builder.build_m1_with_ticks(
            df,
            timestamp_col="timestamp",
            bid_col="bid",
            ask_col="ask",
            volume_col="volume",
        )

        logger.info("Converted ticks → %d M1 candles", len(m1))

        # Store raw ticks on adapter for tick-level backtest
        self.raw_ticks = tick_out

        return m1


# ---------------------------------------------------------------------------
# Parquet adapter
# ---------------------------------------------------------------------------

class ParquetAdapter(DataAdapter):
    """
    Adapter for Parquet files.

    Auto-detects tick data (bid/ask columns) vs bar data (OHLC columns)
    and routes accordingly — same behaviour as HistDataAdapter but reads
    Parquet natively via pd.read_parquet().

    Column mapping is automatic for common names.  Pass ``column_map``
    or ``datetime_col`` to override for non-standard schemas.
    """

    def __init__(
        self,
        datetime_col: str | None = None,
        column_map: dict[str, str] | None = None,
    ) -> None:
        self.datetime_col = datetime_col
        self.column_map = column_map or {}
        self.raw_ticks: pd.DataFrame | None = None

    def load(self, path: str) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info("Loading Parquet file: %s", path)
        df = pd.read_parquet(file_path)

        # ---- Normalise column names ------------------------------------
        df.columns = [c.strip().lower() for c in df.columns]

        # Apply user overrides first
        if self.column_map:
            df.rename(columns=self.column_map, inplace=True)

        # Auto-detect tick data (bid/ask present) vs bar data (OHLC)
        is_tick = "bid" in df.columns and "ask" in df.columns

        if is_tick:
            return self._load_tick_data(df)

        # ---- Bar data: build DatetimeIndex ----------------------------
        df = self._build_timestamp_index(df)

        # ---- Validate required columns --------------------------------
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet file is missing required columns after parsing: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # ---- Cast to numeric ------------------------------------------
        price_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            price_cols.append("volume")

        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---- Drop rows with NaN OHLC values ---------------------------
        before = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with NaN OHLC values.", dropped)

        # ---- Sort and deduplicate -------------------------------------
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            "Loaded %d candles from %s to %s",
            len(df),
            df.index[0],
            df.index[-1],
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_timestamp_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build DatetimeIndex — handles datetime64 (native parquet) or string columns."""
        # If a specific datetime column was requested
        if self.datetime_col and self.datetime_col in df.columns:
            ts = pd.to_datetime(df[self.datetime_col])
            df = df.drop(columns=[self.datetime_col])
            df.index = ts
            df.index.name = "timestamp"
            return df

        # If index is already a DatetimeIndex (parquet often preserves types)
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = "timestamp"
            return df

        # If the index has a datetime-like name, try to convert
        if df.index.name and df.index.name.lower() in ("timestamp", "datetime", "date", "time", "index"):
            try:
                df.index = pd.to_datetime(df.index)
                df.index.name = "timestamp"
                return df
            except (ValueError, TypeError):
                pass

        # Detect epoch-millis column (e.g. timestamp_ms, ts, epoch_ms)
        for col in df.columns:
            if col in ("timestamp_ms", "ts", "epoch_ms", "time_ms"):
                if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                    sample = float(df[col].iloc[0])
                    if sample > 1e12:
                        df.index = pd.to_datetime(df[col], unit="ms")
                    elif sample > 1e9:
                        df.index = pd.to_datetime(df[col], unit="s")
                    else:
                        df.index = pd.to_datetime(df[col])
                    df = df.drop(columns=[col])
                    df.index.name = "timestamp"
                    return df

        # Fall back to column-based detection (reuse HistDataAdapter logic)
        return HistDataAdapter._build_timestamp_index(df)

    def _load_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert tick DataFrame to M1 candles via TickCandleBuilder."""
        from core.tick_candle_builder import TickCandleBuilder

        # ---- Resolve timestamp column -----------------------------------
        # Parquet files may use various names: timestamp_ms, timestamp,
        # datetime, date, time, etc.  Also handle numeric epoch millis.
        ts_col = None

        # 1. Check for common timestamp column names (case-insensitive already lowered)
        for name in ("timestamp", "datetime", "date", "time"):
            if name in df.columns:
                ts_col = name
                break

        # 2. Check for epoch-millis variants (timestamp_ms, ts, etc.)
        if ts_col is None:
            for col in df.columns:
                if col in ("timestamp_ms", "ts", "epoch_ms", "time_ms", "timeESTAMP_MS"):
                    ts_col = col
                    break

        # 3. Fallback: find first integer/float column that looks like epoch data
        if ts_col is None:
            for col in df.columns:
                if col in ("bid", "ask", "volume"):
                    continue
                if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                    # Heuristic: epoch timestamps are > 1e12 (millis) or > 1e9 (secs)
                    sample = df[col].iloc[0]
                    if sample > 1e12:
                        ts_col = col
                        break
                    elif sample > 1e9:
                        ts_col = col
                        break

        if ts_col is None:
            raise ValueError(
                f"Cannot find timestamp column for tick data. "
                f"Available columns: {list(df.columns)}"
            )

        # ---- Convert to datetime ----------------------------------------
        series = df[ts_col]
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            sample = float(series.iloc[0])
            if sample > 1e12:
                # Milliseconds epoch
                logger.info("Detected epoch millis in '%s', converting to datetime", ts_col)
                df["timestamp"] = pd.to_datetime(series, unit="ms")
            elif sample > 1e9:
                # Seconds epoch
                logger.info("Detected epoch seconds in '%s', converting to datetime", ts_col)
                df["timestamp"] = pd.to_datetime(series, unit="s")
            else:
                df["timestamp"] = pd.to_datetime(series)
        else:
            df["timestamp"] = pd.to_datetime(series)

        if ts_col != "timestamp":
            df = df.drop(columns=[ts_col])

        # Cast to numeric
        for col in ["bid", "ask", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0 if col == "volume" else None

        before = len(df)
        df.dropna(subset=["bid", "ask"], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with NaN tick prices.", dropped)

        logger.info(
            "Loaded %d ticks from %s to %s",
            len(df), df["timestamp"].min(), df["timestamp"].max(),
        )

        builder = TickCandleBuilder()
        m1, tick_out = builder.build_m1_with_ticks(
            df,
            timestamp_col="timestamp",
            bid_col="bid",
            ask_col="ask",
            volume_col="volume",
        )

        logger.info("Converted ticks → %d M1 candles", len(m1))
        self.raw_ticks = tick_out
        return m1


# ---------------------------------------------------------------------------
# Adapter factory — picks the right adapter by file extension
# ---------------------------------------------------------------------------

def get_adapter(path: str) -> DataAdapter:
    """
    Return the appropriate DataAdapter for *path* based on file extension.

    - .parquet / .pq / .parq  → ParquetAdapter
    - everything else          → HistDataAdapter (default)

    Callers that need TickDataAdapter for CSV tick files should continue
    to use it directly (e.g. when manually detecting bid/ask headers).
    """
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq", ".parq"):
        return ParquetAdapter()
    return HistDataAdapter()


# ---------------------------------------------------------------------------
# Future adapter stubs (implement without touching the rest of the system)
# ---------------------------------------------------------------------------

class MT5Adapter(DataAdapter):
    """MetaTrader 5 CSV export adapter. To be implemented."""

    def load(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("MT5Adapter is not yet implemented.")


class DukascopyAdapter(DataAdapter):
    """Dukascopy JForex CSV adapter. To be implemented."""

    def load(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("DukascopyAdapter is not yet implemented.")


class CustomAdapter(DataAdapter):
    """
    User-defined column mapping adapter.

    Parameters
    ----------
    column_map : dict
        Mapping from raw CSV column names to standard names.
    datetime_col : str
        Name of the combined datetime column (if date+time are merged).
    datetime_format : str
        strptime format string for the datetime column.
    """

    def __init__(
        self,
        column_map: dict[str, str],
        datetime_col: str = "datetime",
        datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        self.column_map = column_map
        self.datetime_col = datetime_col
        self.datetime_format = datetime_format

    def load(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("CustomAdapter.load() must be fully implemented.")


# ---------------------------------------------------------------------------
# Tick data adapter
# ---------------------------------------------------------------------------

class TickDataAdapter(DataAdapter):
    """
    Adapter for tick-by-tick data.

    Expected CSV columns: timestamp, bid, ask, volume

    Builds M1 candles from ticks using TickCandleBuilder, then
    returns standard OHLCV DataFrame compatible with MarketDataStore.
    """

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        bid_col: str = "bid",
        ask_col: str = "ask",
        volume_col: str = "volume",
    ) -> None:
        self.timestamp_col = timestamp_col
        self.bid_col = bid_col
        self.ask_col = ask_col
        self.volume_col = volume_col

    def load(self, path: str) -> pd.DataFrame:
        from core.tick_candle_builder import TickCandleBuilder

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Tick data file not found: {path}")

        logger.info("Loading tick data file: %s", path)

        # Detect delimiter
        with open(file_path, "r") as f:
            first_line = f.readline()
        delimiter = ";" if ";" in first_line else ","

        # Load tick CSV
        df = pd.read_csv(
            file_path,
            dtype=str,
            low_memory=False,
            sep=delimiter,
        )
        df.columns = [c.strip().lower() for c in df.columns]

        # Parse timestamps
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        else:
            # Try to auto-detect datetime column
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].head(5))
                    df.rename(columns={col: self.timestamp_col}, inplace=True)
                    df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
                    break
                except (ValueError, TypeError):
                    continue
            else:
                raise ValueError(
                    f"Cannot find timestamp column. Available: {list(df.columns)}"
                )

        # Rename columns if needed
        rename_map = {}
        if self.bid_col not in df.columns:
            # Try common alternatives
            for alt in ("bid", "Bid", "BID", "bid_price", "BidPrice"):
                if alt in df.columns:
                    rename_map[alt] = self.bid_col
                    break
        if self.ask_col not in df.columns:
            for alt in ("ask", "Ask", "ASK", "ask_price", "AskPrice"):
                if alt in df.columns:
                    rename_map[alt] = self.ask_col
                    break
        if self.volume_col not in df.columns:
            for alt in ("volume", "Volume", "VOLUME", "vol", "Vol"):
                if alt in df.columns:
                    rename_map[alt] = self.volume_col
                    break

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # Validate required columns
        missing = []
        if self.bid_col not in df.columns:
            missing.append(self.bid_col)
        if self.ask_col not in df.columns:
            missing.append(self.ask_col)
        if missing:
            raise ValueError(
                f"Tick data missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Expected: {self.timestamp_col}, {self.bid_col}, {self.ask_col}, {self.volume_col}"
            )

        # Cast to numeric
        df[self.bid_col] = pd.to_numeric(df[self.bid_col], errors="coerce")
        df[self.ask_col] = pd.to_numeric(df[self.ask_col], errors="coerce")
        if self.volume_col in df.columns:
            df[self.volume_col] = pd.to_numeric(df[self.volume_col], errors="coerce").fillna(0)
        else:
            df[self.volume_col] = 0

        # Drop rows with NaN prices
        before = len(df)
        df.dropna(subset=[self.bid_col, self.ask_col], inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with NaN prices.", dropped)

        logger.info(
            "Loaded %d ticks from %s to %s",
            len(df), df[self.timestamp_col].min(), df[self.timestamp_col].max(),
        )

        # Build M1 candles from ticks
        builder = TickCandleBuilder()
        m1 = builder.build_m1(
            df,
            timestamp_col=self.timestamp_col,
            bid_col=self.bid_col,
            ask_col=self.ask_col,
            volume_col=self.volume_col,
        )

        return m1
