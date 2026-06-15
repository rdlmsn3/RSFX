"""
core/data_loader.py
-------------------
Flexible, adapter-pattern data loading system.

The DataAdapter abstract base class defines the contract.
Concrete adapters handle provider-specific CSV formats without
touching the rest of the system.

Existing adapters
-----------------
    HistDataAdapter   – HistData.com  (Date,Time,Open,High,Low,Close,Volume)

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
        if not has_header:
            col_count = len(df.columns)
            # Detect combined datetime column (space in first value)
            first_val = str(df.iloc[0, 0])
            if ' ' in first_val and first_val[:4].isdigit():
                # Format: datetime, open, high, low, close, volume
                expected_cols = ["datetime", "open", "high", "low", "close", "volume"]
                df.columns = expected_cols[:col_count]
            else:
                # Separate date + time
                positional = ["date", "time", "open", "high", "low", "close", "volume"]
                df.columns = positional[:col_count]
        else:
            df.columns = [c.strip().lower() for c in df.columns]
            df.rename(columns=self._COLUMN_MAP, inplace=True)

        # ---- Build timestamp index ----
        df = self._build_timestamp_index(df)

        # ... rest unchanged (validate, cast to numeric, dropna, sort, dedup)
        # (copy the rest of the method from your original file)

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
            df["time"] = df["time"].str.strip().str.zfill(6)
            df["date"] = df["date"].str.strip()
            combined = df["date"] + " " + df["time"]
            # Try two common HistData datetime formats
            for fmt in ("%Y%m%d %H%M%S", "%Y.%m.%d %H%M%S", "%m/%d/%Y %H%M%S"):
                try:
                    ts = pd.to_datetime(combined, format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
            else:
                ts = pd.to_datetime(combined, infer_datetime_format=True)

            df = df.drop(columns=["date", "time"])
        elif "datetime" in df.columns:
            ts = pd.to_datetime(df["datetime"], infer_datetime_format=True)
            df = df.drop(columns=["datetime"])
        else:
            raise ValueError(
                "Cannot find date/time columns. Expected 'date'+'time' or 'datetime'."
            )

        df.index = ts
        df.index.name = "timestamp"
        return df


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