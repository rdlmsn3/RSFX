"""
core/candle_stream.py
----------------------
Incremental, tick-driven candle construction.

Replaces the "build the whole DataFrame up front, then walk it with an
index" model with a pure streaming model: feed it ticks one at a time,
and it tells you exactly when a candle has finished forming.

No lookahead is possible here by construction — a Bar is only ever
emitted once a tick belonging to the *next* bucket has arrived, which
is the same moment a live feed would tell you the candle is closed.

Usage
-----
    m1_builder = IncrementalCandleBuilder("M1")
    m5_builder = IncrementalCandleBuilder("M5")
    m1_arrays = StreamingCandleArrays()
    m5_arrays = StreamingCandleArrays()

    for ts, bid, ask, vol in tick_stream:
        bar = m1_builder.ingest_tick(ts, bid, ask, vol)
        if bar:
            m1_arrays.append(bar)
        bar5 = m5_builder.ingest_tick(ts, bid, ask, vol)
        if bar5:
            m5_arrays.append(bar5)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.tick_candle_builder import _TF_MAP

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single finished OHLCV candle."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class IncrementalCandleBuilder:
    """
    Stateful single-timeframe OHLCV aggregator driven by individual ticks.

    Call `ingest_tick()` once per tick. Returns a finished `Bar` the
    moment the timestamp crosses into a new bucket (i.e. the previous
    bucket is now provably closed), otherwise returns None.
    """

    def __init__(self, timeframe: str = "M1") -> None:
        if timeframe.upper() not in _TF_MAP:
            raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(_TF_MAP.keys())}")
        self.timeframe = timeframe.upper()
        self._freq = _TF_MAP[self.timeframe]

        self._bucket_start: Optional[pd.Timestamp] = None
        self._o = self._h = self._l = self._c = None
        self._vol: float = 0.0

    def ingest_tick(self, ts: pd.Timestamp, bid: float, ask: float, volume: float = 0.0) -> Optional[Bar]:
        ts = pd.Timestamp(ts)
        mid = (bid + ask) / 2.0
        bucket = ts.floor(self._freq)

        if self._bucket_start is None:
            self._start_bucket(bucket, mid, volume)
            return None

        if bucket == self._bucket_start:
            self._h = max(self._h, mid)
            self._l = min(self._l, mid)
            self._c = mid
            self._vol += volume
            return None

        # Bucket boundary crossed -> the previous candle is now closed.
        finished = Bar(
            timestamp=self._bucket_start,
            open=self._o, high=self._h, low=self._l, close=self._c,
            volume=self._vol,
        )
        self._start_bucket(bucket, mid, volume)
        return finished

    def flush(self) -> Optional[Bar]:
        """Call at end-of-stream to emit the final, still-open bucket."""
        if self._bucket_start is None:
            return None
        bar = Bar(
            timestamp=self._bucket_start,
            open=self._o, high=self._h, low=self._l, close=self._c,
            volume=self._vol,
        )
        self._bucket_start = None
        return bar

    def _start_bucket(self, bucket: pd.Timestamp, mid: float, volume: float) -> None:
        self._bucket_start = bucket
        self._o = self._h = self._l = self._c = mid
        self._vol = volume


class StreamingCandleArrays:
    """
    Append-only, growing replacement for a pre-built CandleArrays.

    Exposes the same attribute surface SignalEngine/strategies expect
    (.timestamps, .opens, .highs, .lows, .closes, .volumes, .n) so it
    drops straight into `signal_engine.evaluate(i, arrays, tf_arrays)`
    without touching strategy code.

    Numpy views are rebuilt lazily and only when a new bar is appended
    (once per candle close, not once per tick), so this stays cheap
    even though it looks like a full rebuild.
    """

    def __init__(self) -> None:
        self._ts: list[np.datetime64] = []
        self._o: list[float] = []
        self._h: list[float] = []
        self._l: list[float] = []
        self._c: list[float] = []
        self._v: list[float] = []
        self._dirty = True
        self._cache: dict[str, np.ndarray] = {}

    def append(self, bar: Bar) -> None:
        self._ts.append(np.datetime64(bar.timestamp))
        self._o.append(bar.open)
        self._h.append(bar.high)
        self._l.append(bar.low)
        self._c.append(bar.close)
        self._v.append(bar.volume)
        self._dirty = True

    def _rebuild(self) -> None:
        self._cache = {
            "timestamps": np.array(self._ts),
            "opens": np.array(self._o, dtype=float),
            "highs": np.array(self._h, dtype=float),
            "lows": np.array(self._l, dtype=float),
            "closes": np.array(self._c, dtype=float),
            "volumes": np.array(self._v, dtype=float),
        }
        self._dirty = False

    def _get(self, key: str) -> np.ndarray:
        if self._dirty:
            self._rebuild()
        return self._cache[key]

    @property
    def n(self) -> int:
        return len(self._ts)

    @property
    def timestamps(self) -> np.ndarray:
        return self._get("timestamps")

    @property
    def opens(self) -> np.ndarray:
        return self._get("opens")

    @property
    def highs(self) -> np.ndarray:
        return self._get("highs")

    @property
    def lows(self) -> np.ndarray:
        return self._get("lows")

    @property
    def closes(self) -> np.ndarray:
        return self._get("closes")

    @property
    def volumes(self) -> np.ndarray:
        return self._get("volumes")
