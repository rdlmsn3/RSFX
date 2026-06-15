"""
core/playback_controller.py
---------------------------
Stateful market replay controller.

The PlaybackController owns the playback cursor and drives the replay loop.
It knows nothing about UI, pattern detection, or trade execution.
All side-effects are triggered through MarketTickEvent on the EventBus.

Responsibilities
----------------
- Track current candle index within the M1 series
- Expose play / pause / reset / step / seek controls
- Publish MarketTickEvent on every tick
- Enforce data boundaries (clamp to [0, len-1])
"""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from core.event_bus import EventBus
from core.events import MarketTickEvent
from core.market_data_store import MarketDataStore

logger = logging.getLogger(__name__)


class PlaybackController:
    """
    Controls the replay cursor and publishes tick events.

    Parameters
    ----------
    event_bus : EventBus
        Shared event bus instance.
    data_store : MarketDataStore
        Loaded market data store.
    symbol : str
        Active symbol to replay (default "EURUSD").
    """

    def __init__(
        self,
        event_bus: EventBus,
        data_store: MarketDataStore,
        symbol: str = "EURUSD",
    ) -> None:
        self._bus = event_bus
        self._store = data_store
        self._symbol = symbol

        # --- State ---
        self.current_index: int = 0
        self.is_playing: bool = False
        self._total_candles: int = data_store.length(symbol, "M1")

        logger.info(
            "PlaybackController initialised: %s, %d M1 candles",
            symbol,
            self._total_candles,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def total_candles(self) -> int:
        return self._total_candles

    @property
    def bars_remaining(self) -> int:
        return max(0, self._total_candles - self.current_index - 1)

    @property
    def current_timestamp(self) -> Optional[pd.Timestamp]:
        if self._total_candles == 0:
            return None
        idx = max(0, min(self.current_index, self._total_candles - 1))
        return self._store.get_timestamp_at_index(self._symbol, "M1", idx)

    @property
    def at_end(self) -> bool:
        return self.current_index >= self._total_candles - 1

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Begin (or resume) automatic playback."""
        if self.at_end:
            logger.info("Playback already at end; ignoring play().")
            return
        self.is_playing = True
        logger.debug("Playback started at index %d.", self.current_index)

    def pause(self) -> None:
        """Pause automatic playback."""
        self.is_playing = False
        logger.debug("Playback paused at index %d.", self.current_index)

    def reset(self) -> None:
        """Return cursor to the first candle and pause."""
        self.current_index = 0
        self.is_playing = False
        self._publish_tick()
        logger.info("Playback reset to index 0.")

    def step_forward(self) -> bool:
        """
        Advance one candle.

        Returns
        -------
        bool
            True if a tick was published, False if already at the end.
        """
        if self.at_end:
            self.is_playing = False
            logger.debug("step_forward(): already at end.")
            return False
        self.current_index += 1
        self._publish_tick()
        return True

    def step_backward(self) -> bool:
        """
        Move back one candle.

        Returns
        -------
        bool
            True if a tick was published, False if already at the start.
        """
        if self.current_index <= 0:
            logger.debug("step_backward(): already at start.")
            return False
        self.current_index -= 1
        self._publish_tick()
        return True

    def seek(self, timestamp: pd.Timestamp) -> None:
        """
        Jump the cursor to the candle closest to *timestamp*.

        Parameters
        ----------
        timestamp : pd.Timestamp
            Target timestamp.  Seeks to the nearest candle ≤ timestamp.
        """
        df = self._store.get_data(self._symbol, "M1")
        pos = int(df.index.searchsorted(timestamp, side="right")) - 1
        self.current_index = max(0, min(pos, self._total_candles - 1))
        self._publish_tick()
        logger.info("Seeked to index %d (%s).", self.current_index, self.current_timestamp)

    def seek_to_index(self, index: int) -> None:
        """Jump the cursor directly to a candle index."""
        self.current_index = max(0, min(index, self._total_candles - 1))
        self._publish_tick()

    def tick(self) -> bool:
        """
        Called once per Streamlit rerun cycle when is_playing is True.

        Advances by one candle and publishes a MarketTickEvent.

        Returns
        -------
        bool
            True if another tick should follow, False if end of data reached.
        """
        if not self.is_playing:
            return False
        advanced = self.step_forward()
        if not advanced:
            self.is_playing = False
        return advanced

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _publish_tick(self) -> None:
        """Publish the current cursor position as a MarketTickEvent."""
        ts = self.current_timestamp
        if ts is None:
            return
        event = MarketTickEvent(
            timestamp=ts,
            current_index=self.current_index,
            symbol=self._symbol,
            timeframe="M1",
        )
        self._bus.publish(event)
        logger.debug("Published MarketTickEvent index=%d ts=%s", self.current_index, ts)