"""
detectors/pattern_detector.py
------------------------------
Strategy-based pattern detection with multi-timeframe support.

Delegates evaluation to a pluggable BaseStrategy instance.
Fetches M1/M5/H1 windows and passes all to the strategy.
Generates signals with "direction" in metadata (LONG / SHORT).
"""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from core.event_bus import EventBus
from core.events import MarketTickEvent, PatternDetectedEvent
from core.market_data_store import MarketDataStore
from detectors.signal import PatternSignal
from detectors.strategies.base import BaseStrategy
from detectors.strategies.ema_stochastic import EMAStochasticStrategy

logger = logging.getLogger(__name__)

# Timeframes fetched for MTF strategies (order matters: finest → coarsest)
MTF_TIMEFRAMES: list[str] = ["M1", "M5", "H1"]


class PatternDetector:
    def __init__(
        self,
        event_bus: EventBus,
        data_store: MarketDataStore,
        symbol: str = "EURUSD",
        lookback: int = 200,
        strategy: Optional[BaseStrategy] = None,
    ) -> None:
        self._bus = event_bus
        self._store = data_store
        self._symbol = symbol
        self._lookback = lookback
        self._strategy = strategy or EMAStochasticStrategy()

        self._signals: list[PatternSignal] = []
        self._bus.subscribe(MarketTickEvent, self._on_market_tick)

        logger.info(
            "PatternDetector initialised (strategy=%s, lookback=%d).",
            self._strategy.name,
            lookback,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> BaseStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: BaseStrategy) -> None:
        """Hot-swap strategy at runtime."""
        self._strategy = new_strategy
        self._signals.clear()
        logger.info("Strategy swapped to %s", new_strategy.name)

    @property
    def signals(self) -> list[PatternSignal]:
        return list(self._signals)

    @property
    def signal_count(self) -> int:
        return len(self._signals)

    def reset(self) -> None:
        self._signals.clear()

    # ------------------------------------------------------------------
    # Core: fetch windows + delegate to strategy
    # ------------------------------------------------------------------

    def _fetch_windows(self, timestamp: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """
        Fetch M1/M5/H1 windows up to current timestamp.
        Missing TFs are silently omitted (strategy decides what's required).
        """
        windows: dict[str, pd.DataFrame] = {}
        for tf in MTF_TIMEFRAMES:
            try:
                windows[tf] = self._store.get_window(
                    symbol=self._symbol,
                    timeframe=tf,
                    current_timestamp=timestamp,
                    lookback=self._lookback,
                )
            except KeyError:
                logger.debug("TF %s not available for %s, skipping.", tf, self._symbol)
        return windows

    def scan_for_patterns(
        self,
        windows: dict[str, pd.DataFrame],
        current_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        return self._strategy.evaluate(windows, current_timestamp)

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _on_market_tick(self, event: MarketTickEvent) -> None:
        windows = self._fetch_windows(event.timestamp)

        new_signals = self.scan_for_patterns(windows, event.timestamp)

        for signal in new_signals:
            self._signals.append(signal)
            self._bus.publish(PatternDetectedEvent(
                pattern_name=signal.name,
                timestamp=signal.end_time,
                confidence=signal.confidence,
                metadata=signal.metadata,
                symbol=event.symbol,
                timeframe="M1",
            ))
