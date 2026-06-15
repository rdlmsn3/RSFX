"""
detectors/pattern_detector.py
------------------------------
Strategy-based pattern detection with multi-timeframe support.

Delegates evaluation to one or more pluggable BaseStrategy instances.
Fetches M1/M5/H1 windows and passes all to each strategy.
Generates signals with "direction" in metadata (LONG / SHORT).

Supports single-strategy and multi-strategy (confluence) modes.
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
        strategies: Optional[list[BaseStrategy]] = None,
        confluence_threshold: int = 2,
    ) -> None:
        self._bus = event_bus
        self._store = data_store
        self._symbol = symbol
        self._lookback = lookback
        self._confluence_threshold = confluence_threshold

        # Multi-strategy support
        if strategies:
            self._strategies = list(strategies)
        elif strategy:
            self._strategies = [strategy]
        else:
            self._strategies = [EMAStochasticStrategy()]

        self._signals: list[PatternSignal] = []
        self._bus.subscribe(MarketTickEvent, self._on_market_tick)

        names = ", ".join(s.name for s in self._strategies)
        logger.info(
            "PatternDetector initialised (strategies=%s, lookback=%d).",
            names, lookback,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> BaseStrategy:
        """Primary strategy (first in list)."""
        return self._strategies[0]

    @strategy.setter
    def strategy(self, new_strategy: BaseStrategy) -> None:
        """Hot-swap primary strategy (replaces the list with a single strategy)."""
        self._strategies = [new_strategy]
        self._signals.clear()
        logger.info("Strategy swapped to %s", new_strategy.name)

    @property
    def strategies(self) -> list[BaseStrategy]:
        return list(self._strategies)

    @strategies.setter
    def strategies(self, new_strategies: list[BaseStrategy]) -> None:
        """Set multiple strategies for confluence mode."""
        self._strategies = list(new_strategies)
        self._signals.clear()
        names = ", ".join(s.name for s in self._strategies)
        logger.info("Strategies set to [%s]", names)

    @property
    def is_confluence_mode(self) -> bool:
        return len(self._strategies) > 1

    @property
    def confluence_threshold(self) -> int:
        return self._confluence_threshold

    @confluence_threshold.setter
    def confluence_threshold(self, value: int) -> None:
        self._confluence_threshold = max(1, value)
        logger.info("Confluence threshold set to %d", self._confluence_threshold)

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
        """Evaluate all strategies and return combined signals."""
        all_signals = []
        for s in self._strategies:
            signals = s.evaluate(windows, current_timestamp)
            for sig in signals:
                sig.metadata["strategy"] = s.name
            all_signals.extend(signals)
        return all_signals

    # ------------------------------------------------------------------
    # Confluence detection
    # ------------------------------------------------------------------

    def _mark_confluence(
        self,
        signals: list[PatternSignal],
        candle_timestamp: pd.Timestamp,
    ) -> list[PatternSignal]:
        """
        Mark confluence: when 2+ strategies agree on direction
        on the same candle.

        Adds metadata["confluence"] = True/False and
        metadata["confluence_count"] = N
        """
        if len(self._strategies) <= 1:
            for sig in signals:
                sig.metadata["confluence"] = False
                sig.metadata["confluence_count"] = 1
            return signals

        # Group by direction
        direction_counts: dict[str, int] = {}
        for sig in signals:
            d = sig.metadata.get("direction", "")
            if d in ("LONG", "SHORT"):
                direction_counts[d] = direction_counts.get(d, 0) + 1

        # Mark each signal
        for sig in signals:
            d = sig.metadata.get("direction", "")
            count = direction_counts.get(d, 0)
            sig.metadata["confluence"] = count >= self._confluence_threshold
            sig.metadata["confluence_count"] = count

        return signals

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _on_market_tick(self, event: MarketTickEvent) -> None:
        windows = self._fetch_windows(event.timestamp)

        new_signals = self.scan_for_patterns(windows, event.timestamp)

        # Mark confluence if multi-strategy
        new_signals = self._mark_confluence(new_signals, event.timestamp)

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
