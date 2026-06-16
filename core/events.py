"""
core/events.py
--------------
Event dataclasses for the Event-Driven Forex Market Replay Platform.

All inter-component communication is done through these event objects.
Adding new event types here does not require changes to any other module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class MarketTickEvent:
    """
    Published by PlaybackController on every tick (candle advance).

    Subscribers:
        - PatternDetector
        - TradeEngine
        - UI (via Streamlit session state bridge)

    Future subscribers:
        - MLEngine
        - RiskManager
        - PerformanceAnalytics
        - JournalSystem
    """
    timestamp: pd.Timestamp
    current_index: int
    symbol: str = "EURUSD"
    timeframe: str = "M1"


@dataclass
class PatternDetectedEvent:
    """
    Published by PatternDetector (or future MLEngine) when a pattern is found.

    Future use:
        - Candlestick pattern recognition results
        - Support/Resistance level breaks
        - ML model signal outputs
        - Neural network predictions
    """
    pattern_name: str
    timestamp: pd.Timestamp
    confidence: float                        # 0.0 – 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    symbol: str = "EURUSD"
    timeframe: str = "M1"


@dataclass
class TradeEvent:
    """
    Published by TradeEngine when a position is opened, modified, or closed.

    Future use:
        - Strategy Engine signals
        - Risk Manager overrides
        - ML-generated trade signals
    """
    action: str                              # "OPEN" | "CLOSE" | "MODIFY_SL" | "MODIFY_TP"
    price: float
    timestamp: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)
    symbol: str = "EURUSD"


@dataclass
class SignalEvent:
    """
    Published by a strategy when it fires a trading signal.

    Carries the full signal payload needed by TradeEngine to open/close positions.
    """
    strategy_name: str
    direction: str                            # "LONG" | "SHORT"
    entry_price: float
    take_profit: float
    stop_loss: float
    timestamp: pd.Timestamp
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BarEvent:
    """
    Published on every M1 bar by the bar aggregator / replay controller.

    Contains full OHLCV data for the bar.
    """
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    symbol: str = "USDJPY"
    timeframe: str = "M1"


@dataclass
class SystemEvent:
    """
    Internal system-level events (load, reset, error).

    Future use:
        - Database sync notifications
        - Secondary feed connect/disconnect
    """
    event_type: str                          # "DATA_LOADED" | "RESET" | "ERROR"
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)