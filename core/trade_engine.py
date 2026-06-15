"""
core/trade_engine.py
--------------------
Trade simulation engine.

Currently a structured placeholder ready for strategy integration.

Future integrations (subscribe to events without modifying this class)
----------------------------------------------------------------------
- StrategyEngine publishes TradeSignalEvent → TradeEngine opens positions
- RiskManager publishes RiskOverrideEvent → TradeEngine modifies SL/TP
- PerformanceAnalytics subscribes to TradeEvent → computes P&L metrics
- JournalSystem subscribes to TradeEvent → persists trade records
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.event_bus import EventBus
from core.events import MarketTickEvent, PatternDetectedEvent, TradeEvent

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single open or closed trade position."""
    id: str
    symbol: str
    direction: str                    # "BUY" | "SELL"
    entry_price: float
    entry_time: pd.Timestamp
    lot_size: float = 0.01
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    status: str = "OPEN"              # "OPEN" | "CLOSED"
    metadata: dict = field(default_factory=dict)


class TradeEngine:
    """
    Manages simulated trading positions.

    Subscribes to MarketTickEvent to:
    - Mark-to-market open positions (future)
    - Check SL/TP hit conditions (future)
    - Update running P&L (future)

    Parameters
    ----------
    event_bus : EventBus
        Shared event bus.
    symbol : str
        Active trading symbol.
    """

    def __init__(self, event_bus: EventBus, symbol: str = "EURUSD", data_store=None) -> None:
        self._bus = event_bus
        self._symbol = symbol
        self._store = data_store

        self._open_positions: dict[str, Position] = {}
        self._closed_positions: list[Position] = []
        self._trade_history: list[dict] = []        # flat record for analytics

        # Subscribe to tick events for position management
        self._bus.subscribe(MarketTickEvent, self._on_market_tick)
        # Subscribe to pattern events for order generation
        self._bus.subscribe(PatternDetectedEvent, self._on_pattern_detected)

        logger.info("TradeEngine initialised for %s.", symbol)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def open_positions(self) -> list[Position]:
        return list(self._open_positions.values())

    @property
    def closed_positions(self) -> list[Position]:
        return list(self._closed_positions)

    @property
    def trade_count(self) -> int:
        return len(self._closed_positions) + len(self._open_positions)

    def open_position(
        self,
        direction: str,
        price: float,
        timestamp: pd.Timestamp,
        lot_size: float = 0.01,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Position:
        """
        Open a new simulated position.

        Parameters
        ----------
        direction : str
            "BUY" or "SELL".
        price : float
            Entry price.
        timestamp : pd.Timestamp
            Entry timestamp.
        lot_size : float
            Trade size in lots.
        stop_loss : float, optional
        take_profit : float, optional
        metadata : dict, optional
            Arbitrary key-value pairs (e.g. {"strategy": "breakout"}).

        Returns
        -------
        Position
            The newly created position.
        """
        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=self._symbol,
            direction=direction.upper(),
            entry_price=price,
            entry_time=timestamp,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {},
        )
        self._open_positions[position.id] = position

        self._bus.publish(TradeEvent(
            action="OPEN",
            price=price,
            timestamp=timestamp,
            symbol=self._symbol,
            metadata={"position_id": position.id, "direction": direction},
        ))

        logger.info("Opened %s position %s at %.5f", direction, position.id, price)
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: pd.Timestamp,
        metadata: Optional[dict] = None,
    ) -> Optional[Position]:
        """
        Close an open position and calculate P&L.

        Parameters
        ----------
        position_id : str
            The ID returned by open_position().
        exit_price : float
        exit_time : pd.Timestamp
        metadata : dict, optional

        Returns
        -------
        Position or None
            The closed position, or None if position_id not found.
        """
        position = self._open_positions.pop(position_id, None)
        if position is None:
            logger.warning("close_position: unknown position_id '%s'", position_id)
            return None

        position.exit_price = exit_price
        position.exit_time = exit_time
        position.status = "CLOSED"
        position.metadata.update(metadata or {})

        # Basic P&L calculation (pip-based; extend for proper lot sizing)
        multiplier = 1 if position.direction == "BUY" else -1
        position.pnl = multiplier * (exit_price - position.entry_price) * position.lot_size * 100_000

        self._closed_positions.append(position)
        self._trade_history.append({
            "id": position.id,
            "symbol": position.symbol,
            "direction": position.direction,
            "entry": position.entry_price,
            "exit": exit_price,
            "pnl": position.pnl,
            "entry_time": position.entry_time,
            "exit_time": exit_time,
        })

        self._bus.publish(TradeEvent(
            action="CLOSE",
            price=exit_price,
            timestamp=exit_time,
            symbol=self._symbol,
            metadata={"position_id": position_id, "pnl": position.pnl},
        ))

        logger.info("Closed position %s at %.5f, PnL=%.2f", position_id, exit_price, position.pnl)
        return position

    def modify_stop_loss(self, position_id: str, new_sl: float, timestamp: pd.Timestamp) -> bool:
        """
        Modify the stop loss of an open position.

        Future: RiskManager can call this directly or via event.
        """
        position = self._open_positions.get(position_id)
        if position is None:
            return False
        position.stop_loss = new_sl

        self._bus.publish(TradeEvent(
            action="MODIFY_SL",
            price=new_sl,
            timestamp=timestamp,
            symbol=self._symbol,
            metadata={"position_id": position_id},
        ))
        return True

    def modify_take_profit(self, position_id: str, new_tp: float, timestamp: pd.Timestamp) -> bool:
        """
        Modify the take profit of an open position.

        Future: Strategy Engine can trail TP via this method.
        """
        position = self._open_positions.get(position_id)
        if position is None:
            return False
        position.take_profit = new_tp

        self._bus.publish(TradeEvent(
            action="MODIFY_TP",
            price=new_tp,
            timestamp=timestamp,
            symbol=self._symbol,
            metadata={"position_id": position_id},
        ))
        return True

    def reset(self) -> None:
        """Clear all positions (called on playback reset)."""
        self._open_positions.clear()
        self._closed_positions.clear()
        self._trade_history.clear()
        logger.info("TradeEngine reset.")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_pattern_detected(self, event: PatternDetectedEvent) -> None:
        """
        Open a position when a confluence signal fires.
        Only trades signals marked as confluence (metadata["confluence"] == True).
        """
        meta = event.metadata

        # Only trade confluence signals
        if not meta.get("confluence", False):
            return

        direction = meta.get("direction", "")
        if direction not in ("LONG", "SHORT"):
            return

        entry_price = meta.get("entry_price")
        if entry_price is None:
            # Fallback: use event price if available
            return

        # Map signal direction to position direction
        pos_direction = "BUY" if direction == "LONG" else "SELL"

        self.open_position(
            direction=pos_direction,
            price=entry_price,
            timestamp=event.timestamp,
            stop_loss=meta.get("stop_loss"),
            take_profit=meta.get("take_profit"),
            metadata={
                "strategy": meta.get("strategy", ""),
                "confluence_count": meta.get("confluence_count", 0),
                "signal_name": event.pattern_name,
            },
        )

    def _on_market_tick(self, event: MarketTickEvent) -> None:
        """
        Check open positions for SL/TP breach on each tick.
        Closes positions that hit stop loss or take profit.
        """
        if not self._open_positions or self._store is None:
            return

        # Get current close price from the data store
        try:
            m1_df = self._store.get_window(
                symbol=self._symbol,
                timeframe="M1",
                current_timestamp=event.timestamp,
                lookback=1,
            )
            if m1_df.empty:
                return
            current_price = float(m1_df["close"].iloc[-1])
        except Exception:
            return

        to_close: list[tuple[str, float, str]] = []

        for pos_id, pos in self._open_positions.items():
            if pos.direction == "BUY":
                if pos.stop_loss is not None and current_price <= pos.stop_loss:
                    to_close.append((pos_id, pos.stop_loss, "SL"))
                elif pos.take_profit is not None and current_price >= pos.take_profit:
                    to_close.append((pos_id, pos.take_profit, "TP"))
            else:  # SELL
                if pos.stop_loss is not None and current_price >= pos.stop_loss:
                    to_close.append((pos_id, pos.stop_loss, "SL"))
                elif pos.take_profit is not None and current_price <= pos.take_profit:
                    to_close.append((pos_id, pos.take_profit, "TP"))

        for pos_id, exit_price, reason in to_close:
            self.close_position(
                position_id=pos_id,
                exit_price=exit_price,
                exit_time=event.timestamp,
                metadata={"exit_reason": reason},
            )