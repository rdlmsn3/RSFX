"""
core/trade_engine.py
--------------------
Unified trade execution engine — tick-driven.

There is no bar-level execution path anymore. Signals are queued, then
filled on the very next tick at that tick's ask (LONG) or bid (SHORT).
TP/SL distances are captured at signal time and re-projected from the
real fill price, so R:R is preserved exactly regardless of the gap
between "candle closed" and "order actually filled."

Usage:
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01)
    engine = TradeEngine(config)

    for ts, bid, ask, vol in tick_stream:
        engine.on_tick(bid, ask, ts)          # manage open pos + fill queue
        bar = m1_builder.ingest_tick(ts, bid, ask, vol)
        if bar:
            m1_arrays.append(bar)
            for sig in signal_engine.evaluate(m1_arrays.n - 1, m1_arrays, tf_arrays):
                engine.queue_order(sig)        # NOT engine.open() anymore
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.event_bus import EventBus
from core.engine import compute_pnl, apply_spread, check_min_rr, check_dedup, update_equity
from core.events import SignalEvent, TradeEvent

logger = logging.getLogger(__name__)


@dataclass
class TradeConfig:
    """Trade execution parameters."""
    symbol: str = "USDJPY"
    pip_value: float = 0.01
    lot_size: float = 0.01
    initial_balance: float = 10_000.0
    spread_pips: float = 0.5
    slippage_pips: float = 0.0
    min_rr: float = 1.0
    use_sr: bool = False
    max_lookback: int = 100


@dataclass
class TradeRecord:
    """Immutable record of a completed trade."""
    id: str
    strategy: str
    direction: str          # "LONG" | "SHORT"
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""   # "TP" | "SL" | "EOD" | "MANUAL"
    pnl_pips: float = 0.0
    mae_pips: float = 0.0
    mfe_pips: float = 0.0
    ticks_held: int = 0
    risk_pips: float = 0.0
    reward_pips: float = 0.0
    rr_ratio: float = 0.0
    spread_cost: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "direction": self.direction,
            "entry_time": str(self.entry_time) if self.entry_time else "",
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_time": str(self.exit_time) if self.exit_time else "",
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pips": round(self.pnl_pips, 2),
            "mae_pips": round(self.mae_pips, 2),
            "mfe_pips": round(self.mfe_pips, 2),
            "ticks_held": self.ticks_held,
            "risk_pips": round(self.risk_pips, 2),
            "reward_pips": round(self.reward_pips, 2),
            "rr_ratio": round(self.rr_ratio, 2),
            "spread_cost": round(self.spread_cost, 2),
            **{f"sig_{k}": v for k, v in self.metadata.items()
               if isinstance(v, (int, float, str, bool))},
        }


@dataclass
class PendingOrder:
    """
    A signal that has been accepted but not yet filled.

    Only risk/reward *distances* are kept, not absolute prices — the
    absolute SL/TP are re-derived once we know the real fill price.
    """
    id: str
    strategy: str
    direction: str
    queued_time: pd.Timestamp
    intended_entry: float       # signal-time price, used only for dedup/sanity
    risk_pips: float            # |intended_entry - intended_sl| / pip_value
    reward_pips: float          # |intended_tp - intended_entry| / pip_value
    metadata: dict = field(default_factory=dict)


@dataclass
class OpenPosition:
    """Live position being tracked tick-by-tick."""
    id: str
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    tick_count: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    metadata: dict = field(default_factory=dict)


class TradeEngine:
    """
    Unified, tick-driven trade execution engine.

    Lifecycle per signal:
      1. queue_order(signal)  -> stored as PendingOrder (no price taken yet)
      2. on_tick(...)         -> first tick to arrive fills it at ask/bid
      3. on_tick(...)         -> every subsequent tick checks SL/TP on bid/ask

    The engine never looks at any tick that hasn't been handed to it via
    on_tick() — there is no lookahead path left.
    """

    def __init__(self, config: TradeConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self._bus = event_bus
        self._open: Optional[OpenPosition] = None
        self._pending: Optional[PendingOrder] = None
        self._trades: list[TradeRecord] = []
        self._balance_curve: list[float] = [config.initial_balance]
        self._peak_balance: float = config.initial_balance
        self._max_dd: float = 0.0
        self._realized_pnl: float = 0.0
        self._last_entry_price: Optional[float] = None

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def trades(self) -> list[TradeRecord]:
        return list(self._trades)

    @property
    def open_position(self) -> Optional[OpenPosition]:
        return self._open

    @property
    def pending_order(self) -> Optional[PendingOrder]:
        return self._pending

    @property
    def balance_curve(self) -> list[float]:
        return list(self._balance_curve)

    @property
    def max_drawdown(self) -> float:
        return self._max_dd

    @property
    def realized_pnl_pips(self) -> float:
        return self._realized_pnl

    # ------------------------------------------------------------------
    # Signal intake — queues only, never fills directly
    # ------------------------------------------------------------------

    def queue_order(self, signal: SignalEvent) -> None:
        """
        Accept a signal at candle-close and queue it for execution on the
        next tick. Does NOT touch price/state beyond sanity + dedup checks.
        """
        if self._open is not None or self._pending is not None:
            return  # one position/order in flight at a time

        direction = signal.direction
        if direction not in ("LONG", "SHORT"):
            return

        entry = signal.entry_price
        tp = signal.take_profit
        sl = signal.stop_loss
        if not tp or not sl:
            return

        # Sanity (same rule as before: TP/SL must sit on the correct side)
        if direction == "LONG":
            if tp <= entry:
                tp = entry + abs(tp - entry)
            if sl >= entry:
                sl = entry - abs(sl - entry)
        else:
            if tp >= entry:
                tp = entry - abs(tp - entry)
            if sl <= entry:
                sl = entry + abs(sl - entry)

        if check_dedup(self._last_entry_price, entry):
            return

        if not check_min_rr(entry, tp, sl, self.config.pip_value, self.config.min_rr):
            return

        risk_pips = abs(entry - sl) / self.config.pip_value
        reward_pips = abs(tp - entry) / self.config.pip_value

        self._pending = PendingOrder(
            id=str(uuid.uuid4())[:8],
            strategy=signal.strategy_name,
            direction=direction,
            queued_time=signal.timestamp,
            intended_entry=entry,
            risk_pips=risk_pips,
            reward_pips=reward_pips,
            metadata=signal.metadata,
        )

        if self._bus:
            self._bus.publish(TradeEvent(
                action="QUEUED", price=entry, timestamp=signal.timestamp,
                symbol=self.config.symbol,
                metadata={"order_id": self._pending.id, "direction": direction},
            ))

    def cancel_pending(self) -> None:
        """Drop a queued order without filling it (e.g. end of session)."""
        self._pending = None

    # ------------------------------------------------------------------
    # The only execution entry point
    # ------------------------------------------------------------------

    def on_tick(self, bid: float, ask: float, timestamp: pd.Timestamp) -> Optional[TradeRecord]:
        """
        Process a single tick:
          1. If a position is open, check SL/TP against bid/ask.
          2. If flat and an order is pending, fill it on THIS tick.

        Both can happen on the same tick (a close freeing up the slot,
        immediately followed by the queued order filling) — that's still
        "the very next tick after queuing," it just happens to coincide
        with an exit.
        """
        closed: Optional[TradeRecord] = None

        if self._open is not None:
            closed = self._check_exit(bid, ask, timestamp)

        if self._open is None and self._pending is not None:
            self._fill_pending(bid, ask, timestamp)

        return closed

    def force_close(self, price: float, timestamp: pd.Timestamp, reason: str = "EOD") -> Optional[TradeRecord]:
        """Force-close the open position (e.g. end of backtest)."""
        if self._open is None:
            return None
        return self._close(price, timestamp, reason)

    def get_stats(self) -> dict:
        """Compute standard stats from trade history."""
        from core.engine import build_result
        return build_result("ENGINE", self._trades, self._max_dd, self._balance_curve)

    def mark_to_market(self, current_close: float) -> None:
        """
        Update the balance/equity curve off a completed candle close.
        Purely for analytics/charting — never touches positions or
        triggers fills/exits. Call this once per finished M1 bar if you
        want a candle-resolution equity curve; safe to skip entirely.
        """
        self._update_equity(current_close)

    def reset(self) -> None:
        """Clear all state for a new backtest."""
        self._open = None
        self._pending = None
        self._trades.clear()
        self._balance_curve = [self.config.initial_balance]
        self._peak_balance = self.config.initial_balance
        self._max_dd = 0.0
        self._realized_pnl = 0.0
        self._last_entry_price = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fill_pending(self, bid: float, ask: float, timestamp: pd.Timestamp) -> None:
        order = self._pending
        assert order is not None

        fill_price = ask if order.direction == "LONG" else bid

        # Re-anchor: preserve the *distances*, not the absolute levels,
        # so R:R survives whatever slippage happened between the signal
        # candle close and this real fill.
        pip = self.config.pip_value
        if order.direction == "LONG":
            sl = fill_price - order.risk_pips * pip
            tp = fill_price + order.reward_pips * pip
        else:
            sl = fill_price + order.risk_pips * pip
            tp = fill_price - order.reward_pips * pip

        pos = OpenPosition(
            id=order.id,
            strategy=order.strategy,
            direction=order.direction,
            entry_time=timestamp,
            entry_price=fill_price,
            stop_loss=sl,
            take_profit=tp,
            metadata={**order.metadata, "signal_time": str(order.queued_time)},
        )
        self._open = pos
        self._pending = None
        self._last_entry_price = fill_price

        if self._bus:
            self._bus.publish(TradeEvent(
                action="OPEN", price=fill_price, timestamp=timestamp,
                symbol=self.config.symbol,
                metadata={"position_id": pos.id, "direction": pos.direction},
            ))

    def _check_exit(self, bid: float, ask: float, timestamp: pd.Timestamp) -> Optional[TradeRecord]:
        pos = self._open
        pos.tick_count += 1

        if pos.direction == "LONG":
            adverse = (bid - pos.entry_price) / self.config.pip_value
            favorable = (bid - pos.entry_price) / self.config.pip_value
            pos.mae = min(pos.mae, adverse)
            pos.mfe = max(pos.mfe, favorable)
            if bid <= pos.stop_loss:
                return self._close(pos.stop_loss, timestamp, "SL")
            if bid >= pos.take_profit:
                return self._close(pos.take_profit, timestamp, "TP")
        else:
            adverse = (pos.entry_price - ask) / self.config.pip_value
            favorable = (pos.entry_price - ask) / self.config.pip_value
            pos.mae = min(pos.mae, adverse)
            pos.mfe = max(pos.mfe, favorable)
            if ask >= pos.stop_loss:
                return self._close(pos.stop_loss, timestamp, "SL")
            if ask <= pos.take_profit:
                return self._close(pos.take_profit, timestamp, "TP")
        return None

    def _close(self, exit_price: float, timestamp: pd.Timestamp, reason: str) -> TradeRecord:
        pos = self._open
        assert pos is not None

        pnl = compute_pnl(pos.direction, pos.entry_price, exit_price, self.config.pip_value)
        spread_cost = self.config.spread_pips + self.config.slippage_pips
        net_pnl = apply_spread(pnl, spread_cost)

        risk = abs(pos.entry_price - pos.stop_loss) / self.config.pip_value
        reward = abs(pos.take_profit - pos.entry_price) / self.config.pip_value

        trade = TradeRecord(
            id=pos.id, strategy=pos.strategy, direction=pos.direction,
            entry_time=pos.entry_time, entry_price=pos.entry_price,
            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
            exit_time=timestamp, exit_price=exit_price, exit_reason=reason,
            pnl_pips=round(net_pnl, 2), mae_pips=round(pos.mae, 2),
            mfe_pips=round(pos.mfe, 2), ticks_held=pos.tick_count,
            risk_pips=round(risk, 2), reward_pips=round(reward, 2),
            rr_ratio=round(reward / risk, 2) if risk > 0 else 0.0,
            spread_cost=round(spread_cost, 2), metadata=pos.metadata,
        )

        self._trades.append(trade)
        self._realized_pnl += net_pnl
        self._open = None

        if self._bus:
            self._bus.publish(TradeEvent(
                action="CLOSE", price=exit_price, timestamp=timestamp,
                symbol=self.config.symbol,
                metadata={"position_id": pos.id, "pnl": net_pnl, "reason": reason},
            ))

        return trade

    def _update_equity(self, current_close: float) -> None:
        open_price = self._open.entry_price if self._open else None
        direction = self._open.direction if self._open else ""
        self._balance_curve, self._peak_balance, self._max_dd = update_equity(
            open_price=open_price,
            direction=direction,
            current_close=current_close,
            realized_pnl_pips=self._realized_pnl,
            pip_value=self.config.pip_value,
            initial_balance=self.config.initial_balance,
            lot_size=self.config.lot_size,
            balance_curve=self._balance_curve,
            peak_balance=self._peak_balance,
            max_dd=self._max_dd,
        )
