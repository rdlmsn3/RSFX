"""
core/trade_engine.py
--------------------
Unified trade execution engine.

Works identically in backtest replay and live mode.
All three UIs (CLI, Backtest Web, Streamlit) consume this.

Usage:
    config = TradeConfig(symbol="USDJPY", pip_value=0.01, lot_size=0.01)
    engine = TradeEngine(config)

    # Bar-level backtest
    for bar in bars:
        signals = signal_engine.evaluate(i, arrays, tf_arrays)
        for sig in signals:
            engine.open(sig)
        engine.on_bar(BarEvent(...))

    # Or tick-level
    engine.on_tick(bid=120.50, ask=120.52, timestamp=ts)
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.event_bus import EventBus
from core.engine import compute_pnl, apply_spread, check_min_rr, check_dedup, update_equity
from core.events import (
    SignalEvent, BarEvent, TradeEvent, MarketTickEvent,
)

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
    bars_held: int = 0
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
            "bars_held": self.bars_held,
            "risk_pips": round(self.risk_pips, 2),
            "reward_pips": round(self.reward_pips, 2),
            "rr_ratio": round(self.rr_ratio, 2),
            "spread_cost": round(self.spread_cost, 2),
            **{f"sig_{k}": v for k, v in self.metadata.items()
               if isinstance(v, (int, float, str, bool))},
        }


@dataclass
class OpenPosition:
    """Live position being tracked."""
    id: str
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    bar_idx: int = 0
    current_bar_idx: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    metadata: dict = field(default_factory=dict)


class TradeEngine:
    """
    Unified trade execution engine.

    Works in two modes:
    - Bar mode: on_bar() called per M1 candle — checks SL/TP on H/L
    - Tick mode: on_tick() called per tick — checks SL/TP on bid/ask

    All UIs call the same engine. The engine does NOT evaluate strategies —
    it receives SignalEvents and manages the trade lifecycle.
    """

    def __init__(self, config: TradeConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self._bus = event_bus
        self._open: Optional[OpenPosition] = None
        self._trades: list[TradeRecord] = []
        self._balance_curve: list[float] = [config.initial_balance]
        self._peak_balance: float = config.initial_balance
        self._max_dd: float = 0.0
        self._realized_pnl: float = 0.0
        self._bar_idx: int = 0
        self._last_entry_price: Optional[float] = None
        # --- TICK PRECISION FIXED ARRAYS ---
        self._tick_ts: Optional[np.ndarray] = None
        self._tick_bid: Optional[np.ndarray] = None
        self._tick_ask: Optional[np.ndarray] = None

    @property
    def trades(self) -> list[TradeRecord]:
        return list(self._trades)

    @property
    def open_position(self) -> Optional[OpenPosition]:
        return self._open

    @property
    def balance_curve(self) -> list[float]:
        return list(self._balance_curve)

    @property
    def max_drawdown(self) -> float:
        return self._max_dd

    @property
    def realized_pnl_pips(self) -> float:
        return self._realized_pnl
        
    def attach_ticks(self, ticks: pd.DataFrame) -> None:
            """Optional: give the engine real bid/ask for realistic entry fills.
            Converts custom string formats ('20070102 054151000') into real datetimes."""
            if ticks is None or ticks.empty:
                return

            # 🛡️ Fix the string timestamp format so Python can search it mathematically
            if "timestamp" in ticks.columns:
                ts_series = pd.to_datetime(ticks["timestamp"], format="%Y%m%d %H%M%S%f")
            else:
                ts_series = pd.to_datetime(ticks.index, format="%Y%m%d %H%M%S%f")
            
            # Store as clean, high-speed NumPy datetime arrays
            self._tick_ts = ts_series.values
            self._tick_bid = ticks.iloc[:, 1].values if "bid" not in ticks.columns else ticks["bid"].values
            self._tick_ask = ticks.iloc[:, 0].values if "ask" not in ticks.columns else ticks["ask"].values

    def _real_fill_price(self, direction: str, ts: pd.Timestamp, timeframe: str = "M1") -> Optional[float]:
            """First real tick at/after the candle closes to avoid lookahead bias."""
            if self._tick_ts is None or len(self._tick_ts) == 0:
                return None
            
            # 🛡️ Dynamic Offset: Shift forward based on the timeframe that triggered it
            minutes_to_add = 5 if timeframe == "M5" else 1
            candle_close_time = ts + pd.Timedelta(minutes=minutes_to_add)
        
            target_dtype = self._tick_ts.dtype
            lookup_target = np.datetime64(candle_close_time).astype(target_dtype)
        
            pos = int(np.searchsorted(self._tick_ts, lookup_target, side="left"))
        
            if pos >= len(self._tick_ts):
                return None
            
            return float(self._tick_ask[pos]) if direction == "LONG" else float(self._tick_bid[pos])
        
    def open(self, signal: SignalEvent) -> None:
        """Open a position from a signal event. Applies spread + slippage."""
        if self._open is not None:
            return

        direction = signal.direction
        if direction not in ("LONG", "SHORT"):
            return

        entry = signal.entry_price
        strategy_tf = signal.metadata.get("timeframe", "M1")
        # Look up true execution price if tick histories are available
        real_fill = self._real_fill_price(direction, signal.timestamp, timeframe=strategy_tf)
        if real_fill is not None:
            entry = real_fill

        tp = signal.take_profit
        sl = signal.stop_loss
        
        if not tp or not sl:
            return

        # Sanity
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
        # Dedup
        if check_dedup(self._last_entry_price, entry):
            return

        # Min R:R
        if not check_min_rr(entry, tp, sl, self.config.pip_value, self.config.min_rr):
            return

        pos = OpenPosition(
            id=str(uuid.uuid4())[:8],
            strategy=signal.strategy_name,
            direction=direction,
            entry_time=signal.timestamp,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            bar_idx=self._bar_idx,
            metadata=signal.metadata,
        )
        self._open = pos
        self._last_entry_price = entry

        if self._bus:
            self._bus.publish(TradeEvent(
                action="OPEN", price=entry, timestamp=signal.timestamp,
                symbol=self.config.symbol,
                metadata={"position_id": pos.id, "direction": direction},
            ))

    def on_bar(self, bar: BarEvent) -> Optional[TradeRecord]:
        """Process a bar event. Checks SL/TP on high/low."""
        self._bar_idx += 1

        if self._open is None:
            self._update_equity(bar.close)
            return None

        pos = self._open
        pos.current_bar_idx = self._bar_idx

        if pos.direction == "LONG":
            adverse = (bar.low - pos.entry_price) / self.config.pip_value
            favorable = (bar.high - pos.entry_price) / self.config.pip_value
        else:
            adverse = (pos.entry_price - bar.high) / self.config.pip_value
            favorable = (pos.entry_price - bar.low) / self.config.pip_value

        pos.mae = min(pos.mae, adverse)
        pos.mfe = max(pos.mfe, favorable)

        closed = self._check_exit(bar.high, bar.low, bar.close, bar.timestamp)
        self._update_equity(bar.close)
        return closed

    def on_tick(self, bid: float, ask: float, timestamp: pd.Timestamp) -> Optional[TradeRecord]:
        """Process a tick event. Checks SL/TP on bid/ask."""
        if self._open is None:
            return None

        pos = self._open

        if pos.direction == "LONG":
            adverse = (bid - pos.entry_price) / self.config.pip_value
            favorable = (bid - pos.entry_price) / self.config.pip_value
            if bid <= pos.stop_loss:
                return self._close(pos.stop_loss, timestamp, "SL")
            elif bid >= pos.take_profit:
                return self._close(pos.take_profit, timestamp, "TP")
        else:
            adverse = (pos.entry_price - ask) / self.config.pip_value
            favorable = (pos.entry_price - ask) / self.config.pip_value
            if ask >= pos.stop_loss:
                return self._close(pos.stop_loss, timestamp, "SL")
            elif ask <= pos.take_profit:
                return self._close(pos.take_profit, timestamp, "TP")

        pos.mae = min(pos.mae, adverse)
        pos.mfe = max(pos.mfe, favorable)
        return None

    def force_close(self, price: float, timestamp: pd.Timestamp, reason: str = "EOD") -> Optional[TradeRecord]:
        """Force-close the open position."""
        if self._open is None:
            return None
        return self._close(price, timestamp, reason)

    def get_stats(self) -> dict:
        """Compute standard stats from trade history."""
        from core.engine import build_result
        return build_result("ENGINE", self._trades, self._max_dd, self._balance_curve)

    def reset(self) -> None:
        """Clear all state for a new backtest."""
        self._open = None
        self._trades.clear()
        self._balance_curve = [self.config.initial_balance]
        self._peak_balance = self.config.initial_balance
        self._max_dd = 0.0
        self._realized_pnl = 0.0
        self._bar_idx = 0
        self._last_entry_price = None

    def _check_exit(self, high: float, low: float, close: float, timestamp: pd.Timestamp) -> Optional[TradeRecord]:
        pos = self._open
        if pos is None:
            return None
        if pos.direction == "LONG":
            if low <= pos.stop_loss:
                return self._close(pos.stop_loss, timestamp, "SL")
            elif high >= pos.take_profit:
                return self._close(pos.take_profit, timestamp, "TP")
        else:
            if high >= pos.stop_loss:
                return self._close(pos.stop_loss, timestamp, "SL")
            elif low <= pos.take_profit:
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
            mfe_pips=round(pos.mfe, 2), bars_held=self._bar_idx - pos.bar_idx,
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
