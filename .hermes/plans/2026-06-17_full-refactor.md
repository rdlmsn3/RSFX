# RSFX Full Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify all three UIs (Backtest Web, CLI, Streamlit) onto a single shared trade execution engine, eliminating duplicated signal/trade/TP/SL/equity logic.

**Architecture:** Event-driven core (`EventBus` + `TradeEngine` + `SignalEngine`) with thin consumer layers. The existing `core/trade_engine.py` already has the right structure — the refactor bridges it to the backtest code. `ConfluenceEngine` becomes a signal orchestrator that feeds the shared `TradeEngine`.

**Tech Stack:** Python 3.12, pandas, numpy, FastAPI, Streamlit, SQLite

---

## Current State (what's broken)

```
backtester.py        → own Trade class, own find_exit(), own equity, own stats
confluence.py        → own _compute_tp_sl(), own equity, own stats (duplicated)
app.py (Streamlit)   → own playback, NO trade engine, NO tick execution
backtest/ui/server.py → calls ConfluenceEngine, duplicated stats computation

core/trade_engine.py → EXISTS but unused by backtest code
core/event_bus.py    → EXISTS but unused by backtest code
core/events.py       → EXISTS but unused by backtest code
```

## Target Architecture

```
RSFX/
├── core/                          # SHARED ENGINE LAYER
│   ├── data_loader.py             # CSV/Parquet adapters          ✓ keep
│   ├── market_data_store.py       # Multi-TF OHLCV store          ✓ keep
│   ├── event_bus.py               # Pub/sub event broker          ✓ keep
│   ├── events.py                  # Event dataclasses             ✓ extend
│   ├── trade_engine.py            # Trade lifecycle (open/close/modify)  ✓ rewrite
│   ├── signal_engine.py           # Strategy eval + confluence buffer    NEW
│   ├── engine.py                  # Trading math (TP/SL, PnL, stats)    ✓ keep
│   ├── trade_store.py             # SQLite persistence            ✓ move here
│   ├── tick_candle_builder.py     # Tick → M1 aggregation         ✓ keep
│   └── playback_controller.py     # Replay cursor                 ✓ keep
│
├── ui/                            # CONSUMER LAYERS (thin, no logic)
│   ├── cli.py                     # CLI wrapper                   NEW
│   ├── backtest/                  # FastAPI web UI
│   │   ├── server.py              # HTTP → engine → JSON          rewrite
│   │   └── index.html             # Frontend                      keep
│   └── streamlit_app/             # Streamlit replay + live
│       └── app.py                 # Streamlit → engine            rewrite
│
├── backtest/                      # ANALYSIS TOOLS (unchanged)
│   ├── correlation.py             # Strategy correlation          keep
│   ├── portfolio.py               # Portfolio optimizer           keep
│   └── buckets.py                 # Named strategy buckets        keep
│
└── detectors/                     # Strategy definitions          ✓ keep all
    └── strategies/
```

---

## Phase 1: Core Engine Unification

### Task 1.1: Extend events.py with new event types

**Files:**
- Modify: `core/events.py`

- [ ] **Step 1: Add SignalEvent and BarEvent**

```python
# Add after TradeEvent class:

@dataclass
class SignalEvent:
    """
    Published by SignalEngine when a strategy fires a signal.
    Consumed by TradeEngine to open/close positions.
    """
    strategy_name: str
    direction: str          # "LONG" | "SHORT"
    entry_price: float
    take_profit: float
    stop_loss: float
    timestamp: pd.Timestamp
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BarEvent:
    """
    Published on every M1 bar during backtest or live.
    TradeEngine uses this to check SL/TP on bar-level.
    """
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    symbol: str = "USDJPY"
    timeframe: str = "M1"
```

- [ ] **Step 2: Verify imports work**

Run: `/usr/bin/python3 -c "from core.events import SignalEvent, BarEvent; print('✓ OK')"`

- [ ] **Step 3: Commit**

```bash
git add core/events.py
git commit -m "feat: add SignalEvent and BarEvent to core events"
```

---

### Task 1.2: Rewrite core/trade_engine.py

**Files:**
- Rewrite: `core/trade_engine.py`

This is the **core piece**. The existing TradeEngine uses BUY/SELL and is event-driven. Rewrite it to:
- Support both LONG/SHORT (matching strategy signals) and BUY/SELL (matching live broker)
- Track equity curve, max drawdown
- Apply spread/slippage costs
- Work in both backtest (bar-by-bar) and live (tick-by-tick) mode

- [ ] **Step 1: Write the new TradeEngine**

```python
"""
core/trade_engine.py
--------------------
Unified trade execution engine.

Works identically in backtest replay and live mode.
All three UIs (CLI, Backtest Web, Streamlit) consume this.

Usage:
    engine = TradeEngine(config)
    engine.on_bar(bar_event)       # bar-level backtest
    engine.on_tick(tick_event)     # tick-level backtest or live
    engine.open(signal_event)      # open position from signal
    stats = engine.get_stats()     # summary statistics
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.event_bus import EventBus
from core.events import (
    SignalEvent, BarEvent, TradeEvent, MarketTickEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TradeConfig:
    """Trade execution parameters."""
    symbol: str = "USDJPY"
    pip_value: float = 0.01        # 0.01 for JPY, 0.0001 for others
    lot_size: float = 0.01
    initial_balance: float = 10_000.0
    spread_pips: float = 0.5       # round-trip spread cost
    slippage_pips: float = 0.0     # per-side slippage
    min_rr: float = 1.0            # minimum risk:reward filter
    use_sr: bool = False           # S/R-aware TP/SL
    max_lookback: int = 100        # for ATR/S/R computation


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

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
    mae_pips: float = 0.0   # max adverse excursion
    mfe_pips: float = 0.0   # max favorable excursion
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
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_time": self.exit_time,
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


# ---------------------------------------------------------------------------
# Open position (stateful, during trade)
# ---------------------------------------------------------------------------

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
    bar_idx: int = 0            # candle index at entry
    tick_idx: int = 0           # tick index at entry (for tick-level)
    current_bar_idx: int = 0    # latest bar seen
    mae: float = 0.0            # running max adverse excursion (pips)
    mfe: float = 0.0            # running max favorable excursion (pips)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trade Engine
# ---------------------------------------------------------------------------

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

        # State
        self._open: Optional[OpenPosition] = None
        self._trades: list[TradeRecord] = []
        self._balance_curve: list[float] = [config.initial_balance]
        self._peak_balance: float = config.initial_balance
        self._max_dd: float = 0.0
        self._realized_pnl: float = 0.0
        self._bar_idx: int = 0
        self._last_entry_price: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def open(self, signal: SignalEvent) -> Optional[TradeRecord]:
        """
        Open a position from a signal event.
        Applies spread + slippage to entry price.
        Returns the trade record if opened, None if filtered.
        """
        if self._open is not None:
            return None  # already in a trade

        direction = signal.direction
        if direction not in ("LONG", "SHORT"):
            return None

        # Apply spread + slippage
        entry = signal.entry_price
        if direction == "LONG":
            entry += (self.config.spread_pips / 2 + self.config.slippage_pips) * self.config.pip_value
        else:
            entry -= (self.config.spread_pips / 2 + self.config.slippage_pips) * self.config.pip_value

        tp = signal.take_profit
        sl = signal.stop_loss
        if not tp or not sl:
            return None

        # Sanity: correct direction
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
        if self._last_entry_price is not None:
            if abs(self._last_entry_price - entry) < 1e-6:
                return None

        # Min R:R filter
        risk = abs(entry - sl) / self.config.pip_value
        reward = abs(tp - entry) / self.config.pip_value
        if risk <= 0:
            return None
        if (reward / risk) < self.config.min_rr:
            return None

        # Create position
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
                action="OPEN",
                price=entry,
                timestamp=signal.timestamp,
                symbol=self.config.symbol,
                metadata={"position_id": pos.id, "direction": direction},
            ))

        logger.debug("Opened %s %s at %.5f (SL=%.5f TP=%.5f)", direction, pos.id, entry, sl, tp)
        return None  # trade is open, not yet closed

    def on_bar(self, bar: BarEvent) -> Optional[TradeRecord]:
        """
        Process a bar event. Checks SL/TP on high/low.
        Returns closed trade record if position was closed, else None.
        """
        self._bar_idx += 1

        if self._open is None:
            self._update_equity(bar.close)
            return None

        pos = self._open
        pos.current_bar_idx = self._bar_idx

        # Update MAE/MFE
        if pos.direction == "LONG":
            adverse = (bar.low - pos.entry_price) / self.config.pip_value
            favorable = (bar.high - pos.entry_price) / self.config.pip_value
        else:
            adverse = (pos.entry_price - bar.high) / self.config.pip_value
            favorable = (pos.entry_price - bar.low) / self.config.pip_value

        pos.mae = min(pos.mae, adverse)   # more negative = worse
        pos.mfe = max(pos.mfe, favorable)

        # Check SL/TP
        closed = self._check_exit(bar.high, bar.low, bar.close, bar.timestamp)
        self._update_equity(bar.close)
        return closed

    def on_tick(self, bid: float, ask: float, timestamp: pd.Timestamp) -> Optional[TradeRecord]:
        """
        Process a tick event. Checks SL/TP on bid/ask.
        Returns closed trade record if position was closed, else None.
        """
        if self._open is None:
            return None

        pos = self._open
        mid = (bid + ask) / 2.0

        # Update MAE/MFE
        if pos.direction == "LONG":
            adverse = (bid - pos.entry_price) / self.config.pip_value
            favorable = (bid - pos.entry_price) / self.config.pip_value
        else:
            adverse = (pos.entry_price - ask) / self.config.pip_value
            favorable = (pos.entry_price - ask) / self.config.pip_value

        pos.mae = min(pos.mae, adverse)
        pos.mfe = max(pos.mfe, favorable)

        # Check SL/TP on tick
        closed = None
        if pos.direction == "LONG":
            if bid <= pos.stop_loss:
                closed = self._close(pos.stop_loss, timestamp, "SL")
            elif bid >= pos.take_profit:
                closed = self._close(pos.take_profit, timestamp, "TP")
        else:
            if ask >= pos.stop_loss:
                closed = self._close(pos.stop_loss, timestamp, "SL")
            elif ask <= pos.take_profit:
                closed = self._close(pos.take_profit, timestamp, "TP")

        return closed

    def force_close(self, price: float, timestamp: pd.Timestamp, reason: str = "MANUAL") -> Optional[TradeRecord]:
        """Force-close the open position (e.g. end of data, manual close)."""
        if self._open is None:
            return None
        return self._close(price, timestamp, reason)

    def get_stats(self) -> dict:
        """Compute standard stats from trade history."""
        from backtest.engine import build_result
        return build_result(
            name="ENGINE",
            trades=[t for t in self._trades],
            max_dd=self._max_dd,
            balance_curve=self._balance_curve,
        )

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_exit(self, high: float, low: float, close: float, timestamp: pd.Timestamp) -> Optional[TradeRecord]:
        """Check bar high/low against SL/TP."""
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
        """Close the open position and create a trade record."""
        pos = self._open
        assert pos is not None

        # PnL
        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) / self.config.pip_value
        else:
            pnl = (pos.entry_price - exit_price) / self.config.pip_value

        # Spread cost (round-trip already applied at entry, but track it)
        spread_cost = self.config.spread_pips

        # Final PnL minus spread
        net_pnl = pnl - spread_cost

        risk = abs(pos.entry_price - pos.stop_loss) / self.config.pip_value
        reward = abs(pos.take_profit - pos.entry_price) / self.config.pip_value

        trade = TradeRecord(
            id=pos.id,
            strategy=pos.strategy,
            direction=pos.direction,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            exit_time=timestamp,
            exit_price=exit_price,
            exit_reason=reason,
            pnl_pips=round(net_pnl, 2),
            mae_pips=round(pos.mae, 2),
            mfe_pips=round(pos.mfe, 2),
            bars_held=self._bar_idx - pos.bar_idx,
            risk_pips=round(risk, 2),
            reward_pips=round(reward, 2),
            rr_ratio=round(reward / risk, 2) if risk > 0 else 0.0,
            spread_cost=round(spread_cost, 2),
            metadata=pos.metadata,
        )

        self._trades.append(trade)
        self._realized_pnl += net_pnl
        self._open = None

        if self._bus:
            self._bus.publish(TradeEvent(
                action="CLOSE",
                price=exit_price,
                timestamp=timestamp,
                symbol=self.config.symbol,
                metadata={"position_id": pos.id, "pnl": net_pnl, "reason": reason},
            ))

        return trade

    def _update_equity(self, current_close: float) -> None:
        """Update balance curve and max drawdown."""
        unreal = 0.0
        if self._open is not None:
            if self._open.direction == "LONG":
                unreal = (current_close - self._open.entry_price) / self.config.pip_value
            else:
                unreal = (self._open.entry_price - current_close) / self.config.pip_value

        bal = self.config.initial_balance + (
            self._realized_pnl + unreal
        ) * self.config.lot_size * 100_000 * self.config.pip_value

        self._balance_curve.append(bal)
        self._peak_balance = max(self._peak_balance, bal)
        if self._peak_balance > 0:
            self._max_dd = max(self._max_dd, (self._peak_balance - bal) / self._peak_balance * 100)
```

- [ ] **Step 2: Verify syntax**

Run: `/usr/bin/python3 -m py_compile core/trade_engine.py && echo "✓ OK"`

- [ ] **Step 3: Commit**

```bash
git add core/trade_engine.py core/events.py
git commit -m "feat: rewrite TradeEngine as unified bar/tick trade executor"
```

---

### Task 1.3: Create core/signal_engine.py

**Files:**
- Create: `core/signal_engine.py`

This extracts the strategy evaluation + confluence buffer from `confluence.py` into a reusable component. Both CLI and web UI will use this.

- [ ] **Step 1: Write signal_engine.py**

```python
"""
core/signal_engine.py
---------------------
Strategy evaluation + signal-buffer confluence.

Reusable by all backtest modes. Evaluates strategies against
CandleArrays and returns SignalEvents.

Usage:
    engine = SignalEngine(strategies, config)
    signals = engine.evaluate(i, arrays, tf_arrays)
    # Each signal is a SignalEvent ready for TradeEngine.open()
"""

from __future__ import annotations
import logging
from typing import Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.events import SignalEvent
from core.trade_engine import TradeConfig
from detectors.strategies.base import BaseStrategy
from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry

logger = logging.getLogger(__name__)


@dataclass
class BufferedSignal:
    """A signal active in the confluence buffer."""
    strategy_name: str
    direction: str
    candle_idx: int
    entry_price: float
    take_profit: float
    stop_loss: float
    signal: Any  # PatternSignal


class SignalBuffer:
    """Rolling buffer for signal-buffer confluence."""

    def __init__(self, lookback: int, threshold: int):
        self._lookback = lookback
        self._threshold = threshold
        self._buffer: list[BufferedSignal] = []

    def add_and_check(
        self,
        strategy_name: str,
        direction: str,
        candle_idx: int,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        signal: Any,
    ) -> Optional[tuple[str, list[BufferedSignal]]]:
        """Add signal and check for confluence. Returns (direction, signals) if triggered."""
        new_sig = BufferedSignal(
            strategy_name=strategy_name,
            direction=direction,
            candle_idx=candle_idx,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            signal=signal,
        )

        self._expire(candle_idx)

        agreeing = [
            s for s in self._buffer
            if s.direction == direction and s.strategy_name != strategy_name
        ]
        all_agreeing = agreeing + [new_sig]
        self._buffer.append(new_sig)

        unique = set(s.strategy_name for s in all_agreeing)
        if len(unique) >= self._threshold:
            return direction, all_agreeing
        return None

    def _expire(self, current_idx: int) -> None:
        cutoff = current_idx - self._lookback
        self._buffer = [s for s in self._buffer if s.candle_idx >= cutoff]

    def clear(self) -> None:
        self._buffer.clear()


class SignalEngine:
    """
    Evaluates strategies and manages confluence buffer.
    Returns SignalEvents ready for TradeEngine.open().
    """

    def __init__(
        self,
        strategy_names: list[str],
        lookback: int = 5,
        threshold: int = 2,
        max_lookback: int = 100,
    ) -> None:
        _populate_registry()

        self._names = strategy_names
        self._lookback = lookback
        self._threshold = threshold
        self._max_lookback = max_lookback

        # Instantiate strategies
        self._strategies: dict[str, BaseStrategy] = {}
        self._required_tfs: dict[str, list[str]] = {}
        for name in strategy_names:
            if name not in STRATEGY_REGISTRY:
                raise ValueError(f"Strategy '{name}' not found.")
            info = STRATEGY_REGISTRY[name]
            self._strategies[name] = info["class"]()
            self._required_tfs[name] = info["timeframes"]

        # Confluence buffer
        self._buffer = SignalBuffer(lookback, threshold)

        # Precomputed indicators
        self._precomputed: dict[str, dict] = {}

    def precompute(self, arrays, tf_arrays: dict) -> None:
        """Pre-compute indicators for all strategies."""
        for name, strategy in self._strategies.items():
            if hasattr(strategy, "precompute") and callable(strategy.precompute):
                try:
                    self._precomputed[name] = strategy.precompute(arrays, tf_arrays) or {}
                except Exception as exc:
                    logger.warning("%s.precompute() failed: %s", name, exc)
                    self._precomputed[name] = {}
            else:
                self._precomputed[name] = {}

    def evaluate(self, i: int, arrays, tf_arrays: dict) -> list[SignalEvent]:
        """
        Evaluate all strategies at candle index i.
        Returns list of SignalEvent if confluence triggered, else empty.
        """
        signals = []

        for name, strategy in self._strategies.items():
            try:
                if self._precomputed.get(name):
                    sigs = strategy.evaluate_fast(i, arrays, self._precomputed[name]) or []
                else:
                    # Fallback: rebuild window DataFrame
                    win_start = max(0, i - self._max_lookback)
                    ts_window = arrays.timestamps[win_start:i+1]
                    window_df = pd.DataFrame({
                        "open": arrays.opens[win_start:i+1],
                        "high": arrays.highs[win_start:i+1],
                        "low": arrays.lows[win_start:i+1],
                        "close": arrays.closes[win_start:i+1],
                        "volume": arrays.volumes[win_start:i+1],
                    }, index=pd.DatetimeIndex(ts_window))
                    windows = {"M1": window_df}
                    for tf in self._required_tfs.get(name, []):
                        if tf == "M1":
                            continue
                        if tf in tf_arrays:
                            tfa = tf_arrays[tf]
                            ts_cur = arrays.timestamps[i]
                            pos = int(np.searchsorted(tfa.timestamps, ts_cur, side="right"))
                            ws = max(0, pos - self._max_lookback)
                            if pos > 0:
                                windows[tf] = pd.DataFrame({
                                    "open": tfa.opens[ws:pos],
                                    "high": tfa.highs[ws:pos],
                                    "low": tfa.lows[ws:pos],
                                    "close": tfa.closes[ws:pos],
                                    "volume": tfa.volumes[ws:pos],
                                }, index=pd.DatetimeIndex(tfa.timestamps[ws:pos]))
                    cur_ts = pd.Timestamp(arrays.timestamps[i])
                    sigs = strategy.evaluate(windows, cur_ts) or []

                if not sigs:
                    continue

                sig = sigs[0]
                direction = sig.metadata.get("direction", "")
                if direction not in ("LONG", "SHORT"):
                    continue

                entry_price = sig.metadata.get("entry_price", float(arrays.closes[i]))
                tp = sig.metadata.get("take_profit", 0.0)
                sl = sig.metadata.get("stop_loss", 0.0)

                # Check confluence
                result = self._buffer.add_and_check(
                    strategy_name=name,
                    direction=direction,
                    candle_idx=i,
                    entry_price=entry_price,
                    take_profit=tp,
                    stop_loss=sl,
                    signal=sig,
                )

                if result is None:
                    continue

                conf_direction, agreeing = result
                trigger = agreeing[-1]

                signal_event = SignalEvent(
                    strategy_name="+".join(sorted(set(s.strategy_name for s in agreeing))),
                    direction=conf_direction,
                    entry_price=trigger.entry_price,
                    take_profit=trigger.take_profit,
                    stop_loss=trigger.stop_loss,
                    timestamp=pd.Timestamp(arrays.timestamps[i]),
                    metadata={
                        "lookback": self._lookback,
                        "threshold": self._threshold,
                        "agreeing": ",".join(sorted(set(s.strategy_name for s in agreeing))),
                        "trigger_strategy": trigger.strategy_name,
                    },
                )
                signals.append(signal_event)

            except Exception as exc:
                logger.debug("%s evaluate error at i=%d: %s", name, i, exc)
                continue

        return signals

    def reset_buffer(self) -> None:
        self._buffer.clear()
```

- [ ] **Step 2: Verify syntax**

Run: `/usr/bin/python3 -m py_compile core/signal_engine.py && echo "✓ OK"`

- [ ] **Step 3: Commit**

```bash
git add core/signal_engine.py
git commit -m "feat: add SignalEngine — strategy eval + confluence buffer"
```

---

### Task 1.4: Move trade_store.py to core/

**Files:**
- Move: `backtest/trade_store.py` → `core/trade_store.py`
- Modify: `backtest/backtester.py` (update import)
- Modify: `backtest/confluence.py` (update import)

- [ ] **Step 1: Move the file**

```bash
mv backtest/trade_store.py core/trade_store.py
```

- [ ] **Step 2: Update imports in backtester.py**

Change `from backtest.trade_store import` → `from core.trade_store import`

- [ ] **Step 3: Update imports in confluence.py**

Change `from backtest.trade_store import` → `from core.trade_store import`

- [ ] **Step 4: Verify**

Run: `/usr/bin/python3 -m py_compile core/trade_store.py && /usr/bin/python3 -m py_compile backtest/backtester.py && /usr/bin/python3 -m py_compile backtest/confluence.py && echo "✓ OK"`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move trade_store.py to core/"
```

---

## Phase 2: Consumer Layer Refactoring

### Task 2.1: Create ui/cli.py — thin CLI wrapper

**Files:**
- Create: `ui/cli.py`

A minimal CLI that uses `SignalEngine` + `TradeEngine` instead of the old Backtester class.

- [ ] **Step 1: Write ui/cli.py**

(This will be ~100 lines — argparse + data load + SignalEngine + TradeEngine loop + print results)

- [ ] **Step 2: Verify it runs**

Run: `/usr/bin/python3 ui/cli.py --help`

- [ ] **Step 3: Commit**

```bash
git add ui/cli.py
git commit -m "feat: add unified CLI using SignalEngine + TradeEngine"
```

---

### Task 2.2: Rewrite backtest/ui/server.py to use SignalEngine + TradeEngine

**Files:**
- Rewrite: `backtest/ui/server.py`

Replace ConfluenceEngine usage with SignalEngine + TradeEngine.

- [ ] **Step 1: Rewrite server.py**

The `/run` endpoint becomes:
1. Load data → MarketDataStore → CandleArrays
2. Create SignalEngine(strategies, lookback, threshold)
3. Create TradeEngine(config)
4. Loop: signals = signal_engine.evaluate(i, arrays, tf_arrays)
5. For each signal: trade_engine.open(signal)
6. For each bar: trade_engine.on_bar(bar)
7. Return trade_engine.get_stats()

- [ ] **Step 2: Verify API works**

Start server, hit `/run` endpoint

- [ ] **Step 3: Commit**

```bash
git add backtest/ui/server.py
git commit -m "refactor: backtest UI uses unified SignalEngine + TradeEngine"
```

---

### Task 2.3: Rewrite Streamlit app to use SignalEngine + TradeEngine

**Files:**
- Rewrite: `app.py`

Same engine, different frontend. Replay mode uses the same bar loop.

- [ ] **Step 1: Rewrite app.py**

The Streamlit app becomes:
1. Sidebar: select data file, strategies, lookback, threshold
2. Load data → MarketDataStore → CandleArrays
3. Create SignalEngine + TradeEngine
4. Playback loop: for each bar, evaluate signals, open trades, update chart
5. Display: equity curve, open trades, trade history, stats

- [ ] **Step 2: Verify Streamlit runs**

Run: `streamlit run app.py`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "refactor: Streamlit app uses unified SignalEngine + TradeEngine"
```

---

## Phase 3: Cleanup

### Task 3.1: Delete dead code

**Files:**
- Delete: `backtest/backtester.py` (Backtester class, _run_strategy_job, parallel orchestration)
- Delete: `backtest/confluence.py` (ConfluenceEngine — replaced by SignalEngine + TradeEngine)
- Keep: `backtest/engine.py` (shared math — still used)
- Keep: `backtest/correlation.py`, `backtest/portfolio.py`, `backtest/buckets.py`

- [ ] **Step 1: Verify nothing imports from deleted files**

Run: `grep -r "from backtest.backtester" --include="*.py"` and `grep -r "from backtest.confluence" --include="*.py"`

- [ ] **Step 2: Delete files**

```bash
git rm backtest/backtester.py backtest/confluence.py
```

- [ ] **Step 3: Update __main__.py**

Update `backtest/__main__.py` to use new CLI entry point.

- [ ] **Step 4: Update README.md**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "cleanup: remove old backtester.py and confluence.py"
```

---

### Task 3.2: Final verification

- [ ] **Step 1: Run full compile check**

Run: `/usr/bin/python3 -m py_compile core/trade_engine.py && /usr/bin/python3 -m py_compile core/signal_engine.py && /usr/bin/python3 -m py_compile core/trade_store.py && /usr/bin/python3 -m py_compile ui/cli.py && echo "✓ All core files compile"`

- [ ] **Step 2: Test CLI backtest**

Run: `/usr/bin/python3 ui/cli.py -s tweezer_reversal,cci_ema --csv data/DAT_ASCII_USDJPY_M1_202605.csv`

- [ ] **Step 3: Test backtest UI**

Start server, run backtest via web UI

- [ ] **Step 4: Test Streamlit**

Run: `streamlit run app.py`, load data, run replay

- [ ] **Step 5: Final commit + push**

```bash
git add -A && git commit -m "refactor: complete — all UIs use unified engine" && git push origin main
```
