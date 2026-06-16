"""
core/signal_engine.py
---------------------
Strategy evaluation + signal-buffer confluence.

Reusable by all backtest modes. Evaluates strategies against
CandleArrays and returns SignalEvents.

Usage:
    engine = SignalEngine(["tweezer_reversal", "cci_ema"], lookback=5, threshold=2)
    engine.precompute(arrays, tf_arrays)  # optional fast path
    signals = engine.evaluate(i, arrays, tf_arrays)
    for sig in signals:
        trade_engine.open(sig)
"""

from __future__ import annotations
import logging
from typing import Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.events import SignalEvent
from core.engine import compute_tp_sl
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

    def __len__(self) -> int:
        return len(self._buffer)


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
                raise ValueError(
                    f"Strategy '{name}' not found. "
                    f"Available: {sorted(STRATEGY_REGISTRY.keys())}"
                )
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

                # Use trigger's TP/SL, fall back to ATR if missing
                tp = trigger.take_profit
                sl = trigger.stop_loss
                if not tp or not sl:
                    tp, sl = compute_tp_sl(trigger.signal, arrays, i, lookback=100, use_sr=False)

                signal_event = SignalEvent(
                    strategy_name="+".join(sorted(set(s.strategy_name for s in agreeing))),
                    direction=conf_direction,
                    entry_price=trigger.entry_price,
                    take_profit=tp,
                    stop_loss=sl,
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
