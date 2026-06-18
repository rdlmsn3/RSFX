"""
run_backtest_tick_driven.py
----------------------------
Skeleton for the new tick-driven backtest loop.

This wires together:
    - raw tick stream (DataAdapter.raw_ticks, e.g. from TickDataAdapter)
    - IncrementalCandleBuilder / StreamingCandleArrays (core/candle_stream.py)
    - SignalEngine (unchanged)
    - the refactored TradeEngine (core/trade_engine.py)

NOTE: a few pieces depend on core/engine.py and core/events.py, which
weren't in the files I was given, so I've marked the exact spots that
may need a one-line adjustment to match your real signatures
(`compute_tp_sl`, `SignalEvent` construction, etc.) with `# ADAPT:`.

IMPORTANT PERFORMANCE CAVEAT — read before running at scale:
    `SignalEngine.precompute()` is a batch operation: each strategy's
    `precompute()` is written to vectorize indicators (pandas rolling/
    ewm/etc.) over an array it assumes is already fully known. That
    assumption breaks in a pure streaming model where `arrays` grows
    one bar at a time. Two honest options, not papered over:

      (a) Skip precompute() entirely and let SignalEngine fall back to
          its slow path (`strategy.evaluate(windows, cur_ts)`, rebuilt
          per candle close). Correct immediately, but O(lookback) work
          per candle instead of O(1).
      (b) Re-run precompute() over the growing arrays every N candles
          (e.g. every 500) as a pragmatic middle ground — still
          batch-vectorized, just refreshed periodically instead of once.

    Making precompute() itself incremental (true O(1) rolling state per
    strategy) is a real follow-on project, not a drop-in change — most
    of the 72+ strategy files would need their indicator math rewritten
    to maintain running state instead of slicing a full array. Flagging
    this now so it doesn't get discovered halfway through a 10M-tick run.
"""

from __future__ import annotations
import logging

import pandas as pd

from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays
from core.trade_engine import TradeEngine, TradeConfig
from core.signal_engine import SignalEngine
from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry

logger = logging.getLogger(__name__)


def run_tick_backtest(
    raw_ticks: pd.DataFrame,          # index=timestamp, columns=[bid, ask, volume]
    strategy_names: list[str],
    trade_config: TradeConfig,
    lookback: int = 5,
    threshold: int = 2,
    precompute_refresh_every: int | None = None,  # None = never (slow-path only)
) -> TradeEngine:
    engine = TradeEngine(trade_config)
    signal_engine = SignalEngine(strategy_names, lookback=lookback, threshold=threshold)

    # --- One M1 builder always; one extra builder per higher TF any strategy needs ---
    m1_builder = IncrementalCandleBuilder("M1")
    m1_arrays = StreamingCandleArrays()

    _populate_registry()
    needed_tfs: set[str] = set()
    for name in strategy_names:
        needed_tfs.update(STRATEGY_REGISTRY[name]["timeframes"])
    needed_tfs.discard("M1")

    tf_builders = {tf: IncrementalCandleBuilder(tf) for tf in needed_tfs}
    tf_arrays = {tf: StreamingCandleArrays() for tf in needed_tfs}

    candles_seen = 0

    for ts, row in raw_ticks.iterrows():
        bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))

        # 1) Manage open position / fill queued order — always first.
        engine.on_tick(bid, ask, ts)

        # 2) Feed every higher-timeframe builder (cheap: just bucket math).
        for tf, builder in tf_builders.items():
            bar = builder.ingest_tick(ts, bid, ask, vol)
            if bar:
                tf_arrays[tf].append(bar)

        # 3) M1 boundary -> evaluate strategies, queue any resulting signal.
        bar = m1_builder.ingest_tick(ts, bid, ask, vol)
        if bar is None:
            continue

        m1_arrays.append(bar)
        engine.mark_to_market(bar.close)  # equity curve only, no execution effect
        candles_seen += 1

        if precompute_refresh_every and candles_seen % precompute_refresh_every == 0:
            signal_engine.precompute(m1_arrays, tf_arrays)  # see perf caveat above

        i = m1_arrays.n - 1
        signals = signal_engine.evaluate(i, m1_arrays, tf_arrays)
        for sig in signals:
            engine.queue_order(sig)  # ADAPT: confirm SignalEvent field names match

    # End of stream: flush trailing partial bars (not evaluated — no close yet)
    # and force-close anything still open at the last known price.
    if engine.open_position is not None:
        last_bid, last_ask = float(raw_ticks.iloc[-1]["bid"]), float(raw_ticks.iloc[-1]["ask"])
        last_price = last_bid if engine.open_position.direction == "LONG" else last_ask
        engine.force_close(last_price, raw_ticks.index[-1], reason="EOD")

    return engine
