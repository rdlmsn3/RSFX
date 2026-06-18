"""
ui/cli.py
---------
Unified CLI for RSFX backtesting — EventBus-driven tick loop.

Uses CandleStream + EventBus + SignalEngine + TradeEngine.
Only accepts tick data (bid/ask/volume).

Usage:
    /usr/bin/python3 ui/cli.py -s tweezer_reversal --csv data/ticks_EURUSD.csv
    /usr/bin/python3 ui/cli.py -s tweezer_reversal,cci_ema --threshold 2
    /usr/bin/python3 ui/cli.py --all --csv data/ticks_EURUSD.csv
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import HistDataAdapter, TickDataAdapter, get_adapter
from core.trade_engine import TradeConfig, TradeEngine
from core.signal_engine import SignalEngine
from core.event_bus import EventBus
from core.candle_stream import IncrementalCandleBuilder, StreamingCandleArrays


def run_single_backtest(
    strategies: list[str],
    args,
    csv_path: Path,
    adapter,
    run_number: Optional[int] = None,
    total_runs: Optional[int] = None,
) -> dict:
    """Run a single backtest with EventBus-driven tick loop."""

    n = len(strategies)

    if run_number is not None and total_runs is not None:
        print(f"\n{'='*70}")
        print(f"  RUN {run_number}/{total_runs} — {args.threshold}-of-{n} on {args.symbol} (tick-driven)")
    else:
        print(f"\n{'='*70}")
        print(f"  RSFX BACKTEST — {args.threshold}-of-{n} on {args.symbol} (tick-driven)")
    if len(strategies) <= 5:
        print(f"  Strategies: {', '.join(strategies)}")
    else:
        print(f"  Strategies: {', '.join(strategies[:5])} ... and {len(strategies) - 5} more")
    print(f"  Lookback: {args.lookback} | Threshold: {args.threshold}")
    print(f"  Spread: {args.spread}p | Min R:R: {args.min_rr}")
    print("=" * 70)

    # --- Validate tick data ---
    if not hasattr(adapter, 'raw_ticks') or adapter.raw_ticks is None:
        raise ValueError(f"No tick data (bid/ask) in {csv_path}. Only tick files supported.")
    raw_ticks = adapter.raw_ticks
    print(f"\n  {len(raw_ticks):,} ticks  {raw_ticks.index[0]} → {raw_ticks.index[-1]}")

    # --- Wire up EventBus ---
    bus = EventBus()

    config = TradeConfig(
        symbol=args.symbol,
        pip_value=0.01,
        lot_size=args.lot_size,
        initial_balance=args.balance,
        spread_pips=args.spread,
        min_rr=args.min_rr,
    )
    trade_engine = TradeEngine(config, event_bus=bus)
    signal_engine = SignalEngine(
        strategy_names=strategies,
        lookback=args.lookback,
        threshold=args.threshold,
        event_bus=bus,
    )

    # --- Determine needed timeframes from strategy registry ---
    from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry
    _populate_registry()
    needed_tfs: set[str] = set()
    for name in strategies:
        needed_tfs.update(STRATEGY_REGISTRY[name]["timeframes"])
    needed_tfs.discard("M1")

    # --- Streaming arrays + EventBus-aware builders ---
    m1_arrays = StreamingCandleArrays()
    tf_arrays: dict[str, StreamingCandleArrays] = {tf: StreamingCandleArrays() for tf in needed_tfs}

    m1_builder = IncrementalCandleBuilder("M1", event_bus=bus, symbol=args.symbol)
    tf_builders = {
        tf: IncrementalCandleBuilder(tf, event_bus=bus, symbol=args.symbol)
        for tf in needed_tfs
    }

    # Bind arrays to SignalEngine for EventBus-driven evaluation
    signal_engine.attach_arrays(m1_arrays, tf_arrays)

    # Precompute indicators for performance (refresh every 500 candles)
    signal_engine.precompute(m1_arrays, tf_arrays)

    # --- Tick loop: 2 lines of core logic ---
    candles_seen = 0
    t0 = time.perf_counter()

    for ts, row in raw_ticks.iterrows():
        bid, ask, vol = float(row["bid"]), float(row["ask"]), float(row.get("volume", 0.0))

        # 1) Manage positions — direct call, every tick
        trade_engine.on_tick(bid, ask, ts)

        # 2) Feed candle builders — BarEvent → EventBus → SignalEngine → TradeEngine
        m1_builder.ingest_tick(ts, bid, ask, vol)
        for builder in tf_builders.values():
            builder.ingest_tick(ts, bid, ask, vol)

        candles_seen += 1

        # Refresh precompute every 500 candles for performance
        if candles_seen % 500 == 0:
            signal_engine.precompute(m1_arrays, tf_arrays)

        # Progress
        if candles_seen % 100000 == 0:
            elapsed = time.perf_counter() - t0
            pct = candles_seen / len(raw_ticks) * 100
            print(f"  {pct:.0f}% | tick {candles_seen:,}/{len(raw_ticks):,} | "
                  f"bars={m1_arrays.n} | trades={len(trade_engine.trades)} | {elapsed:.1f}s")

    # Force close any open position at end
    if trade_engine.open_position:
        last_bid = float(raw_ticks.iloc[-1]["bid"])
        last_ask = float(raw_ticks.iloc[-1]["ask"])
        last_price = last_bid if trade_engine.open_position.direction == "LONG" else last_ask
        trade_engine.force_close(last_price, raw_ticks.index[-1], "EOD")

    elapsed = time.perf_counter() - t0

    # --- Get stats ---
    stats = trade_engine.get_stats()

    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Trades:        {stats['total_trades']}")
    print(f"  Win rate:      {stats['win_rate']:.1f}%")
    print(f"  Total PnL:     {stats['total_pnl_pips']:+.1f} pips")
    print(f"  Avg PnL:       {stats['avg_pnl_pips']:+.1f} pips/trade")
    print(f"  Expectancy:    {stats['expectancy_pips']:+.1f} pips")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Max drawdown:  {stats['max_drawdown_pct']:.1f}%")
    print(f"  Avg MAE:       {stats['avg_mae_pips']:+.1f} pips")
    print(f"  Avg MFE:       {stats['avg_mfe_pips']:+.1f} pips")
    print(f"  Bars built:    {m1_arrays.n}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"{'='*70}")

    # Save to SQLite if not disabled
    if not args.no_save and trade_engine.trades:
        try:
            from core.trade_store import init_db, save_trades
            db_path = Path(__file__).parent.parent / "results" / "trades.db"
            conn = init_db(db_path)
            run_meta = {
                "data_file": str(csv_path),
                "symbol": args.symbol,
                "strategies": strategies,
                "lookback": args.lookback,
                "threshold": args.threshold,
                "n_strategies": n,
                "all_strategies": args.all,
                "run_number": run_number,
                "total_runs": total_runs,
            }
            run_id = save_trades(conn, trade_engine.trades, run_meta, stats)
            conn.close()
            print(f"\n  SQLite saved → {db_path} (run #{run_id})")
        except Exception as exc:
            print(f"\n  ⚠ SQLite save failed: {exc}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="RSFX unified backtest CLI (tick-driven)")
    parser.add_argument("-s", "--strategies", default=None,
                        help="Comma-separated strategy names (ignored if --all is used)")
    parser.add_argument("--all", action="store_true",
                        help="Run ALL strategies individually (one backtest per strategy)")
    parser.add_argument("--csv", default=None,
                        help="Path to tick CSV or Parquet data file (bid/ask/volume required)")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--lookback", "-l", type=int, default=5,
                        help="Confluence lookback window (default: 5)")
    parser.add_argument("--threshold", "-t", type=int, default=1,
                        help="Min strategies agreeing (default: 1 for single strategy)")
    parser.add_argument("--spread", type=float, default=0.5,
                        help="Round-trip spread in pips (default: 0.5)")
    parser.add_argument("--min-rr", type=float, default=0.0,
                        help="Minimum risk:reward ratio (default: 0.0 = no filter)")
    parser.add_argument("--lot-size", type=float, default=0.01)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to SQLite")
    args = parser.parse_args()

    # Determine what to run
    if args.all:
        from detectors.strategies.registry import STRATEGY_REGISTRY
        all_strategies = sorted(STRATEGY_REGISTRY.keys())
        print(f"\n{'='*70}")
        print(f"  RSFX BATCH BACKTEST — ALL {len(all_strategies)} STRATEGIES (tick-driven)")
        print(f"  Running each strategy individually with threshold={args.threshold}")
        print(f"{'='*70}")
        strategies_list = [[s] for s in all_strategies]
    elif args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
        strategies_list = [strategies]
    else:
        parser.error("Either --strategies or --all must be specified")

    # Load tick data
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = Path(__file__).parent.parent / "data" / "DAT_ASCII_USDJPY_M1_202605.csv"

    print(f"\nLoading {csv_path}...")
    _ext = csv_path.suffix.lower()
    if _ext in (".parquet", ".pq", ".parq"):
        adapter = get_adapter(str(csv_path))
        m1_df = adapter.load(str(csv_path))
    else:
        # Detect tick vs bar: read first line, check column count and content
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()
        first_cells = first_line.split(",")

        # Tick format: header with "bid"/"ask" → TickDataAdapter
        # Headerless 4-column (datetime,bid,ask,vol) → HistDataAdapter (handles it natively)
        # Bar data → HistDataAdapter
        if "bid" in first_line.lower() and "ask" in first_line.lower():
            adapter = TickDataAdapter()
            m1_df = adapter.load(str(csv_path))
        else:
            adapter = HistDataAdapter()
            m1_df = adapter.load(str(csv_path))

    # Validate tick data
    if not hasattr(adapter, 'raw_ticks') or adapter.raw_ticks is None:
        print(f"  ⚠ No tick data available in {csv_path}. Only tick files supported.")
        sys.exit(1)

    print(f"  {len(adapter.raw_ticks):,} ticks loaded")

    # Run all backtests
    all_stats = []
    total_runs = len(strategies_list)

    for idx, strategies in enumerate(strategies_list, 1):
        if args.all:
            stats = run_single_backtest(
                strategies=strategies, args=args, csv_path=csv_path,
                adapter=adapter, run_number=idx, total_runs=total_runs,
            )
        else:
            stats = run_single_backtest(
                strategies=strategies, args=args, csv_path=csv_path,
                adapter=adapter,
            )
        all_stats.append((strategies, stats))

    # Print summary if multiple runs
    if len(all_stats) > 1:
        print(f"\n{'='*70}")
        print(f"  BATCH SUMMARY — {len(all_stats)} STRATEGIES")
        print(f"{'='*70}")
        print(f"  {'Strategy':<30} {'Trades':>8} {'Win%':>8} {'P/L':>12} {'PF':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")

        sorted_stats = sorted(all_stats, key=lambda x: x[1]['total_pnl_pips'], reverse=True)

        for strategies, stats in sorted_stats:
            name = strategies[0] if len(strategies) == 1 else "+".join(strategies[:2])
            if len(strategies) > 2:
                name += f"+{len(strategies)-2}"
            print(f"  {name:<30} {stats['total_trades']:>8} {stats['win_rate']:>7.1f}% "
                  f"{stats['total_pnl_pips']:>+11.1f} {stats['profit_factor']:>7.2f}")

        total_trades = sum(s['total_trades'] for _, s in sorted_stats)
        avg_win_rate = sum(s['win_rate'] for _, s in sorted_stats) / len(sorted_stats)
        avg_pf = sum(s['profit_factor'] for _, s in sorted_stats) / len(sorted_stats)
        best_strat = sorted_stats[0][0][0] if len(sorted_stats[0][0]) == 1 else "+".join(sorted_stats[0][0])
        best_pnl = sorted_stats[0][1]['total_pnl_pips']

        print(f"\n  {'TOTAL TRADES:':<30} {total_trades:>8}")
        print(f"  {'AVG WIN RATE:':<30} {avg_win_rate:>7.1f}%")
        print(f"  {'AVG PROFIT FACTOR:':<30} {avg_pf:>7.2f}")
        print(f"  {'BEST STRATEGY:':<30} {best_strat:>8} ({best_pnl:+.1f} pips)")
        print(f"{'='*70}")

    print()


if __name__ == "__main__":
    main()
