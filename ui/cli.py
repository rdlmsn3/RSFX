"""
ui/cli.py
---------
Unified CLI for RSFX backtesting.

Uses SignalEngine + TradeEngine — the same engine as the web UI and Streamlit.

Usage:
    # Single strategy (threshold=1)
    /usr/bin/python3 ui/cli.py -s tweezer_reversal --csv data/DAT_ASCII_USDJPY_M1_202605.csv

    # Multi-strategy confluence (threshold=2)
    /usr/bin/python3 ui/cli.py -s tweezer_reversal,cci_ema,h1_trend_m5_rsi --threshold 2 --lookback 5

    # With options
    /usr/bin/python3 ui/cli.py -s tweezer_reversal --spread 0.5 --min-rr 1.5 --symbol USDJPY
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import HistDataAdapter, TickDataAdapter, get_adapter
from core.market_data_store import MarketDataStore
from core.trade_engine import TradeConfig, TradeEngine
from core.signal_engine import SignalEngine
from core.engine import CandleArrays


def main():
    parser = argparse.ArgumentParser(description="RSFX unified backtest CLI")
    parser.add_argument("-s", "--strategies", required=True,
                        help="Comma-separated strategy names")
    parser.add_argument("--csv", default=None,
                        help="Path to CSV or Parquet data file")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--lookback", "-l", type=int, default=5,
                        help="Confluence lookback window (default: 5)")
    parser.add_argument("--threshold", "-t", type=int, default=2,
                        help="Min strategies agreeing (default: 2)")
    parser.add_argument("--spread", type=float, default=0.5,
                        help="Round-trip spread in pips (default: 0.5)")
    parser.add_argument("--min-rr", type=float, default=1.0,
                        help="Minimum risk:reward ratio (default: 1.0)")
    parser.add_argument("--lot-size", type=float, default=0.01)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to SQLite")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    n = len(strategies)

    print("=" * 70)
    print(f"  RSFX BACKTEST — {args.threshold}-of-{n} on {args.symbol} M1")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Lookback: {args.lookback} | Threshold: {args.threshold}")
    print(f"  Spread: {args.spread}p | Min R:R: {args.min_rr}")
    print("=" * 70)

    # Load data
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
        with open(csv_path, "r") as f:
            header = f.readline().strip().lower()
        if "bid" in header and "ask" in header:
            adapter = TickDataAdapter()
            m1_df = adapter.load(str(csv_path))
        else:
            adapter = HistDataAdapter()
            m1_df = adapter.load(str(csv_path))

    raw_ticks = getattr(adapter, "raw_ticks", None)
    print(f"  {len(m1_df):,} M1 candles  {m1_df.index[0]} → {m1_df.index[-1]}")

    # Build store + arrays
    store = MarketDataStore()
    store.load_symbol(args.symbol, m1_df)

    m1_df_store = store.get_data(args.symbol, "M1")
    arrays = CandleArrays.from_dataframe(m1_df_store)

    tf_arrays = {}
    for tf in store.available_timeframes(args.symbol):
        if tf == "M1":
            continue
        try:
            tf_arrays[tf] = CandleArrays.from_dataframe(store.get_data(args.symbol, tf))
        except Exception:
            pass

    # Create engines
    config = TradeConfig(
        symbol=args.symbol,
        pip_value=0.01,
        lot_size=args.lot_size,
        initial_balance=args.balance,
        spread_pips=args.spread,
        min_rr=args.min_rr,
    )

    signal_engine = SignalEngine(
        strategy_names=strategies,
        lookback=args.lookback,
        threshold=args.threshold,
    )

    trade_engine = TradeEngine(config)

    # Pre-compute indicators
    print(f"\nPre-computing indicators for {n} strategies...")
    t0 = time.perf_counter()
    signal_engine.precompute(arrays, tf_arrays)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Run backtest loop
    print(f"\nRunning backtest on {arrays.n} candles...")
    t0 = time.perf_counter()

    max_start = max(args.lookback, 100)

    for i in range(max_start, arrays.n):
        # Evaluate strategies
        signals = signal_engine.evaluate(i, arrays, tf_arrays)
        for sig in signals:
            trade_engine.open(sig)

        # Process bar
        bar = {
            "timestamp": arrays.timestamps[i],
            "open": float(arrays.opens[i]),
            "high": float(arrays.highs[i]),
            "low": float(arrays.lows[i]),
            "close": float(arrays.closes[i]),
            "volume": float(arrays.volumes[i]),
        }

        from core.events import BarEvent
        bar_event = BarEvent(
            timestamp=bar["timestamp"],
            open=bar["open"], high=bar["high"],
            low=bar["low"], close=bar["close"],
            volume=bar["volume"], symbol=args.symbol,
        )
        trade_engine.on_bar(bar_event)

        # Progress
        if i % 10000 == 0 and i > max_start:
            elapsed = time.perf_counter() - t0
            pct = (i - max_start) / (arrays.n - max_start) * 100
            print(f"  {pct:.0f}% | candle {i}/{arrays.n} | "
                  f"trades={len(trade_engine.trades)} | {elapsed:.1f}s")

    # Force close any open position
    if trade_engine.open_position:
        import pandas as pd
        last_ts = pd.Timestamp(arrays.timestamps[arrays.n - 1])
        last_close = float(arrays.closes[arrays.n - 1])
        trade_engine.force_close(last_close, last_ts, "EOD")

    elapsed = time.perf_counter() - t0

    # Print results
    stats = trade_engine.get_stats()
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
    print(f"  Time:          {elapsed:.1f}s")
    print(f"{'='*70}")

    # Save to SQLite
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
            }
            run_id = save_trades(conn, trade_engine.trades, run_meta, stats)
            conn.close()
            print(f"\n  SQLite saved → {db_path} (run #{run_id})")
        except Exception as exc:
            print(f"\n  ⚠ SQLite save failed: {exc}")

    print()


if __name__ == "__main__":
    main()
