#!/usr/bin/env python3
"""
tick_dump.py — Tick-by-tick data flowing through CandleStream.

Usage:
    python scripts/tick_dump.py --file xxx.parquet --top 1000
    python scripts/tick_dump.py --file xxx.parquet --tail 100
    python scripts/tick_dump.py --file xxx.parquet --search 060327
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from core.data_loader import get_adapter
from core.candle_stream import IncrementalCandleBuilder


def main():
    parser = argparse.ArgumentParser(description="Tick-by-tick dump through CandleStream")
    parser.add_argument("--file", "-f", required=True, help="Path to parquet/csv file")
    parser.add_argument("--top", "-n", type=int, default=20, help="Show first N ticks (default: 20)")
    parser.add_argument("--tail", "-t", type=int, default=0, help="Show last N ticks")
    parser.add_argument("--search", "-s", type=str, default="", help="Search for timestamp containing this string")
    parser.add_argument("--raw", action="store_true", help="Dump raw data only (no CandleStream)")
    parser.add_argument("--info", action="store_true", help="Show only file info")
    parser.add_argument("--tf", type=str, default="M1", help="Timeframe for candle building (default: M1)")
    args = parser.parse_args()

    file_path = args.file
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Loading: {file_path}")
    adapter = get_adapter(file_path)
    adapter.load(file_path)

    raw_ticks = getattr(adapter, "raw_ticks", None)
    if raw_ticks is None or len(raw_ticks) == 0:
        print("No raw ticks found (might be bar data)")
        sys.exit(1)

    total = len(raw_ticks)
    print(f"Total ticks: {total:,}")
    print(f"Time range:  {raw_ticks.index[0]} → {raw_ticks.index[-1]}")
    print(f"Candle TF:   {args.tf}")
    print()

    if args.info:
        return

    # --- Search mode ---
    if args.search:
        query = args.search
        print(f"Searching for '{query}'...")
        try:
            ts_query = pd.Timestamp(query)
            mask = raw_ticks.index == ts_query
        except Exception:
            mask = raw_ticks.index.astype(str).str.contains(query, na=False)

        results = raw_ticks[mask]
        if len(results) == 0:
            print(f"  No exact match for '{query}'")
            try:
                idx = raw_ticks.index
                before = idx[idx < ts_query]
                after = idx[idx > ts_query]
                if len(before) > 0:
                    print(f"  Nearest BEFORE: {before[-1]}")
                if len(after) > 0:
                    print(f"  Nearest AFTER:  {after[0]}")
                if len(before) > 0 and len(after) > 0:
                    gap = (after[0] - before[-1]).total_seconds()
                    print(f"  Gap: {gap:.1f} seconds")
            except Exception:
                pass
        else:
            print(f"  Found {len(results)} tick(s):")
            for ts, row in results.iterrows():
                print(f"    {ts}  bid={row['bid']:.5f}  ask={row['ask']:.5f}")
        return

    # --- Raw mode (no processing) ---
    if args.raw:
        if args.tail > 0:
            df = raw_ticks.tail(args.tail)
            label = f"LAST {args.tail}"
        else:
            df = raw_ticks.head(args.top)
            label = f"FIRST {args.top}"

        print(f"--- {label} RAW TICKS (no processing) ---")
        print(df.to_string())
        return

    # --- Top/Tail mode with CandleStream ---
    if args.tail > 0:
        df = raw_ticks.tail(args.tail)
        label = f"LAST {args.tail}"
    else:
        df = raw_ticks.head(args.top)
        label = f"FIRST {args.top}"

    builder = IncrementalCandleBuilder(timeframe=args.tf)
    candle_count = 0

    print(f"--- {label} TICKS THROUGH {args.tf} CANDLESTREAM ---")
    print(f"{'#':>5}  {'Timestamp':<26}  {'Bid':>10}  {'Ask':>10}  {'Mid':>10}  {'M1 Bucket':<20}  Event")
    print("-" * 105)

    for i, (ts, row) in enumerate(df.iterrows()):
        bid = float(row["bid"])
        ask = float(row["ask"])
        vol = float(row.get("volume", 0.0))
        mid = (bid + ask) / 2.0

        # Get current bucket BEFORE ingesting (to show which bucket this tick belongs to)
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        bucket = ts.floor(builder._freq)

        # Ingest into candle stream
        bar = builder.ingest_tick(ts, bid, ask, vol)

        # Format event
        event = ""
        if bar is not None:
            candle_count += 1
            event = f"🕯️  CLOSE {bar.timestamp} O={bar.open:.5f} H={bar.high:.5f} L={bar.low:.5f} C={bar.close:.5f}"

        print(f"{i+1:>5}  {str(ts):<26}  {bid:>10.5f}  {ask:>10.5f}  {mid:>10.5f}  {str(bucket):<20}  {event}")

    # Flush remaining candle
    final = builder.flush()
    if final:
        candle_count += 1
        print(f"\n  🕯️  FLUSH  {final.timestamp} O={final.open:.5f} H={final.high:.5f} L={final.low:.5f} C={final.close:.5f}")

    print(f"\nTotal candles built: {candle_count}")


if __name__ == "__main__":
    main()
