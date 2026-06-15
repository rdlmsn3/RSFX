"""
confluence_backtest.py
----------------------
Signal-buffer confluence backtester.

Unlike same-candle confluence, this uses a lookback window:
  - When strategy A fires, its signal is stored in a buffer (active for N candles)
  - If strategy B fires within that N-candle window, confluence is triggered
  - Trade entry uses the triggering signal's TP/SL

This is how real multi-timeframe / multi-strategy bots work — signals
don't have to fire on the exact same candle, just within a time window.

Usage:
    python3 confluence_backtest.py -s tweezer_reversal,h1_trend_m5_rsi,cci_ema --lookback 5
    python3 confluence_backtest.py -s h1_trend --lookback 10 --threshold 2
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import HistDataAdapter
from core.market_data_store import MarketDataStore
from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry
from detectors.strategies.base import BaseStrategy
from detectors.signal import PatternSignal

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("confluence")

from backtest import (
    CandleArrays,
    Trade,
    find_exit,
    _build_result,
)


# =========================================================================
# Signal Buffer
# =========================================================================

@dataclass
class BufferedSignal:
    """A signal that was fired by a strategy and is 'active' in the buffer."""
    strategy_name: str
    direction: str        # LONG or SHORT
    candle_idx: int       # which candle it fired on
    entry_price: float    # price at signal time
    take_profit: float
    stop_loss: float
    signal: PatternSignal # original signal object


class SignalBuffer:
    """
    Maintains a rolling buffer of recent signals.

    When a strategy fires:
      1. Add it to the buffer
      2. Check buffer for OTHER strategies that fired within lookback candles
         and agree on direction
      3. If count >= threshold → confluence triggered
      4. Expire signals older than lookback candles
    """

    def __init__(self, lookback: int, threshold: int, pip_value: float):
        self._lookback = lookback
        self._threshold = threshold
        self._pip_value = pip_value
        self._buffer: list[BufferedSignal] = []

    def add_and_check(
        self,
        strategy_name: str,
        direction: str,
        candle_idx: int,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        signal: PatternSignal,
    ) -> Optional[tuple[str, list[BufferedSignal]]]:
        """
        Add new signal to buffer and check for confluence.

        Returns (direction, [agreeing_signals]) if threshold met, else None.
        The triggering signal is always included in the agreeing list.
        """
        new_sig = BufferedSignal(
            strategy_name=strategy_name,
            direction=direction,
            candle_idx=candle_idx,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            signal=signal,
        )

        # Expire old signals
        self._expire(candle_idx)

        # Check for confluence: count signals in buffer that agree on direction
        # (excluding the new one — we need threshold OTHER strategies)
        agreeing = [
            s for s in self._buffer
            if s.direction == direction and s.strategy_name != strategy_name
        ]

        # Include the new signal in the agreeing list for the final count
        all_agreeing = agreeing + [new_sig]

        # Add to buffer
        self._buffer.append(new_sig)

        # Check if we have enough unique strategies agreeing
        unique_strategies = set(s.strategy_name for s in all_agreeing)
        if len(unique_strategies) >= self._threshold:
            return direction, all_agreeing

        return None

    def _expire(self, current_idx: int) -> None:
        """Remove signals older than lookback candles."""
        cutoff = current_idx - self._lookback
        self._buffer = [s for s in self._buffer if s.candle_idx >= cutoff]

    def get_buffer_state(self) -> list[dict]:
        """Return current buffer state for debugging."""
        return [
            {
                "strategy": s.strategy_name,
                "direction": s.direction,
                "candle_idx": s.candle_idx,
                "age": "current",
            }
            for s in self._buffer
        ]


# =========================================================================
# Confluence Engine (Signal-Buffer)
# =========================================================================

class ConfluenceEngine:
    """
    Signal-buffer confluence backtester.

    Each strategy fires independently. When a signal fires:
      1. Check the buffer for other strategies that fired recently
      2. If enough agree within the lookback window → enter trade
      3. TP/SL from the triggering signal (most recent)
    """

    def __init__(
        self,
        data_store: MarketDataStore,
        strategy_names: list[str],
        symbol: str = "USDJPY",
        lookback: int = 5,
        threshold: int = 2,
        initial_balance: float = 10_000.0,
        lot_size: float = 0.01,
        pip_value: float = 0.01,
        max_lookback: int = 100,
        show_buffer: bool = False,
    ) -> None:
        self._store = data_store
        self._symbol = symbol
        self._lookback = lookback
        self._threshold = threshold
        self._initial_balance = initial_balance
        self._lot_size = lot_size
        self._pip_value = pip_value
        self._max_lookback = max_lookback
        self._show_buffer = show_buffer

        # Load strategies
        _populate_registry()
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

        # Extract arrays
        m1_df = data_store.get_data(symbol, "M1")
        self._arrays = CandleArrays.from_dataframe(m1_df)

        self._tf_arrays: dict[str, CandleArrays] = {}
        for tf in data_store.available_timeframes(symbol):
            if tf == "M1":
                continue
            try:
                self._tf_arrays[tf] = CandleArrays.from_dataframe(
                    data_store.get_data(symbol, tf)
                )
            except Exception:
                pass

    def _precompute_all(self) -> dict[str, dict[str, Any]]:
        """Pre-compute indicators for all strategies."""
        precomputed = {}
        for name, strategy in self._strategies.items():
            if hasattr(strategy, "precompute") and callable(strategy.precompute):
                try:
                    precomputed[name] = strategy.precompute(
                        self._arrays, self._tf_arrays
                    ) or {}
                except Exception as exc:
                    logger.warning("%s.precompute() failed: %s", name, exc)
                    precomputed[name] = {}
            else:
                precomputed[name] = {}
        return precomputed

    def _evaluate_strategy(
        self,
        name: str,
        strategy: BaseStrategy,
        i: int,
        precomputed: dict[str, Any],
    ) -> list[PatternSignal]:
        """Evaluate a single strategy at candle index i."""
        try:
            if precomputed:
                return strategy.evaluate_fast(i, self._arrays, precomputed) or []
            else:
                win_start = max(0, i - self._max_lookback)
                ts_window = self._arrays.timestamps[win_start:i+1]
                window_df = pd.DataFrame(
                    {
                        "open":  self._arrays.opens[win_start:i+1],
                        "high":  self._arrays.highs[win_start:i+1],
                        "low":   self._arrays.lows[win_start:i+1],
                        "close": self._arrays.closes[win_start:i+1],
                        "volume": self._arrays.volumes[win_start:i+1],
                    },
                    index=pd.DatetimeIndex(ts_window),
                )
                windows = {"M1": window_df}
                for tf in self._required_tfs.get(name, []):
                    if tf == "M1":
                        continue
                    if tf in self._tf_arrays:
                        tfa = self._tf_arrays[tf]
                        ts_cur = self._arrays.timestamps[i]
                        pos = int(np.searchsorted(
                            tfa.timestamps, ts_cur, side="right"
                        ))
                        ws = max(0, pos - self._max_lookback)
                        if pos > 0:
                            windows[tf] = pd.DataFrame(
                                {
                                    "open":  tfa.opens[ws:pos],
                                    "high":  tfa.highs[ws:pos],
                                    "low":   tfa.lows[ws:pos],
                                    "close": tfa.closes[ws:pos],
                                    "volume": tfa.volumes[ws:pos],
                                },
                                index=pd.DatetimeIndex(tfa.timestamps[ws:pos]),
                            )
                cur_ts = pd.Timestamp(self._arrays.timestamps[i])
                return strategy.evaluate(windows, cur_ts) or []
        except Exception as exc:
            logger.debug("%s evaluate error at i=%d: %s", name, i, exc)
            return []

    def _compute_tp_sl(
        self,
        signal: PatternSignal,
        i: int,
    ) -> tuple[float, float]:
        """Get TP/SL from signal, with ATR fallback."""
        tp = signal.metadata.get("take_profit", 0.0)
        sl = signal.metadata.get("stop_loss", 0.0)

        if tp and sl:
            entry = float(self._arrays.closes[i])
            # Sanity: ensure correct direction
            if signal.metadata.get("direction") == "LONG":
                if tp <= entry:
                    tp = entry + abs(tp - entry)
                if sl >= entry:
                    sl = entry - abs(sl - entry)
            else:
                if tp >= entry:
                    tp = entry - abs(tp - entry)
                if sl <= entry:
                    sl = entry + abs(sl - entry)
            return tp, sl

        # ATR fallback
        win_start = max(0, i - self._max_lookback)
        _df = pd.DataFrame(
            {"high": self._arrays.highs[win_start:i+1],
             "low":  self._arrays.lows[win_start:i+1],
             "close": self._arrays.closes[win_start:i+1]},
        )
        try:
            BaseStrategy.compute_tp_sl(signal, _df)
            tp = signal.metadata.get("take_profit", 0.0)
            sl = signal.metadata.get("stop_loss", 0.0)
            return tp, sl
        except Exception:
            return 0.0, 0.0

    def run(self) -> list[Trade]:
        """Run the signal-buffer confluence backtest."""
        print(f"\nPre-computing indicators for {len(self._strategies)} strategies...")
        t0 = time.perf_counter()
        precomputed = self._precompute_all()
        strategies_fast = sum(1 for v in precomputed.values() if v)
        print(f"  Done in {time.perf_counter() - t0:.1f}s "
              f"({strategies_fast}/{len(self._strategies)} fast path)")

        # Signal buffer
        buffer = SignalBuffer(
            lookback=self._lookback,
            threshold=self._threshold,
            pip_value=self._pip_value,
        )

        trades: list[Trade] = []
        open_trade: Optional[Trade] = None
        realized_pnl_pips = 0.0
        balance_curve = [self._initial_balance]
        peak_balance = self._initial_balance
        max_dd = 0.0
        last_entry_price: Optional[float] = None

        _dedup_tol = 1e-6
        n_signals_total = 0
        n_confluences = 0
        confluence_log: list[dict] = []

        t_start = time.perf_counter()

        for i in range(self._max_lookback, self._arrays.n):
            # 1. Close resolved trade
            if open_trade is not None and i >= open_trade._exit_idx:
                realized_pnl_pips += open_trade.pnl_pips
                trades.append(open_trade)
                open_trade = None

            # 2. Skip if trade is open
            if open_trade is not None:
                ep = open_trade.entry_price
                close = float(self._arrays.closes[i])
                unreal = (
                    (close - ep) if open_trade.direction == "LONG"
                    else (ep - close)
                ) / self._pip_value
                bal = self._initial_balance + (
                    realized_pnl_pips + unreal
                ) * self._lot_size * 100
                balance_curve.append(bal)
                peak_balance = max(peak_balance, bal)
                max_dd = max(
                    max_dd, (peak_balance - bal) / peak_balance * 100
                )
                continue

            # 3. Evaluate ALL strategies at this candle
            for name, strategy in self._strategies.items():
                signals = self._evaluate_strategy(
                    name, strategy, i, precomputed.get(name, {})
                )
                if not signals:
                    continue

                n_signals_total += 1
                sig = signals[0]
                direction = sig.metadata.get("direction", "")
                if direction not in ("LONG", "SHORT"):
                    continue

                entry_price = sig.metadata.get(
                    "entry_price", float(self._arrays.closes[i])
                )
                tp, sl = self._compute_tp_sl(sig, i)

                # 4. Add to buffer and check for confluence
                result = buffer.add_and_check(
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

                conf_direction, agreeing_signals = result
                agreeing_names = [s.strategy_name for s in agreeing_signals]

                n_confluences += 1

                # 5. Execute trade
                # Use the LATEST (triggering) signal's TP/SL
                trigger = agreeing_signals[-1]
                trade_tp = trigger.take_profit
                trade_sl = trigger.stop_loss
                trade_entry = float(self._arrays.closes[i])

                # Dedup
                if (last_entry_price is not None and
                        abs(last_entry_price - trade_entry) < _dedup_tol):
                    continue

                if not (trade_tp and trade_sl):
                    continue

                risk   = abs(trade_entry - trade_sl)  / self._pip_value
                reward = abs(trade_tp - trade_entry)   / self._pip_value

                if risk <= 0:
                    continue

                exit_idx, reason, exit_price, mae, mfe = find_exit(
                    self._arrays, i, conf_direction,
                    trade_tp, trade_sl, self._pip_value,
                )
                cur_ts = pd.Timestamp(self._arrays.timestamps[i])
                exit_ts = pd.Timestamp(self._arrays.timestamps[exit_idx])

                if conf_direction == "LONG":
                    pnl = (exit_price - trade_entry) / self._pip_value
                else:
                    pnl = (trade_entry - exit_price) / self._pip_value

                # Calculate ages of agreeing signals
                ages = {s.strategy_name: i - s.candle_idx for s in agreeing_signals}

                trade = Trade(
                    strategy=" + ".join(sorted(set(agreeing_names))),
                    direction=conf_direction,
                    entry_time=cur_ts,
                    entry_price=trade_entry,
                    stop_loss=trade_sl,
                    take_profit=trade_tp,
                    exit_time=exit_ts,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl_pips=round(pnl, 2),
                    mae_pips=round(mae, 2),
                    mfe_pips=round(mfe, 2),
                    bars_held=exit_idx - i,
                    risk_pips=round(risk, 2),
                    reward_pips=round(reward, 2),
                    rr_ratio=round(reward / risk, 2) if risk > 0 else 0.0,
                    signal_meta={
                        "lookback": self._lookback,
                        "threshold": self._threshold,
                        "n_strategies": len(self._strategies),
                        "agreeing": ",".join(sorted(set(agreeing_names))),
                        "signal_ages": str(ages),
                        "trigger_strategy": trigger.strategy_name,
                    },
                )
                trade._exit_idx = exit_idx
                open_trade = trade
                last_entry_price = trade_entry

                confluence_log.append({
                    "candle": i,
                    "timestamp": cur_ts,
                    "direction": conf_direction,
                    "agreeing": sorted(set(agreeing_names)),
                    "ages": ages,
                    "tp": trade_tp,
                    "sl": trade_sl,
                })

                # One trade per candle max
                break

            # 6. Equity curve
            close = float(self._arrays.closes[i])
            unreal = 0.0
            if open_trade is not None:
                ep = open_trade.entry_price
                unreal = (
                    (close - ep) if open_trade.direction == "LONG"
                    else (ep - close)
                ) / self._pip_value
            bal = self._initial_balance + (
                realized_pnl_pips + unreal
            ) * self._lot_size * 100
            balance_curve.append(bal)
            peak_balance = max(peak_balance, bal)
            max_dd = max(
                max_dd, (peak_balance - bal) / peak_balance * 100
            )

            # Progress
            if i % 5000 == 0 and i > self._max_lookback:
                elapsed = time.perf_counter() - t_start
                pct = (i - self._max_lookback) / (
                    self._arrays.n - self._max_lookback
                ) * 100
                n_active = len(buffer._buffer)
                print(
                    f"  {pct:.0f}% | candle {i}/{self._arrays.n} | "
                    f"trades={len(trades) + (1 if open_trade else 0)} | "
                    f"confluences={n_confluences} | "
                    f"buffer={n_active} | {elapsed:.1f}s"
                )

        # Close open trade
        if open_trade is not None:
            realized_pnl_pips += open_trade.pnl_pips
            trades.append(open_trade)

        elapsed = time.perf_counter() - t_start
        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Total signals evaluated: {n_signals_total}")
        print(f"  Confluences triggered:   {n_confluences}")
        print(f"  Trades executed:         {len(trades)}")

        # Print confluence log
        if confluence_log:
            self._print_confluence_log(confluence_log)

        return trades

    def _print_confluence_log(self, log: list[dict]) -> None:
        """Print summary of confluence events."""
        print(f"\n{'='*80}")
        print(f"CONFLUENCE LOG — {len(log)} entries")
        print(f"{'='*80}")

        # Strategy participation in confluences
        strategy_counts = Counter()
        for entry in log:
            for name in entry["agreeing"]:
                strategy_counts[name] += 1

        print(f"\n  Strategy participation in confluences:")
        for name, count in strategy_counts.most_common():
            pct = count / len(log) * 100
            print(f"    {name:40s} {count:>5d} ({pct:.1f}%)")

        # Signal age distribution
        all_ages = []
        for entry in log:
            all_ages.extend(entry["ages"].values())
        if all_ages:
            print(f"\n  Signal age distribution (candles):")
            print(f"    Mean: {np.mean(all_ages):.1f}")
            print(f"    Median: {np.median(all_ages):.1f}")
            print(f"    Max: {np.max(all_ages)}")

        # Direction breakdown
        dir_counts = Counter(e["direction"] for e in log)
        print(f"\n  Direction: {dict(dir_counts)}")

        # Show last 5 confluences
        print(f"\n  Last 5 confluences:")
        for entry in log[-5:]:
            print(
                f"    Candle {entry['candle']:>5d} | "
                f"{entry['direction']:5s} | "
                f"Agreed: {', '.join(entry['agreeing'])} | "
                f"Ages: {entry['ages']}"
            )


# =========================================================================
# Results
# =========================================================================

def _print_results(trades: list[Trade]) -> dict:
    """Print and return results dict."""
    result = _build_result("CONFLUENCE", trades, 0.0, [])

    print(f"\n{'='*80}")
    print(f"CONFLUENCE BACKTEST RESULTS (Signal-Buffer)")
    print(f"{'='*80}")
    print(f"  Total trades:    {result['total_trades']}")
    print(f"  Winning trades:  {result['winning_trades']}")
    print(f"  Losing trades:   {result['losing_trades']}")
    print(f"  Win rate:        {result['win_rate']:.1f}%")
    print(f"  Total PnL:       {result['total_pnl_pips']:+.1f} pips")
    print(f"  Avg PnL/trade:   {result['avg_pnl_pips']:+.1f} pips")
    print(f"  Expectancy:      {result['expectancy_pips']:+.1f} pips")
    print(f"  Profit factor:   {result['profit_factor']:.2f}")
    print(f"  Avg MAE:         {result['avg_mae_pips']:+.1f} pips")
    print(f"  Avg MFE:         {result['avg_mfe_pips']:+.1f} pips")

    longs = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]
    long_pnl = sum(t.pnl_pips for t in longs)
    short_pnl = sum(t.pnl_pips for t in shorts)
    long_wr = (
        sum(1 for t in longs if t.pnl_pips > 0) / len(longs) * 100
        if longs else 0
    )
    short_wr = (
        sum(1 for t in shorts if t.pnl_pips > 0) / len(shorts) * 100
        if shorts else 0
    )

    print(f"\n  LONG trades:  {len(longs):>4d} (WR: {long_wr:.1f}%, PnL: {long_pnl:+.1f})")
    print(f"  SHORT trades: {len(shorts):>4d} (WR: {short_wr:.1f}%, PnL: {short_pnl:+.1f})")

    reasons = Counter(t.exit_reason for t in trades)
    print(f"\n  Exit reasons:")
    for reason, count in reasons.most_common():
        print(f"    {reason:10s} {count:>5d} ({count/len(trades)*100:.1f}%)")

    return result


def _save_results(
    trades: list[Trade],
    result: dict,
    strategies: list[str],
    lookback: int,
    threshold: int,
    data_file: str,
    symbol: str,
) -> None:
    """Save results to timestamped CSV."""
    base = Path(__file__).parent
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    filter_tag = "_".join(strategies)
    n = len(strategies)

    # Summary
    fname = f"confluence_buf_{n}of{n}_lb{lookback}_{filter_tag}_{run_ts}.csv"
    meta = {
        "backtest_timestamp": run_ts,
        "data_file": data_file,
        "symbol": symbol,
        "strategies_filter": ",".join(strategies),
        "lookback": lookback,
        "threshold": threshold,
        "n_strategies": n,
    }
    row = {**meta, **{k: result[k] for k in (
        "total_trades", "winning_trades", "losing_trades", "win_rate",
        "total_pnl_pips", "avg_pnl_pips", "expectancy_pips", "profit_factor",
        "gross_profit", "gross_loss", "avg_mae_pips", "avg_mfe_pips",
    )}}
    pd.DataFrame([row]).to_csv(str(base / fname), index=False)
    print(f"\nSummary saved  → {base / fname}")

    # Trades
    fname_t = f"confluence_buf_trades_{n}of{n}_lb{lookback}_{filter_tag}_{run_ts}.csv"
    trade_rows = []
    for t in trades:
        d = t.to_dict()
        d.update(meta)
        trade_rows.append(d)
    pd.DataFrame(trade_rows).sort_values("entry_time").to_csv(
        str(base / fname_t), index=False
    )
    print(f"Trades saved   → {base / fname_t}  ({len(trade_rows):,} rows)")

    # Latest symlinks
    for latest_name, actual in [
        ("confluence_results_latest.csv", base / fname),
        ("confluence_trades_latest.csv", base / fname_t),
    ]:
        latest = base / latest_name
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(actual.name)


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Signal-buffer confluence backtester"
    )
    parser.add_argument(
        "--strategies", "-s",
        type=str, required=True,
        help="Comma-separated strategy names",
    )
    parser.add_argument(
        "--lookback", "-l",
        type=int, default=5,
        help="Candle window for signal matching (default: 5)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int, default=2,
        help="Min strategies that must agree (default: 2)",
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--symbol", type=str, default="USDJPY")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-buffer", action="store_true")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    n = len(strategies)

    print("=" * 80)
    print(f"SIGNAL-BUFFER CONFLUENCE — {args.threshold}-of-{n} "
          f"(lookback: {args.lookback} candles) on {args.symbol} M1")
    print("=" * 80)

    # Load data
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = (
            Path(__file__).parent / "data" / "DAT_ASCII_USDJPY_M1_202605.csv"
        )
    print(f"\nLoading {csv_path}...")
    m1_df = HistDataAdapter().load(str(csv_path))
    print(f"Loaded {len(m1_df):,} M1 candles  "
          f"{m1_df.index[0]} → {m1_df.index[-1]}")

    store = MarketDataStore()
    store.load_symbol(args.symbol, m1_df)

    # Print strategies
    _populate_registry()
    print(f"\nStrategies ({n}):")
    for s in strategies:
        if s in STRATEGY_REGISTRY:
            tfs = STRATEGY_REGISTRY[s]["timeframes"]
            print(f"  ✓ {s:40s} (TF: {', '.join(tfs)})")
        else:
            print(f"  ✗ {s:40s} NOT FOUND")
            sys.exit(1)

    print(f"\nConfig: {args.threshold}-of-{n} must agree within "
          f"{args.lookback} candles")

    # Run
    engine = ConfluenceEngine(
        data_store=store,
        strategy_names=strategies,
        symbol=args.symbol,
        lookback=args.lookback,
        threshold=args.threshold,
        show_buffer=args.show_buffer,
    )

    trades = engine.run()
    result = _print_results(trades)

    if not args.no_save:
        data_file = (
            str(csv_path)
            if not csv_path.is_relative_to(Path(__file__).parent)
            else str(csv_path.relative_to(Path(__file__).parent))
        )
        _save_results(
            trades, result, strategies,
            args.lookback, args.threshold,
            data_file, args.symbol,
        )


if __name__ == "__main__":
    main()
