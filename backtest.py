"""
backtest.py
-----------
Backtest all registered strategies against historical M1 data.

Usage:
    python3 backtest.py

Output:
    - Ranked PnL table (console)
    - backtest_results.csv
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import HistDataAdapter
from core.market_data_store import MarketDataStore
from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry
from detectors.strategies.base import BaseStrategy
from detectors.signal import PatternSignal

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("backtest")


# =========================================================================
# Trade model
# =========================================================================

@dataclass
class Trade:
    strategy: str
    direction: str          # LONG or SHORT
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""   # "TP", "SL", "EOD"
    pnl_pips: float = 0.0


# =========================================================================
# Backtester engine
# =========================================================================

class Backtester:
    """
    Walk-forward backtester.

    At each M1 candle:
      1. Call strategy.evaluate(windows) with current M5/M1 (or M1/M5/H1)
      2. If signal fires → open trade with TP/SL from metadata
      3. Check subsequent candles for TP/SL hit (intra-bar)
      4. If neither hit by end of data → close at last price (EOD)
    """

    def __init__(
        self,
        data_store: MarketDataStore,
        symbol: str = "USDJPY",
        initial_balance: float = 10000.0,
        lot_size: float = 0.01,       # mini lot
        pip_value: float = 0.01,      # JPY pair: 1 pip = 0.01
        max_open_trades: int = 1,      # max concurrent trades
    ) -> None:
        self._store = data_store
        self._symbol = symbol
        self._initial_balance = initial_balance
        self._lot_size = lot_size
        self._pip_value = pip_value
        self._max_open = max_open_trades

    def run(
        self,
        strategy: BaseStrategy,
        required_tfs: list[str],
    ) -> dict:
        """
        Run backtest for a single strategy.

        Returns dict with:
          - trades: list[Trade]
          - total_pnl_pips: float
          - win_rate: float
          - profit_factor: float
          - max_drawdown_pips: float
          - total_trades: int
          - winning_trades: int
          - losing_trades: int
        """
        # Get M1 data for walk-forward
        m1_df = self._store.get_data(self._symbol, "M1")
        total_candles = len(m1_df)

        trades: list[Trade] = []
        open_trade: Optional[Trade] = None
        balance_curve = [self._initial_balance]
        peak_balance = self._initial_balance
        max_dd = 0.0

        # Minimum lookback for indicators
        lookback = 100

        for i in range(lookback, total_candles):
            current_ts = m1_df.index[i]
            current_candle = m1_df.iloc[i]

            # --- Check open trade for TP/SL ---
            if open_trade is not None:
                hit_tp, hit_sl = False, False

                if open_trade.direction == "LONG":
                    # SL hit: low <= SL
                    if current_candle["low"] <= open_trade.stop_loss:
                        hit_sl = True
                        open_trade.exit_price = open_trade.stop_loss
                    # TP hit: high >= TP
                    elif current_candle["high"] >= open_trade.take_profit:
                        hit_tp = True
                        open_trade.exit_price = open_trade.take_profit
                else:  # SHORT
                    # SL hit: high >= SL
                    if current_candle["high"] >= open_trade.stop_loss:
                        hit_sl = True
                        open_trade.exit_price = open_trade.stop_loss
                    # TP hit: low <= TP
                    elif current_candle["low"] <= open_trade.take_profit:
                        hit_tp = True
                        open_trade.exit_price = open_trade.take_profit

                if hit_tp or hit_sl:
                    open_trade.exit_time = current_ts
                    open_trade.exit_reason = "TP" if hit_tp else "SL"
                    open_trade.pnl_pips = self._calc_pnl(open_trade)
                    trades.append(open_trade)
                    open_trade = None

            # --- Fetch windows for strategy ---
            windows = {}
            for tf in required_tfs:
                try:
                    windows[tf] = self._store.get_window(
                        self._symbol, tf, current_ts, lookback=lookback
                    )
                except KeyError:
                    pass

            if not windows:
                continue

            # --- Evaluate strategy ---
            signals = strategy.evaluate(windows, current_ts)

            # --- Add TP/SL to signals that don't have it ---
            for sig in signals:
                if "entry_price" not in sig.metadata or not sig.metadata.get("take_profit"):
                    # Use the primary TF window for ATR calculation
                    primary_tf = required_tfs[0] if required_tfs else "M1"
                    primary_window = windows.get(primary_tf)
                    if primary_window is not None and not primary_window.empty:
                        BaseStrategy.compute_tp_sl(sig, primary_window)

            # --- Open new trade if no open trade ---
            if signals and open_trade is None:
                sig = signals[0]  # take first signal
                direction = sig.metadata.get("direction", "")
                entry_price = sig.metadata.get("entry_price", current_candle["close"])
                tp = sig.metadata.get("take_profit", 0)
                sl = sig.metadata.get("stop_loss", 0)

                # Deduplicate: skip if same entry as last trade
                if trades and trades[-1].entry_price == entry_price and trades[-1].strategy == strategy.name:
                    continue

                if direction in ("LONG", "SHORT") and tp and sl:
                    open_trade = Trade(
                        strategy=strategy.name,
                        direction=direction,
                        entry_time=current_ts,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                    )

            # --- Update balance curve ---
            realized_pnl = sum(t.pnl_pips for t in trades)
            unrealized = 0.0
            if open_trade:
                if open_trade.direction == "LONG":
                    unrealized = (current_candle["close"] - open_trade.entry_price) / self._pip_value
                else:
                    unrealized = (open_trade.entry_price - current_candle["close"]) / self._pip_value

            current_balance = self._initial_balance + (realized_pnl + unrealized) * self._lot_size * 100
            balance_curve.append(current_balance)
            peak_balance = max(peak_balance, current_balance)
            dd = (peak_balance - current_balance) / peak_balance * 100
            max_dd = max(max_dd, dd)

        # --- Close any remaining open trade at last price ---
        if open_trade is not None:
            open_trade.exit_time = m1_df.index[-1]
            open_trade.exit_price = float(m1_df["close"].iloc[-1])
            open_trade.exit_reason = "EOD"
            open_trade.pnl_pips = self._calc_pnl(open_trade)
            trades.append(open_trade)

        # --- Compute stats ---
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl_pips > 0]
        losing = [t for t in trades if t.pnl_pips <= 0]
        total_pnl = sum(t.pnl_pips for t in trades)
        gross_profit = sum(t.pnl_pips for t in winning)
        gross_loss = abs(sum(t.pnl_pips for t in losing))

        return {
            "strategy": strategy.name,
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / total_trades * 100 if total_trades > 0 else 0,
            "total_pnl_pips": round(total_pnl, 2),
            "avg_pnl_pips": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "max_drawdown_pct": round(max_dd, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "trades": trades,
        }

    def _calc_pnl(self, trade: Trade) -> float:
        """Calculate PnL in pips."""
        if trade.exit_price is None:
            return 0.0
        if trade.direction == "LONG":
            return (trade.exit_price - trade.entry_price) / self._pip_value
        else:
            return (trade.entry_price - trade.exit_price) / self._pip_value


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 80)
    print("RSFX BACKTESTER — All Strategies on USDJPY M1")
    print("=" * 80)

    # --- Load data ---
    csv_path = Path(__file__).parent / "data" / "DAT_ASCII_USDJPY_M1_202605.csv"
    print(f"\nLoading {csv_path}...")
    adapter = HistDataAdapter()
    m1_df = adapter.load(str(csv_path))
    print(f"Loaded {len(m1_df)} M1 candles ({m1_df.index[0]} → {m1_df.index[-1]})")

    store = MarketDataStore()
    store.load_symbol("USDJPY", m1_df)

    # --- Populate strategy registry ---
    print("Loading strategies...")
    _populate_registry()
    print(f"Found {len(STRATEGY_REGISTRY)} strategies\n")

    # --- Run backtest for each strategy ---
    bt = Backtester(store, symbol="USDJPY")
    results = []

    for name, info in STRATEGY_REGISTRY.items():
        cls = info["class"]
        required_tfs = info["timeframes"]

        try:
            strategy = cls()
            result = bt.run(strategy, required_tfs)
            results.append(result)
            status = "✓" if result["total_trades"] > 0 else "○"
            print(f"  {status} {name:40s} trades={result['total_trades']:3d}  pnl={result['total_pnl_pips']:+8.1f} pips  WR={result['win_rate']:.0f}%")
        except Exception as e:
            print(f"  ✗ {name:40s} ERROR: {e}")
            results.append({
                "strategy": name,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl_pips": 0,
                "avg_pnl_pips": 0,
                "profit_factor": 0,
                "max_drawdown_pct": 0,
                "gross_profit": 0,
                "gross_loss": 0,
                "trades": [],
            })

    # --- Sort by PnL ---
    results.sort(key=lambda x: x["total_pnl_pips"], reverse=True)

    # --- Print ranked table ---
    print("\n" + "=" * 80)
    print("RANKED RESULTS (by Total PnL)")
    print("=" * 80)
    print(f"{'Rank':>4}  {'Strategy':40s}  {'Trades':>6}  {'Win%':>5}  {'PnL':>10}  {'Avg':>8}  {'PF':>6}  {'MaxDD%':>7}")
    print("-" * 100)

    for i, r in enumerate(results, 1):
        if r["total_trades"] == 0:
            continue
        print(
            f"{i:4d}  {r['strategy']:40s}  {r['total_trades']:6d}  "
            f"{r['win_rate']:5.1f}  {r['total_pnl_pips']:+10.1f}  "
            f"{r['avg_pnl_pips']:+8.1f}  {r['profit_factor']:6.2f}  "
            f"{r['max_drawdown_pct']:7.2f}"
        )

    # --- Save CSV ---
    csv_rows = []
    for r in results:
        csv_rows.append({
            "rank": 0,
            "strategy": r["strategy"],
            "total_trades": r["total_trades"],
            "win_rate": r["win_rate"],
            "total_pnl_pips": r["total_pnl_pips"],
            "avg_pnl_pips": r["avg_pnl_pips"],
            "profit_factor": r["profit_factor"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "gross_profit": r["gross_profit"],
            "gross_loss": r["gross_loss"],
        })

    df = pd.DataFrame(csv_rows)
    df = df[df["total_trades"] > 0].sort_values("total_pnl_pips", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    csv_path = Path(__file__).parent / "backtest_results.csv"
    df.to_csv(str(csv_path))
    print(f"\nResults saved to {csv_path}")

    # --- Top 5 summary ---
    print("\n" + "=" * 80)
    print("TOP 5 STRATEGIES")
    print("=" * 80)
    for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
        print(f"\n  #{i} {row['strategy']}")
        print(f"     Trades: {row['total_trades']} | Win Rate: {row['win_rate']:.1f}%")
        print(f"     Total PnL: {row['total_pnl_pips']:+.1f} pips | Avg: {row['avg_pnl_pips']:+.1f} pips/trade")
        print(f"     Profit Factor: {row['profit_factor']:.2f} | Max DD: {row['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()
