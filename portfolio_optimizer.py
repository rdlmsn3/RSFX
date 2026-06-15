"""
portfolio_optimizer.py
----------------------
Brute-force portfolio optimizer — finds best strategy combinations.

Merges trades from N strategies, builds combined equity curves,
and computes risk-adjusted metrics (Sharpe, Sortino, Calmar, MaxDD).

Reads backtest_trades.csv + backtest_results.csv (from backtest.py).

Usage:
    python3 portfolio_optimizer.py                  # 2 + 3 strategy combos
    python3 portfolio_optimizer.py --max-combo 4    # also test 4-strategy combos
    python3 portfolio_optimizer.py --top 30         # show top 30 results
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================================
# Data Loading
# =========================================================================

def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    return df


def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# =========================================================================
# 1. Portfolio Merger + Equity Curve
# =========================================================================

def build_portfolio_equity(
    trades: pd.DataFrame,
    strategy_names: list[str],
    initial_balance: float = 10_000.0,
    lot_size: float = 0.01,
) -> pd.Series:
    """
    Merge trades from selected strategies and build a combined equity curve.

    Returns a Series indexed by hourly timestamps with running balance.
    """
    # Filter to combo strategies
    mask = trades["strategy"].isin(strategy_names)
    combo_trades = trades[mask].copy()

    if combo_trades.empty:
        return pd.Series(dtype=float)

    # Sort by exit_time (trade is "realized" at exit)
    combo_trades = combo_trades.sort_values("exit_time")

    # Build equity curve: each trade adds its PnL * lot_size * 100 to balance
    pnl_per_dollar = lot_size * 100  # same as backtest.py
    combo_trades["balance_delta"] = combo_trades["pnl_pips"] * pnl_per_dollar

    # Set index to exit_time, cumsum, add initial balance
    equity = (
        combo_trades.set_index("exit_time")["balance_delta"]
        .cumsum()
        + initial_balance
    )

    # Resample to hourly for consistent comparison (ffill for gaps)
    equity = equity.resample("1h").last().ffill()

    # Ensure starts at initial_balance
    if len(equity) > 0:
        equity.iloc[0] = initial_balance

    return equity


# =========================================================================
# 2. Risk Metrics
# =========================================================================

@dataclass
class PortfolioMetrics:
    strategy_combo: str
    n_strategies: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_pips: float
    total_pnl_dollars: float
    avg_pnl_pips: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_dollars: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_win: float
    avg_loss: float
    expectancy: float
    avg_bars_held: float
    equity_curve_length: int

    def to_dict(self) -> dict:
        return {
            "strategy_combo": self.strategy_combo,
            "n_strategies": self.n_strategies,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_pnl_pips": round(self.total_pnl_pips, 2),
            "total_pnl_dollars": round(self.total_pnl_dollars, 2),
            "avg_pnl_pips": round(self.avg_pnl_pips, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_drawdown_dollars": round(self.max_drawdown_dollars, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "expectancy": round(self.expectancy, 2),
            "avg_bars_held": round(self.avg_bars_held, 1),
            "equity_curve_length": self.equity_curve_length,
        }


def compute_risk_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    strategy_names: list[str],
    initial_balance: float = 10_000.0,
    lot_size: float = 0.01,
) -> Optional[PortfolioMetrics]:
    """Compute full risk metrics from equity curve + trade data."""

    combo_str = " + ".join(strategy_names)

    if equity.empty or len(equity) < 3:
        return None

    # --- Trade stats ---
    mask = trades["strategy"].isin(strategy_names)
    ct = trades[mask]
    total = len(ct)
    if total == 0:
        return None

    winning = ct[ct["pnl_pips"] > 0]
    losing = ct[ct["pnl_pips"] <= 0]

    total_pnl_pips = ct["pnl_pips"].sum()
    gross_profit = winning["pnl_pips"].sum() if len(winning) else 0
    gross_loss = abs(losing["pnl_pips"].sum()) if len(losing) else 0
    win_rate = len(winning) / total * 100
    avg_win = gross_profit / len(winning) if len(winning) else 0
    avg_loss = gross_loss / len(losing) if len(losing) else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss

    pnl_per_dollar = lot_size * 100
    total_pnl_dollars = total_pnl_pips * pnl_per_dollar

    # --- Equity curve stats ---
    eq = equity.values
    eq_len = len(eq)

    # Returns (hourly)
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]

    # Max Drawdown
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max * 100
    drawdown_dollars = eq - running_max
    max_dd_pct = abs(float(drawdowns.min()))
    max_dd_dollars = abs(float(drawdown_dollars.min()))

    # Sharpe (annualized: ~252 days * 24 hours = 6048 hourly periods per year)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(6048))
    else:
        sharpe = 0.0

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(6048))
    else:
        sortino = 0.0

    # Calmar (annualized return / max DD)
    total_hours = (equity.index[-1] - equity.index[0]).total_seconds() / 3600
    total_years = total_hours / (365.25 * 24) if total_hours > 0 else 1
    annual_return_pct = ((eq[-1] / initial_balance) ** (1 / total_years) - 1) * 100 if total_years > 0 else 0
    calmar = annual_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

    return PortfolioMetrics(
        strategy_combo=combo_str,
        n_strategies=len(strategy_names),
        total_trades=total,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        total_pnl_pips=total_pnl_pips,
        total_pnl_dollars=total_pnl_dollars,
        avg_pnl_pips=total_pnl_pips / total,
        profit_factor=pf,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_dollars=max_dd_dollars,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        avg_bars_held=float(ct["bars_held"].mean()),
        equity_curve_length=eq_len,
    )


# =========================================================================
# 3-4. Brute-Force Combo Evaluator
# =========================================================================

def _eval_combo(
    args: tuple,
) -> Optional[dict]:
    """Evaluate a single combo. Top-level for picklability in multiprocessing."""
    combo, trades_dict, initial_balance, lot_size = args

    # Reconstruct trades DataFrame from dict (picklable)
    trades = pd.DataFrame(trades_dict)

    equity = build_portfolio_equity(trades, list(combo), initial_balance, lot_size)
    metrics = compute_risk_metrics(equity, trades, list(combo), initial_balance, lot_size)
    if metrics is None:
        return None
    return metrics.to_dict()


def run_optimizer(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    min_combo: int = 2,
    max_combo: int = 3,
    top_n: int = 20,
    max_workers: int = 4,
) -> list[dict]:
    """
    Brute-force evaluate all strategy combos from min_combo to max_combo.
    Returns sorted list of result dicts.
    """
    strategies = sorted(trades["strategy"].unique())

    # Only use strategies with trades
    active = summary[summary["total_trades"] > 0]["strategy"].tolist()
    active = [s for s in active if s in strategies]
    print(f"  Active strategies: {len(active)}")

    # Pre-convert trades to dict for picklability
    trades_dict = trades.to_dict(orient="records")

    all_results: list[dict] = []

    for combo_size in range(min_combo, max_combo + 1):
        combos = list(combinations(active, combo_size))
        print(f"\n  Testing {len(combos)} combos of {combo_size} strategies...")

        t0 = time.perf_counter()
        evaluated = 0

        # Sequential — fast enough for 20K combos
        for combo in combos:
            equity = build_portfolio_equity(trades, list(combo))
            metrics = compute_risk_metrics(equity, trades, list(combo))
            if metrics is not None:
                all_results.append(metrics.to_dict())
            evaluated += 1

            if evaluated % 5000 == 0:
                elapsed = time.perf_counter() - t0
                rate = evaluated / elapsed if elapsed > 0 else 0
                eta = (len(combos) - evaluated) / rate if rate > 0 else 0
                print(f"    [{evaluated}/{len(combos)}] {rate:.0f} combos/s  ETA {eta:.0f}s")

        elapsed = time.perf_counter() - t0
        print(f"    Done: {len([r for r in all_results if r['n_strategies'] == combo_size])} valid combos in {elapsed:.1f}s")

    # Sort by Sharpe (primary), then PnL (secondary)
    all_results.sort(key=lambda x: (x["sharpe_ratio"], x["total_pnl_pips"]), reverse=True)

    return all_results


# =========================================================================
# 5. Pairs Quick-Scan (optimized path)
# =========================================================================

def run_pairs_scan(trades: pd.DataFrame, summary: pd.DataFrame) -> list[dict]:
    """Fast 2-strategy pair scan."""
    return run_optimizer(trades, summary, min_combo=2, max_combo=2)


# =========================================================================
# 6. Terminal Report + CSV Export
# =========================================================================

def _print_results(results: list[dict], title: str, top_n: int = 20) -> None:
    """Print ranked results table."""
    if not results:
        print(f"\n  No valid combos found for {title}")
        return

    print(f"\n{'=' * 120}")
    print(f"{title} — TOP {min(top_n, len(results))}")
    print(f"{'=' * 120}")
    print(
        f"{'#':>3}  {'Combo':55s}  {'Trades':>6}  {'WR%':>5}  "
        f"{'PnL$':>10}  {'MaxDD%':>7}  {'Sharpe':>7}  {'Sortino':>8}  "
        f"{'Calmar':>7}  {'PF':>6}"
    )
    print("-" * 120)

    for i, r in enumerate(results[:top_n], 1):
        combo = r["strategy_combo"]
        if len(combo) > 55:
            combo = combo[:52] + "..."
        print(
            f"{i:>3d}  {combo:55s}  {r['total_trades']:>6d}  "
            f"{r['win_rate']:>5.1f}  "
            f"{r['total_pnl_dollars']:>+10.0f}  "
            f"{r['max_drawdown_pct']:>7.2f}  "
            f"{r['sharpe_ratio']:>+7.3f}  "
            f"{r['sortino_ratio']:>+8.3f}  "
            f"{r['calmar_ratio']:>+7.2f}  "
            f"{r['profit_factor']:>6.2f}"
        )


def run_analysis(
    trades_path: Path,
    summary_path: Path,
    output_dir: Path | None = None,
    min_combo: int = 2,
    max_combo: int = 3,
    top_n: int = 20,
) -> dict:
    """Full optimization pipeline."""
    if output_dir is None:
        output_dir = trades_path.parent

    print(f"Loading trades from {trades_path}...")
    trades = load_trades(trades_path)
    print(f"  {len(trades):,} trades across {trades['strategy'].nunique()} strategies")

    print(f"Loading summary from {summary_path}...")
    summary = load_summary(summary_path)

    # Single-strategy baselines
    print("\n--- Single Strategy Baselines ---")
    strategies = sorted(trades["strategy"].unique())
    single_results = []
    for s in strategies:
        equity = build_portfolio_equity(trades, [s])
        metrics = compute_risk_metrics(equity, trades, [s])
        if metrics is not None:
            single_results.append(metrics.to_dict())

    single_results.sort(key=lambda x: (x["sharpe_ratio"], x["total_pnl_pips"]), reverse=True)
    _print_results(single_results, "SINGLE STRATEGY BASELINES", top_n=10)

    # Multi-strategy combos
    print(f"\n--- Portfolio Optimizer: {min_combo} to {max_combo} strategies ---")
    t0 = time.perf_counter()
    combo_results = run_optimizer(trades, summary, min_combo=min_combo, max_combo=max_combo, top_n=top_n)
    elapsed = time.perf_counter() - t0

    print(f"\nTotal optimization time: {elapsed:.1f}s ({len(combo_results):,} combos evaluated)")

    # Split by combo size and print
    for size in range(min_combo, max_combo + 1):
        sized = [r for r in combo_results if r["n_strategies"] == size]
        _print_results(sized, f"{size}-STRATEGY PORTFOLIOS", top_n=top_n)

    # Save CSVs
    all_results = single_results + combo_results
    df = pd.DataFrame(all_results)
    df.to_csv(str(output_dir / "portfolio_results.csv"), index=False)
    print(f"\n✅ Saved portfolio_results.csv ({len(df):,} rows)")

    # Also save top combos separately
    top_df = pd.DataFrame(combo_results[:50])
    top_df.to_csv(str(output_dir / "portfolio_top50.csv"), index=False)
    print(f"✅ Saved portfolio_top50.csv")

    return {
        "single_results": single_results,
        "combo_results": combo_results,
    }


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Brute-force portfolio optimizer for strategy combos"
    )
    parser.add_argument(
        "--trades", type=Path,
        default=Path(__file__).parent / "backtest_trades.csv",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path(__file__).parent / "backtest_results.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-combo", type=int, default=2)
    parser.add_argument("--max-combo", type=int, default=3)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if not args.trades.exists():
        print(f"ERROR: {args.trades} not found. Run backtest.py first.")
        sys.exit(1)
    if not args.summary.exists():
        print(f"ERROR: {args.summary} not found. Run backtest.py first.")
        sys.exit(1)

    run_analysis(args.trades, args.summary, args.output, args.min_combo, args.max_combo, args.top)


if __name__ == "__main__":
    main()
