"""
strategy_correlation.py
-----------------------
Strategy correlation analysis based on backtest results.

Reads backtest_trades.csv and backtest_results.csv (produced by backtest.py)
and computes:

  1. Trade overlap matrix        — % of trades that overlap in time per pair
  2. PnL correlation matrix      — Pearson correlation of per-trade PnL
  3. Equity curve correlation    — Correlation of resampled balance curves
  4. Confluence analysis         — Combined win rate when N strategies agree
  5. Diversification score       — Rank strategy baskets by low mutual correlation
  6. CSV + terminal output       — strategy_correlation.csv + ranked report

Usage:
    python3 strategy_correlation.py
    python3 strategy_correlation.py --trades backtest_trades.csv --summary backtest_results.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd


# =========================================================================
# Data Loading
# =========================================================================

def load_trades(path: Path) -> pd.DataFrame:
    """Load backtest_trades.csv and parse timestamps."""
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    return df


def load_summary(path: Path) -> pd.DataFrame:
    """Load backtest_results.csv."""
    return pd.read_csv(path)


# =========================================================================
# 1. Trade Overlap Matrix
# =========================================================================

def compute_overlap_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    """
    For each strategy pair, compute % of trades that overlap in time.

    Two trades overlap if their [entry_time, exit_time] windows intersect.
    Returns a symmetric DataFrame where cell (A, B) = overlap % of A's trades
    that overlap with any of B's trades.
    """
    strategies = sorted(trades["strategy"].unique())
    n = len(strategies)

    # Group trades by strategy
    strat_trades: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for strat in strategies:
        mask = trades["strategy"] == strat
        strat_trades[strat] = list(
            zip(trades.loc[mask, "entry_time"], trades.loc[mask, "exit_time"])
        )

    matrix = pd.DataFrame(np.zeros((n, n)), index=strategies, columns=strategies)

    for i, s_a in enumerate(strategies):
        trades_a = strat_trades[s_a]
        if not trades_a:
            continue

        for j, s_b in enumerate(strategies):
            if j <= i:
                continue
            trades_b = strat_trades[s_b]
            if not trades_b:
                continue

            # For each trade in A, check if it overlaps any trade in B
            overlap_count = 0
            for a_entry, a_exit in trades_a:
                for b_entry, b_exit in trades_b:
                    # Overlap exists if a_entry < b_exit and b_entry < a_exit
                    if a_entry < b_exit and b_entry < a_exit:
                        overlap_count += 1
                        break  # count each A-trade at most once

            pct = overlap_count / len(trades_a) * 100 if trades_a else 0.0
            matrix.loc[s_a, s_b] = round(pct, 1)
            matrix.loc[s_b, s_a] = round(pct, 1)

    # Diagonal = 100% (self-overlap)
    for s in strategies:
        matrix.loc[s, s] = 100.0

    return matrix


# =========================================================================
# 2. PnL Correlation Matrix
# =========================================================================

def compute_pnl_correlation(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation of per-trade PnL between strategy pairs.

    Aligns trades by entry_time (within 1-minute tolerance) and computes
    correlation of their pnl_pips values.
    """
    strategies = sorted(trades["strategy"].unique())

    # Pivot: for each entry_time, get PnL per strategy
    # Use minute-level grouping for alignment
    trades = trades.copy()
    trades["entry_minute"] = trades["entry_time"].dt.floor("1min")

    pivot = trades.pivot_table(
        index="entry_minute",
        columns="strategy",
        values="pnl_pips",
        aggfunc="mean",  # if multiple trades same minute, average
    )

    # Fill NaN for strategies that didn't trade at that minute
    # Use pairwise correlation (min periods = 10 for significance)
    corr = pivot.corr(method="pearson", min_periods=10)

    # Reindex to match strategy order
    corr = corr.reindex(index=strategies, columns=strategies)

    # Round
    corr = corr.round(3)

    return corr


# =========================================================================
# 3. Equity Curve Correlation
# =========================================================================

def compute_equity_correlation(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate balance curves across strategies.

    Builds a per-strategy equity curve from trades (cumulative PnL),
    resamples to hourly timestamps, then computes Pearson correlation.
    """
    strategies = sorted(trades["strategy"].unique())

    # Build equity curves per strategy
    equity_curves: dict[str, pd.Series] = {}
    for strat in strategies:
        strat_trades = (
            trades[trades["strategy"] == strat]
            .sort_values("exit_time")
            .set_index("exit_time")["pnl_pips"]
        )
        # Handle duplicate timestamps by grouping and summing PnL
        strat_trades = strat_trades.groupby(strat_trades.index).sum()
        strat_trades = strat_trades.cumsum()
        equity_curves[strat] = strat_trades

    # Combine into DataFrame, resample to hourly
    combined = pd.DataFrame(equity_curves)
    combined = combined.resample("1h").last().ffill()

    # Correlation
    corr = combined.corr(method="pearson", min_periods=5)
    corr = corr.reindex(index=strategies, columns=strategies).round(3)

    return corr


# =========================================================================
# 4. Confluence Analysis
# =========================================================================

def compute_confluence(trades: pd.DataFrame) -> pd.DataFrame:
    """
    When multiple strategies fire on the same candle (entry_time match),
    what's the combined win rate vs individual?

    Groups trades by entry_minute, counts how many strategies fired,
    and computes stats per confluence level.
    """
    trades = trades.copy()
    trades["entry_minute"] = trades["entry_time"].dt.floor("1min")
    trades["is_win"] = trades["pnl_pips"] > 0

    # Count strategies per minute
    group = trades.groupby("entry_minute").agg(
        n_strategies=("strategy", "nunique"),
        n_trades=("strategy", "count"),
        total_pnl=("pnl_pips", "sum"),
        wins=("is_win", "sum"),
        total=("is_win", "count"),
        strategies=("strategy", lambda x: ", ".join(sorted(set(x)))),
    )

    group["win_rate"] = (group["wins"] / group["total"] * 100).round(1)
    group["avg_pnl"] = (group["total_pnl"] / group["n_trades"]).round(2)

    # Aggregate by confluence level
    confluence = (
        group.groupby("n_strategies")
        .agg(
            n_signals=("total", "sum"),
            n_windows=("n_strategies", "count"),
            total_wins=("wins", "sum"),
            total_trades=("total", "sum"),
            avg_pnl_per_trade=("avg_pnl", "mean"),
        )
    )
    confluence["win_rate"] = (
        confluence["total_wins"] / confluence["total_trades"] * 100
    ).round(1)
    confluence["avg_pnl_per_trade"] = confluence["avg_pnl_per_trade"].round(2)

    confluence.index.name = "confluence_level"
    confluence = confluence.reset_index()
    confluence.columns = [
        "confluence_level",
        "total_signals",
        "n_time_windows",
        "total_wins",
        "total_trades",
        "avg_pnl_per_trade",
        "win_rate_pct",
    ]

    return confluence


# =========================================================================
# 5. Diversification Score + Portfolio Basket Ranking
# =========================================================================

def compute_diversification_score(
    equity_corr: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each strategy, compute a diversification score based on its
    average correlation with all other profitable strategies.

    Lower avg correlation = better diversification candidate.

    Also ranks top baskets of 3 strategies by minimum mutual correlation.
    """
    strategies = sorted(equity_corr.columns)

    # Filter to profitable strategies from summary
    profitable = summary[summary["total_pnl_pips"] > 0]["strategy"].tolist()
    profitable = [s for s in profitable if s in strategies]

    if not profitable:
        return pd.DataFrame()

    # Per-strategy avg correlation with other profitable strategies
    rows = []
    for s in profitable:
        peers = [p for p in profitable if p != s]
        if not peers:
            continue
        avg_corr = equity_corr.loc[s, peers].mean()
        rows.append({
            "strategy": s,
            "avg_correlation": round(avg_corr, 3),
            "diversification_score": round(1 - avg_corr, 3),  # higher = better
            "total_pnl_pips": summary[summary["strategy"] == s]["total_pnl_pips"].values[0],
            "win_rate": summary[summary["strategy"] == s]["win_rate"].values[0],
            "profit_factor": summary[summary["strategy"] == s]["profit_factor"].values[0],
        })

    scores = pd.DataFrame(rows).sort_values("diversification_score", ascending=False)

    # Top baskets of 3 — lowest mutual avg correlation
    baskets = []
    for combo in combinations(profitable, 3):
        sub = equity_corr.loc[list(combo), list(combo)]
        # Average of off-diagonal elements
        mask = np.ones((3, 3), dtype=bool)
        np.fill_diagonal(mask, False)
        avg_mutual_corr = sub.values[mask].mean()
        total_pnl = sum(
            summary[summary["strategy"] == s]["total_pnl_pips"].values[0]
            for s in combo
        )
        avg_wr = np.mean([
            summary[summary["strategy"] == s]["win_rate"].values[0]
            for s in combo
        ])
        baskets.append({
            "strategies": " + ".join(combo),
            "avg_mutual_correlation": round(avg_mutual_corr, 3),
            "combined_pnl_pips": round(total_pnl, 1),
            "avg_win_rate": round(avg_wr, 1),
        })

    basket_df = (
        pd.DataFrame(baskets)
        .sort_values("avg_mutual_correlation", ascending=True)
        .head(20)
        .reset_index(drop=True)
    )

    return scores, basket_df


# =========================================================================
# 6. Terminal Report + CSV Export
# =========================================================================

def _print_matrix(matrix: pd.DataFrame, title: str, fmt: str = ".1f") -> None:
    """Print a correlation/overlap matrix in a compact format."""
    n = len(matrix)
    if n == 0:
        return

    # Truncate strategy names for display
    short = {c: (c[:18] + ".." if len(c) > 20 else c) for c in matrix.columns}

    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")

    # Header
    header = f"{'':20s}"
    for c in matrix.columns:
        header += f" {short[c]:>7s}"
    print(header)
    print("-" * (20 + 8 * n))

    for i, row_name in enumerate(matrix.columns):
        line = f"{short[row_name]:20s}"
        for j, col_name in enumerate(matrix.columns):
            val = matrix.iloc[i, j]
            if i == j:
                line += f" {'---':>7s}"
            else:
                line += f" {val:{fmt}}{'' :>1s}"
        print(line)


def _print_confluence(confluence: pd.DataFrame) -> None:
    """Print confluence analysis table."""
    print(f"\n{'=' * 70}")
    print("CONFLUENCE ANALYSIS — Win rate when N strategies fire together")
    print(f"{'=' * 70}")
    print(
        f"{'Level':>8}  {'Signals':>8}  {'Windows':>8}  "
        f"{'Trades':>8}  {'Wins':>6}  {'WinRate':>8}  {'AvgPnL':>8}"
    )
    print("-" * 70)
    for _, row in confluence.iterrows():
        print(
            f"{int(row['confluence_level']):>8d}  "
            f"{int(row['total_signals']):>8d}  "
            f"{int(row['n_time_windows']):>8d}  "
            f"{int(row['total_trades']):>8d}  "
            f"{int(row['total_wins']):>6d}  "
            f"{row['win_rate_pct']:>7.1f}%  "
            f"{row['avg_pnl_per_trade']:>+7.2f}"
        )


def _print_baskets(baskets: pd.DataFrame) -> None:
    """Print top portfolio baskets."""
    print(f"\n{'=' * 90}")
    print("TOP 20 PORTFOLIO BASKETS — Lowest mutual correlation = best diversification")
    print(f"{'=' * 90}")
    print(
        f"{'#':>3}  {'AvgCorr':>8}  {'Combined PnL':>13}  {'AvgWR':>7}  Strategies"
    )
    print("-" * 90)
    for i, row in baskets.iterrows():
        strats = row["strategies"]
        if len(strats) > 60:
            strats = strats[:57] + "..."
        print(
            f"{i+1:>3d}  "
            f"{row['avg_mutual_correlation']:>+8.3f}  "
            f"{row['combined_pnl_pips']:>+12.1f}  "
            f"{row['avg_win_rate']:>6.1f}%  "
            f"{strats}"
        )


def run_analysis(
    trades_path: Path,
    summary_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """
    Run full correlation analysis. Returns dict of DataFrames for programmatic use.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"

    print(f"Loading trades from {trades_path}...")
    trades = load_trades(trades_path)
    print(f"  {len(trades):,} trades across {trades['strategy'].nunique()} strategies")

    print(f"Loading summary from {summary_path}...")
    summary = load_summary(summary_path)

    # 1. Overlap matrix
    print("\n[1/6] Computing trade overlap matrix...")
    overlap = compute_overlap_matrix(trades)

    # 2. PnL correlation
    print("[2/6] Computing PnL correlation...")
    pnl_corr = compute_pnl_correlation(trades)

    # 3. Equity curve correlation
    print("[3/6] Computing equity curve correlation...")
    equity_corr = compute_equity_correlation(trades)

    # 4. Confluence analysis
    print("[4/6] Computing confluence analysis...")
    confluence = compute_confluence(trades)

    # 5. Diversification score
    print("[5/6] Computing diversification scores...")
    scores, baskets = compute_diversification_score(equity_corr, summary)

    # 6. Terminal report
    print("[6/6] Generating report...\n")
    _print_matrix(overlap, "TRADE OVERLOlap MATRIX (% of A's trades overlapping B)")
    _print_matrix(pnl_corr, "PNL CORRELATION MATRIX (Pearson per-trade PnL)", fmt=".3f")
    _print_matrix(equity_corr, "EQUITY CURVE CORRELATION MATRIX", fmt=".3f")
    _print_confluence(confluence)

    if not scores.empty:
        print(f"\n{'=' * 80}")
        print("DIVERSIFICATION SCORES (lower avg correlation = better)")
        print(f"{'=' * 80}")
        print(
            f"{'Strategy':35s}  {'AvgCorr':>8}  {'DivScore':>9}  "
            f"{'PnL':>10}  {'WR':>6}  {'PF':>6}"
        )
        print("-" * 80)
        for _, row in scores.iterrows():
            print(
                f"{row['strategy']:35s}  "
                f"{row['avg_correlation']:>+8.3f}  "
                f"{row['diversification_score']:>9.3f}  "
                f"{row['total_pnl_pips']:>+10.1f}  "
                f"{row['win_rate']:>5.1f}%  "
                f"{row['profit_factor']:>6.2f}"
            )

    if not baskets.empty:
        _print_baskets(baskets)

    # Save CSVs
    overlap.to_csv(str(output_dir / "corr_overlap.csv"))
    pnl_corr.to_csv(str(output_dir / "corr_pnl.csv"))
    equity_corr.to_csv(str(output_dir / "corr_equity.csv"))
    confluence.to_csv(str(output_dir / "corr_confluence.csv"), index=False)
    if not baskets.empty:
        baskets.to_csv(str(output_dir / "corr_baskets.csv"), index=False)
    if not scores.empty:
        scores.to_csv(str(output_dir / "corr_diversification.csv"), index=False)

    print(f"\n✅ CSVs saved to {output_dir}/")
    print("   corr_overlap.csv, corr_pnl.csv, corr_equity.csv")
    print("   corr_confluence.csv, corr_baskets.csv, corr_diversification.csv")

    return {
        "overlap": overlap,
        "pnl_correlation": pnl_corr,
        "equity_correlation": equity_corr,
        "confluence": confluence,
        "diversification_scores": scores,
        "top_baskets": baskets,
    }


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strategy correlation analysis from backtest results"
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "backtest_trades_latest.csv",
        help="Path to backtest_trades.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "backtest_results_latest.csv",
        help="Path to backtest_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as trades file)",
    )
    args = parser.parse_args()

    if not args.trades.exists():
        print(f"ERROR: {args.trades} not found. Run backtest.py first.")
        sys.exit(1)
    if not args.summary.exists():
        print(f"ERROR: {args.summary} not found. Run backtest.py first.")
        sys.exit(1)

    run_analysis(args.trades, args.summary, args.output)


if __name__ == "__main__":
    main()
