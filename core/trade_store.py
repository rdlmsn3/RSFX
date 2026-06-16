"""
backtest/trade_store.py
-----------------------
SQLite persistence for backtest trades.

Stores every trade from backtest / confluence runs in a single
SQLite database with two tables:

  runs    — one row per backtest execution (metadata, summary stats)
  trades  — one row per trade (linked to run_id)

Usage:
    from core.trade_store import init_db, save_trades

    db = init_db("results/trades.db")
    run_id = save_trades(db, trades, run_meta, result_stats)
"""

from __future__ import annotations
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    data_file       TEXT,
    symbol          TEXT,
    strategies      TEXT,
    lookback        INTEGER,
    threshold       INTEGER,
    n_strategies    INTEGER,
    total_trades    INTEGER,
    winning_trades  INTEGER,
    losing_trades   INTEGER,
    win_rate        REAL,
    total_pnl_pips  REAL,
    avg_pnl_pips    REAL,
    expectancy_pips REAL,
    profit_factor   REAL,
    gross_profit    REAL,
    gross_loss      REAL,
    avg_mae_pips    REAL,
    avg_mfe_pips    REAL,
    initial_balance REAL,
    final_balance   REAL,
    elapsed_sec     REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    strategy        TEXT,
    direction       TEXT,
    entry_time      TEXT,
    entry_price     REAL,
    stop_loss       REAL,
    take_profit     REAL,
    exit_time       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    pnl_pips        REAL,
    mae_pips        REAL,
    mfe_pips        REAL,
    bars_held       INTEGER,
    risk_pips       REAL,
    reward_pips     REAL,
    rr_ratio        REAL,
    signal_meta     TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_run_id    ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy  ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_entry     ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_pnl       ON trades(pnl_pips);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(db_path: str | Path = "results/trades.db") -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and ensure schema exists.

    Returns a connection with WAL mode for concurrent reads.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()

    logger.info("Trade store ready: %s", db_path)
    return conn


def save_trades(
    conn: sqlite3.Connection,
    trades: list,                      # list[Trade] or list[dict]
    run_meta: dict[str, Any],          # data_file, symbol, strategies, etc.
    result_stats: dict[str, Any] | None = None,  # summary stats from _print_results
) -> int:
    """
    Insert a run + its trades into the database.

    Parameters
    ----------
    conn        : SQLite connection from init_db()
    trades      : list of Trade dataclasses (must have .to_dict()) or dicts
    run_meta    : dict with keys: data_file, symbol, strategies (list),
                  lookback, threshold, n_strategies
    result_stats: dict from _print_results() — total_trades, win_rate, etc.

    Returns
    -------
    int — the run_id (primary key)
    """
    result_stats = result_stats or {}
    run_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- Insert run row ------------------------------------------------
    cur = conn.execute(
        """INSERT INTO runs (
            timestamp, data_file, symbol, strategies, lookback, threshold,
            n_strategies, total_trades, winning_trades, losing_trades,
            win_rate, total_pnl_pips, avg_pnl_pips, expectancy_pips,
            profit_factor, gross_profit, gross_loss, avg_mae_pips, avg_mfe_pips,
            initial_balance, final_balance, elapsed_sec
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_ts,
            run_meta.get("data_file", ""),
            run_meta.get("symbol", ""),
            ",".join(run_meta.get("strategies", [])),
            run_meta.get("lookback", 0),
            run_meta.get("threshold", 0),
            run_meta.get("n_strategies", 0),
            result_stats.get("total_trades", len(trades)),
            result_stats.get("winning_trades", 0),
            result_stats.get("losing_trades", 0),
            result_stats.get("win_rate", 0.0),
            result_stats.get("total_pnl_pips", 0.0),
            result_stats.get("avg_pnl_pips", 0.0),
            result_stats.get("expectancy_pips", 0.0),
            result_stats.get("profit_factor", 0.0),
            result_stats.get("gross_profit", 0.0),
            result_stats.get("gross_loss", 0.0),
            result_stats.get("avg_mae_pips", 0.0),
            result_stats.get("avg_mfe_pips", 0.0),
            result_stats.get("initial_balance", 0.0),
            result_stats.get("final_balance", 0.0),
            result_stats.get("elapsed_sec", 0.0),
        ),
    )
    run_id: int = cur.lastrowid or 0  # guaranteed non-None after INSERT

    # ---- Insert trades -------------------------------------------------
    rows = []
    for t in trades:
        d = t.to_dict() if hasattr(t, "to_dict") else dict(t)
        # Flatten signal_meta to JSON
        meta_raw = d.pop("signal_meta", None)
        if meta_raw is None:
            # to_dict() flattens signal_meta with sig_ prefix
            meta_raw = {k.removeprefix("sig_"): v
                        for k, v in d.items() if k.startswith("sig_")}
            for k in list(meta_raw.keys()):
                d.pop(f"sig_{k}", None)

        meta_json = json.dumps(meta_raw) if meta_raw else None

        rows.append((
            run_id,
            d.get("strategy", ""),
            d.get("direction", ""),
            str(d.get("entry_time", "")),
            d.get("entry_price", 0.0),
            d.get("stop_loss", 0.0),
            d.get("take_profit", 0.0),
            str(d.get("exit_time", "")),
            d.get("exit_price", 0.0),
            d.get("exit_reason", ""),
            d.get("pnl_pips", 0.0),
            d.get("mae_pips", 0.0),
            d.get("mfe_pips", 0.0),
            d.get("bars_held", 0),
            d.get("risk_pips", 0.0),
            d.get("reward_pips", 0.0),
            d.get("rr_ratio", 0.0),
            meta_json,
        ))

    conn.executemany(
        """INSERT INTO trades (
            run_id, strategy, direction, entry_time, entry_price,
            stop_loss, take_profit, exit_time, exit_price, exit_reason,
            pnl_pips, mae_pips, mfe_pips, bars_held,
            risk_pips, reward_pips, rr_ratio, signal_meta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    logger.info("Saved run #%d: %d trades → trades.db", run_id, len(rows))
    return run_id


def get_run_summary(conn: sqlite3.Connection, run_id: int | None = None) -> dict:
    """Return the most recent run (or specific run_id) summary."""
    if run_id:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    else:
        row = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return {}
    cols = [d[0] for d in conn.execute("SELECT * FROM runs LIMIT 0").description]
    return dict(zip(cols, row))


def get_trades(
    conn: sqlite3.Connection,
    run_id: int | None = None,
    strategy: str | None = None,
    min_pnl: float | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Query trades with optional filters."""
    query = "SELECT t.*, r.timestamp AS run_time FROM trades t JOIN runs r ON t.run_id = r.id WHERE 1=1"
    params: list = []

    if run_id:
        query += " AND t.run_id = ?"
        params.append(run_id)
    if strategy:
        query += " AND t.strategy LIKE ?"
        params.append(f"%{strategy}%")
    if min_pnl is not None:
        query += " AND t.pnl_pips >= ?"
        params.append(min_pnl)

    query += " ORDER BY t.entry_time DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    cols = [d[0] for d in conn.execute(
        "SELECT t.*, r.timestamp AS run_time FROM trades t JOIN runs r ON t.run_id = r.id LIMIT 0"
    ).description]
    return [dict(zip(cols, r)) for r in rows]
