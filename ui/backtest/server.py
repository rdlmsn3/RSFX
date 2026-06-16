"""
backtest/ui/server.py
-----------------------
Minimal FastAPI backend for the backtester UI.
Uses the unified SignalEngine + TradeEngine.

Usage:
    cd /home/rudi/RSFX
    python3 backtest/ui/server.py

Then open http://localhost:8502
"""

import sys
import time
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loader import HistDataAdapter, get_adapter
from core.market_data_store import MarketDataStore
from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry

app = FastAPI(title="RSFX Backtester")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Cache: csv_path -> (MarketDataStore, candle_count, date_range, raw_ticks)
_store_cache: dict[str, tuple[MarketDataStore, int, str, pd.DataFrame | None]] = {}


def get_store(csv_path: str, symbol: str) -> tuple[MarketDataStore, int, pd.DataFrame | None]:
    """Load CSV on demand, cache by path+symbol. Returns (store, candle_count, raw_ticks)."""
    key = f"{csv_path}:{symbol}"
    if key in _store_cache:
        store, n, _, ticks = _store_cache[key]
        return store, n, ticks

    full_path = csv_path
    if not Path(full_path).is_absolute():
        full_path = str(DATA_DIR / csv_path)

    print(f"Loading {full_path}...")
    adapter = get_adapter(full_path)
    m1_df = adapter.load(full_path)
    raw_ticks = adapter.raw_ticks  # None for bar data, DataFrame for tick data
    store = MarketDataStore()
    store.load_symbol(symbol, m1_df)
    n = len(m1_df)
    date_range = f"{m1_df.index[0]} → {m1_df.index[-1]}"
    tick_info = f", {len(raw_ticks):,} ticks" if raw_ticks is not None else ""
    print(f"  Loaded {n:,} candles ({date_range}{tick_info})")

    _store_cache[key] = (store, n, date_range, raw_ticks)
    return store, n, raw_ticks


print("Loading strategies...")
_populate_registry()
print(f"  Found {len(STRATEGY_REGISTRY)} strategies")


class BacktestRequest(BaseModel):
    strategies: list[str]
    lookback: int = 5
    threshold: int = 2
    csv_file: str = "DAT_ASCII_USDJPY_M1_202605.csv"
    symbol: str = "USDJPY"
    use_sr: bool = False
    spread_pips: float = 0.5
    min_rr: float = 1.0


class SaveBucketRequest(BaseModel):
    name: str
    strategies: list[str]
    lookback: int = 5
    threshold: int = 2
    csv_file: str = "DAT_ASCII_USDJPY_M1_202605.csv"
    symbol: str = "USDJPY"
    use_sr: bool = False
    backtest_result: dict | None = None


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/strategies")
async def list_strategies():
    return {
        "strategies": [
            {"name": name, "timeframes": info["timeframes"]}
            for name, info in sorted(STRATEGY_REGISTRY.items())
        ]
    }


@app.get("/files")
async def list_files():
    """List available data files (CSV + Parquet)."""
    files = []
    for f in sorted(DATA_DIR.glob("*.*")):
        if f.suffix.lower() not in (".csv", ".parquet", ".pq", ".parq"):
            continue
        size_mb = f.stat().st_size / 1024 / 1024
        files.append({
            "name": f.name,
            "size_mb": round(size_mb, 1),
            "path": str(f.relative_to(DATA_DIR)),
        })
    return {"files": files, "data_dir": str(DATA_DIR)}


@app.post("/run")
async def run_backtest(req: BacktestRequest):
    invalid = [s for s in req.strategies if s not in STRATEGY_REGISTRY]
    if invalid:
        return {"error": f"Unknown strategies: {invalid}"}
    if req.threshold > len(req.strategies):
        return {"error": f"Threshold {req.threshold} exceeds strategy count {len(req.strategies)}"}

    try:
        from core.trade_engine import TradeConfig, TradeEngine
        from core.signal_engine import SignalEngine
        from core.engine import CandleArrays
        from core.events import BarEvent

        store, n_candles, raw_ticks = get_store(req.csv_file, req.symbol)

        # Build arrays
        m1_df = store.get_data(req.symbol, "M1")
        arrays = CandleArrays.from_dataframe(m1_df)
        tf_arrays = {}
        for tf in store.available_timeframes(req.symbol):
            if tf == "M1":
                continue
            try:
                tf_arrays[tf] = CandleArrays.from_dataframe(store.get_data(req.symbol, tf))
            except Exception:
                pass

        # Create engines
        config = TradeConfig(
            symbol=req.symbol,
            pip_value=0.01,
            lot_size=0.01,
            initial_balance=10000.0,
            spread_pips=req.spread_pips,
            min_rr=req.min_rr,
            use_sr=req.use_sr,
        )
        signal_engine = SignalEngine(
            strategy_names=req.strategies,
            lookback=req.lookback,
            threshold=req.threshold,
        )
        trade_engine = TradeEngine(config)

        # Pre-compute indicators
        signal_engine.precompute(arrays, tf_arrays)

        # Run the backtest loop
        t0 = time.perf_counter()
        max_start = max(req.lookback, 100)
        for i in range(max_start, arrays.n):
            signals = signal_engine.evaluate(i, arrays, tf_arrays)
            for sig in signals:
                trade_engine.open(sig)
            bar_event = BarEvent(
                timestamp=arrays.timestamps[i],
                open=float(arrays.opens[i]),
                high=float(arrays.highs[i]),
                low=float(arrays.lows[i]),
                close=float(arrays.closes[i]),
                volume=float(arrays.volumes[i]),
                symbol=req.symbol,
            )
            trade_engine.on_bar(bar_event)

        # Force close any remaining open position
        if trade_engine.open_position:
            trade_engine.force_close(
                float(arrays.closes[arrays.n - 1]),
                pd.Timestamp(arrays.timestamps[arrays.n - 1]),
                "EOD",
            )
        elapsed = round(time.perf_counter() - t0, 1)

        trades = trade_engine.trades

        # Use unified get_stats() from TradeEngine
        result = trade_engine.get_stats()
        # Add extra stats the UI expects
        result["n_candles"] = n_candles
        result["spread_pips"] = req.spread_pips
        result["min_rr"] = req.min_rr
        result["elapsed_sec"] = elapsed

        # Extended stats not in get_stats()
        import statistics
        winning = [t for t in trades if t.pnl_pips > 0]
        losing = [t for t in trades if t.pnl_pips <= 0]
        win_pips = [t.pnl_pips for t in winning]
        lose_pips = [abs(t.pnl_pips) for t in losing]
        durations = [t.bars_held for t in trades]

        result["avg_win_pips"] = round(statistics.mean(win_pips), 2) if win_pips else 0
        result["avg_loss_pips"] = round(statistics.mean(lose_pips), 2) if lose_pips else 0
        result["median_win_pips"] = round(statistics.median(win_pips), 2) if win_pips else 0
        result["median_loss_pips"] = round(statistics.median(lose_pips), 2) if lose_pips else 0
        result["avg_bars_held"] = round(statistics.mean(durations), 1) if durations else 0

        # Build trade list using TradeRecord.to_dict()
        trade_list = [t.to_dict() for t in trades]

        return {
            "result": result,
            "trades": trade_list,
            "elapsed": elapsed,
            "config": {
                "strategies": req.strategies,
                "lookback": req.lookback,
                "threshold": req.threshold,
                "csv_file": req.csv_file,
                "symbol": req.symbol,
                "use_sr": req.use_sr,
            }
        }

    except Exception as exc:
        import traceback
        return {"error": str(exc), "traceback": traceback.format_exc()}


@app.get("/progress")
async def get_progress():
    """Return current backtest progress (polled by frontend during run)."""
    # Simplified: no real-time progress tracking yet
    return {"running": False, "pct": 100}


@app.get("/buckets")
async def list_buckets():
    """List all saved strategy buckets."""
    from backtest.buckets import StrategyBucket
    buckets = StrategyBucket.list_buckets()
    return {"buckets": buckets}


@app.post("/save-bucket")
async def save_bucket(req: SaveBucketRequest):
    """Save current config as a named strategy bucket."""
    from backtest.buckets import StrategyBucket
    bucket = StrategyBucket(
        name=req.name,
        strategies=req.strategies,
        lookback=req.lookback,
        threshold=req.threshold,
        csv_file=req.csv_file,
        symbol=req.symbol,
        use_sr=req.use_sr,
        backtest_result=req.backtest_result or {},
    )
    path = bucket.save()
    return {"ok": True, "path": str(path), "name": req.name}


@app.get("/load-bucket/{name}")
async def load_bucket(name: str):
    """Load a bucket by name and return its full config."""
    from backtest.buckets import StrategyBucket
    buckets_dir = Path(__file__).parent.parent.parent / "buckets"
    if not buckets_dir.exists():
        return {"error": f"Bucket '{name}' not found"}

    # Find the bucket file by name
    for f in buckets_dir.glob("*.json"):
        try:
            bucket = StrategyBucket.load(f)
            if bucket.name == name:
                return {"bucket": {
                    "name": bucket.name,
                    "strategies": bucket.strategies,
                    "lookback": bucket.lookback,
                    "threshold": bucket.threshold,
                    "csv_file": bucket.csv_file,
                    "symbol": bucket.symbol,
                    "use_sr": bucket.use_sr,
                    "backtest_result": bucket.backtest_result,
                }}
        except Exception:
            continue
    return {"error": f"Bucket '{name}' not found"}


def main():
    import uvicorn
    print("Starting RSFX Backtester UI on http://localhost:8502")
    uvicorn.run(app, host="0.0.0.0", port=8502)


if __name__ == "__main__":
    main()
