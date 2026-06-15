"""
confluence_ui/server.py
-----------------------
Minimal FastAPI backend for the confluence backtester UI.

Usage:
    cd /home/rudi/RSFX
    python3 confluence_ui/server.py

Then open http://localhost:8502
"""

import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data_loader import HistDataAdapter
from core.market_data_store import MarketDataStore
from detectors.strategies.registry import STRATEGY_REGISTRY, _populate_registry

app = FastAPI(title="RSFX Confluence Backtester")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Cache: csv_path -> (MarketDataStore, candle_count, date_range)
_store_cache: dict[str, tuple[MarketDataStore, int, str]] = {}


def get_store(csv_path: str, symbol: str) -> tuple[MarketDataStore, int]:
    """Load CSV on demand, cache by path+symbol."""
    key = f"{csv_path}:{symbol}"
    if key in _store_cache:
        store, n, _ = _store_cache[key]
        return store, n

    full_path = csv_path
    if not Path(full_path).is_absolute():
        full_path = str(DATA_DIR / csv_path)

    print(f"Loading {full_path}...")
    m1_df = HistDataAdapter().load(full_path)
    store = MarketDataStore()
    store.load_symbol(symbol, m1_df)
    n = len(m1_df)
    date_range = f"{m1_df.index[0]} → {m1_df.index[-1]}"
    print(f"  Loaded {n:,} candles ({date_range})")

    _store_cache[key] = (store, n, date_range)
    return store, n


print("Loading strategies...")
_populate_registry()
print(f"  Found {len(STRATEGY_REGISTRY)} strategies")


class BacktestRequest(BaseModel):
    strategies: list[str]
    lookback: int = 5
    threshold: int = 2
    csv_file: str = "DAT_ASCII_USDJPY_M1_202605.csv"
    symbol: str = "USDJPY"


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
    """List available CSV data files."""
    files = []
    for f in sorted(DATA_DIR.glob("*.csv")):
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
        store, n_candles = get_store(req.csv_file, req.symbol)

        from backtest.confluence import ConfluenceEngine

        engine = ConfluenceEngine(
            data_store=store,
            strategy_names=req.strategies,
            symbol=req.symbol,
            lookback=req.lookback,
            threshold=req.threshold,
        )

        t0 = time.perf_counter()
        trades = engine.run()
        elapsed = round(time.perf_counter() - t0, 1)

        total = len(trades)
        winning = [t for t in trades if t.pnl_pips > 0]
        losing = [t for t in trades if t.pnl_pips <= 0]
        total_pnl = sum(t.pnl_pips for t in trades)
        gross_profit = sum(t.pnl_pips for t in winning)
        gross_loss = abs(sum(t.pnl_pips for t in losing))
        win_rate = len(winning) / total * 100 if total else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = gross_profit / len(winning) if winning else 0
        avg_loss = gross_loss / len(losing) if losing else 0
        expectancy = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss

        result = {
            "total_trades": total,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": round(win_rate, 1),
            "total_pnl_pips": round(total_pnl, 1),
            "avg_pnl_pips": round(total_pnl / total, 1) if total else 0,
            "expectancy_pips": round(expectancy, 1),
            "profit_factor": round(pf, 2),
            "n_candles": n_candles,
        }

        trade_list = []
        for t in trades:
            trade_list.append({
                "entry_time": str(t.entry_time),
                "direction": t.direction,
                "entry_price": t.entry_price,
                "take_profit": t.take_profit,
                "stop_loss": t.stop_loss,
                "exit_price": t.exit_price,
                "exit_time": str(t.exit_time),
                "exit_reason": t.exit_reason,
                "pnl_pips": t.pnl_pips,
                "mae_pips": t.mae_pips,
                "mfe_pips": t.mfe_pips,
                "bars_held": t.bars_held,
                "strategies": t.strategy,
            })

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
            }
        }

    except Exception as exc:
        import traceback
        return {"error": str(exc), "traceback": traceback.format_exc()}


def main():
    import uvicorn
    print("Starting RSFX Confluence UI on http://localhost:8502")
    uvicorn.run(app, host="0.0.0.0", port=8502)


if __name__ == "__main__":
    main()
