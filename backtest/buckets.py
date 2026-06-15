"""
backtest/buckets.py
-------------------
Strategy bucket system — named groups of strategies with full config.

A bucket captures a complete, tested configuration:
  - Strategy names
  - S/R toggle
  - Lookback window
  - Confluence threshold
  - Backtest results (optional)

Shared between backtester and replay UI.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# Default buckets directory
BUCKETS_DIR = Path(__file__).parent.parent / "buckets"


@dataclass
class StrategyBucket:
    """A named group of strategies with full configuration."""
    name: str
    strategies: list[str]
    use_sr: bool = False
    lookback: int = 5
    threshold: int = 2
    csv_file: str = "DAT_ASCII_USDJPY_M1_202605.csv"
    symbol: str = "USDJPY"
    description: str = ""
    created: str = ""
    backtest_result: dict = field(default_factory=dict)

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save bucket to JSON file.

        If path is None, saves to buckets/<slugified_name>.json
        Returns the path where saved.
        """
        if path is None:
            slug = self.name.lower().replace(" ", "_").replace("/", "_")
            slug = "".join(c for c in slug if c.isalnum() or c == "_")
            path = BUCKETS_DIR / f"{slug}.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        # Remove empty backtest_result
        if not data.get("backtest_result"):
            del data["backtest_result"]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path) -> StrategyBucket:
        """Load bucket from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def list_buckets(cls, buckets_dir: Optional[Path] = None) -> list[dict]:
        """
        List all available buckets.

        Returns list of dicts with name, path, strategy_count, has_results.
        """
        if buckets_dir is None:
            buckets_dir = BUCKETS_DIR

        if not buckets_dir.exists():
            return []

        buckets = []
        for f in sorted(buckets_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                buckets.append({
                    "name": data.get("name", f.stem),
                    "path": str(f),
                    "strategy_count": len(data.get("strategies", [])),
                    "use_sr": data.get("use_sr", False),
                    "threshold": data.get("threshold", 2),
                    "csv_file": data.get("csv_file", "DAT_ASCII_USDJPY_M1_202605.csv"),
                    "symbol": data.get("symbol", "USDJPY"),
                    "has_results": bool(data.get("backtest_result")),
                    "backtest_result": data.get("backtest_result", {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return buckets

    @classmethod
    def from_strategies(
        cls,
        name: str,
        strategies: list[str],
        use_sr: bool = False,
        lookback: int = 5,
        threshold: int = 2,
        csv_file: str = "DAT_ASCII_USDJPY_M1_202605.csv",
        symbol: str = "USDJPY",
        description: str = "",
    ) -> StrategyBucket:
        """Create a bucket from strategy names."""
        import time
        return cls(
            name=name,
            strategies=strategies,
            use_sr=use_sr,
            lookback=lookback,
            threshold=threshold,
            csv_file=csv_file,
            symbol=symbol,
            description=description,
            created=time.strftime("%Y-%m-%d"),
        )

    def to_confluence_args(self) -> dict:
        """Convert bucket to confluence_backtest CLI args."""
        return {
            "strategies": ",".join(self.strategies),
            "lookback": self.lookback,
            "threshold": self.threshold,
            "use_sr": self.use_sr,
            "csv_file": self.csv_file,
            "symbol": self.symbol,
        }

    def __str__(self) -> str:
        sr_tag = " +S/R" if self.use_sr else ""
        return f"{self.name} ({len(self.strategies)} strategies, lb={self.lookback}, t={self.threshold}{sr_tag})"
