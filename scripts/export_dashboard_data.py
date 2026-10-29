"""Copy the small result files the dashboard needs into app/data/ (committed, used by hosted deploys).

Run after `nba-sim backtest`, `nba-sim report` and `nba-sim demo-historical`:
    .venv/bin/python scripts/export_dashboard_data.py
Only aggregate backtest summaries and the single demo game's frozen predictions are copied; raw data,
per-game scores and the processed dataset stay local.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "backtest" / "main"
DEMO = ROOT / "outputs" / "demo"
DEST = ROOT / "app" / "data"

BACKTEST_FILES = ["summary_overall.csv", "summary_team.csv", "summary_season.csv", "summary_joint.csv",
                  "summary_threshold.csv", "comparisons.csv", "manifest.json"]
DEMO_PATTERNS = ["comparison.csv", "team_points.csv", "manifest.json", "frozen_predictions.json",
                 "predictions_*.npz", "predictions_*.rows.parquet", "predictions_*.meta.json"]


def main() -> int:
    missing = [f for f in BACKTEST_FILES if not (RUN / f).exists()]
    if missing:
        print(f"error: missing backtest outputs {missing}; run `nba-sim backtest` and `nba-sim report`", file=sys.stderr)
        return 1
    games = sorted(p for p in DEMO.glob("*") if (p / "comparison.csv").exists()) if DEMO.exists() else []
    if not games:
        print("error: no demo outputs; run `nba-sim demo-historical`", file=sys.stderr)
        return 1
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "backtest").mkdir(parents=True)
    for f in BACKTEST_FILES:
        shutil.copy2(RUN / f, DEST / "backtest" / f)
    for g in games:
        out = DEST / "demo" / g.name
        out.mkdir(parents=True)
        for pat in DEMO_PATTERNS:
            for f in g.glob(pat):
                shutil.copy2(f, out / f.name)
    size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"exported dashboard data -> {DEST} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
