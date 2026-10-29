"""Reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from nba_sim.config import PROJECT_ROOT


def code_fingerprint() -> dict:
    """SHA-256 over the package source and config files, plus git state when available."""
    h = hashlib.sha256()
    files = sorted((PROJECT_ROOT / "src").rglob("*.py")) + sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
    for f in files:
        h.update(str(f.relative_to(PROJECT_ROOT)).encode())
        h.update(f.read_bytes())
    out = {"source_sha256": h.hexdigest(), "n_files": len(files)}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, timeout=5)
        out["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                               text=True, timeout=5)
        out["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        out["git_commit"] = None
    return out


def write_manifest(path: Path, cfg: dict, dataset_manifest: dict, extra: dict | None = None) -> dict:
    manifest = {
        "written_utc": str(pd.Timestamp.now(tz="UTC")),
        "config": {k: v for k, v in cfg.items()},
        "dataset": dataset_manifest,
        "code": code_fingerprint(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "platform": platform.platform()},
    }
    if extra:
        manifest.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, default=str))
    return manifest
