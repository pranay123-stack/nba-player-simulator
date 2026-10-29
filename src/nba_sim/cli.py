"""Command-line interface: `nba-sim --help`."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from nba_sim.config import PROJECT_ROOT, load_config, resolve, setup_logging

app = typer.Typer(add_completion=False, help="NBA player outcome simulation and chronological validation.")
log = logging.getLogger("nba_sim")

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: configs/default.yaml)")


def _cfg(config: Optional[Path], **overrides):
    setup_logging()
    return load_config(config, overrides or None)


def _fail(msg: str, code: int = 1):
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _load(cfg):
    from nba_sim.data.ingest import DataNotFoundError, load_dataset

    try:
        return load_dataset(resolve(cfg["data"]["processed_dir"]))
    except DataNotFoundError as e:
        _fail(str(e))


def _features(cfg):
    import pandas as pd

    p = resolve(cfg["data"]["processed_dir"]) / "features.parquet"
    if not p.exists():
        _fail(f"{p} not found; run `nba-sim build-dataset` first")
    return pd.read_parquet(p)


@app.command("inspect-data")
def inspect_data(data_path: Path = typer.Option(Path("data/raw"), help="directory with the raw CSVs"),
                 config: Optional[Path] = ConfigOpt):
    """Validate the raw box scores and print coverage (no files modified except the report)."""
    from nba_sim.data.ingest import DataNotFoundError, build_team_games, normalize, raw_files, read_raw
    from nba_sim.data.validate import validate

    cfg = _cfg(config)
    try:
        files = raw_files(resolve(data_path), cfg["data"]["include_playoffs"])
    except DataNotFoundError as e:
        _fail(str(e))
    pg = normalize(read_raw(files))
    tg = build_team_games(pg)
    rep = validate(pg, tg)
    out = resolve("outputs/data_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep.to_dict(), indent=1, default=str))
    typer.echo(rep.coverage.to_string())
    for k, v in rep.summary.items():
        typer.echo(f"  {k}: {v}")
    for w in rep.warnings:
        typer.secho(f"warning: {w}", fg=typer.colors.YELLOW)
    for e in rep.errors:
        typer.secho(f"ERROR: {e}", fg=typer.colors.RED)
    typer.echo(f"validation report -> {out}")
    if not rep.ok:
        raise typer.Exit(2)


@app.command("build-dataset")
def build_dataset(data_path: Path = typer.Option(Path("data/raw")), config: Optional[Path] = ConfigOpt):
    """Ingest raw CSVs, validate, write normalized tables and pregame features."""
    from nba_sim.data.ingest import DataNotFoundError, ingest, save_dataset
    from nba_sim.data.validate import validate
    from nba_sim.features.build import build_features

    cfg = _cfg(config)
    try:
        ds = ingest(resolve(data_path), cfg["data"]["include_playoffs"])
    except DataNotFoundError as e:
        _fail(str(e))
    rep = validate(ds.player_games, ds.team_games)
    if not rep.ok:
        _fail(f"validation failed: {rep.errors}")
    out = resolve(cfg["data"]["processed_dir"])
    paths = save_dataset(ds, out)
    feats = build_features(ds.player_games, ds.team_games, cfg)
    fp = out / "features.parquet"
    feats.to_parquet(fp, index=False)
    for k, p in {**paths, "features": fp}.items():
        typer.echo(f"{k:13s} -> {p}")
    typer.echo(f"dataset_id {ds.manifest['dataset_id']}: {ds.manifest['n_games']} games, "
               f"{ds.manifest['date_min']} .. {ds.manifest['date_max']}")


@app.command()
def backtest(start_season: str = typer.Option("2021-22"), end_season: str = typer.Option("2023-24"),
             models: str = typer.Option("A,B,C", help="comma-separated subset of A,B,C"),
             n_simulations: Optional[int] = typer.Option(None, help="draws per player-game (default: config)"),
             run_name: str = typer.Option("main", help="outputs/backtest/<run_name>"),
             max_windows: Optional[int] = typer.Option(None, help="limit rolling origins (smoke runs)"),
             config: Optional[Path] = ConfigOpt):
    """Rolling-origin backtest: freeze predictions per origin, then score against outcomes."""
    from nba_sim.evaluation.backtest import run_backtest, season_list

    cfg = _cfg(config)
    ds = _load(cfg)
    feats = _features(cfg)
    seasons = season_list(feats)
    if start_season not in seasons or end_season not in seasons:
        _fail(f"seasons must be within {seasons[0]}..{seasons[-1]} (available: {seasons})")
    eval_seasons = seasons[seasons.index(start_season): seasons.index(end_season) + 1]
    lookback = cfg["backtest"]["train_lookback_seasons"]
    if seasons.index(start_season) < 1:
        _fail(f"{start_season} has no earlier season to train on")
    out = resolve(f"outputs/backtest/{run_name}")
    log.info("evaluation seasons %s (training lookback %d seasons) -> %s", eval_seasons, lookback, out)
    man = run_backtest(cfg, feats, out, ds.manifest, seasons=eval_seasons, models=models.split(","),
                       n_sims=n_simulations, max_windows=max_windows)
    typer.echo(f"manifest -> {out / 'manifest.json'}")
    for k, p in man["score_files"].items():
        typer.echo(f"{k:9s} scores -> {p}")


@app.command()
def evaluate(run_name: str = typer.Option("main"), models: str = typer.Option("A,B,C"),
             config: Optional[Path] = ConfigOpt):
    """Re-score frozen predictions of an existing backtest run."""
    from nba_sim.evaluation.backtest import run_evaluation

    cfg = _cfg(config)
    paths = run_evaluation(cfg, _features(cfg), resolve(f"outputs/backtest/{run_name}"), models.split(","))
    for k, p in paths.items():
        typer.echo(f"{k:9s} scores -> {p}")


@app.command()
def simulate(home: str = typer.Option(..., help="home team tricode, e.g. BOS"),
             away: str = typer.Option(..., help="away team tricode, e.g. MIL"),
             date: str = typer.Option(..., help="game date YYYY-MM-DD; only data before it is used"),
             n_simulations: int = typer.Option(10000), seed: int = typer.Option(42),
             model: str = typer.Option("C", help="A, B or C"),
             exclude: str = typer.Option("", help="comma-separated player ids to treat as inactive"),
             config: Optional[Path] = ConfigOpt):
    """Simulate a matchup on a date from data strictly before that date (roster = recent players)."""
    import pandas as pd

    from nba_sim.reporting.matchup_demo import simulate_matchup

    cfg = _cfg(config)
    ds = _load(cfg)
    try:
        paths = simulate_matchup(ds, cfg, home, away, pd.Timestamp(date), n_simulations, seed, model,
                                 [int(x) for x in exclude.split(",") if x.strip()])
    except ValueError as e:
        _fail(str(e))
    for k, p in paths.items():
        typer.echo(f"{k:8s} -> {p}")


@app.command("demo-historical")
def demo_historical(game_id: Optional[int] = typer.Option(None, help="NBA game id; default: pregame selection rule"),
                    n_simulations: int = typer.Option(10000), seed: int = typer.Option(42),
                    backtest_run: str = typer.Option("main", help="backtest run used for calibration context"),
                    config: Optional[Path] = ConfigOpt):
    """Held-out historical matchup: freeze 10,000-draw forecasts per model, then compare to the box score."""
    from nba_sim.reporting.matchup_demo import run_demo

    cfg = _cfg(config)
    ds = _load(cfg)
    try:
        paths = run_demo(ds, cfg, game_id, n_simulations, seed, backtest_run)
    except ValueError as e:
        _fail(str(e))
    for k, p in paths.items():
        typer.echo(f"{k:10s} -> {p}")


@app.command()
def report(output: Path = typer.Option(Path("reports"), help="report directory"),
           run_name: str = typer.Option("main"), config: Optional[Path] = ConfigOpt):
    """Generate the validation report and research note from saved scores."""
    from nba_sim.reporting.validation_report import build_reports

    cfg = _cfg(config)
    run = resolve(f"outputs/backtest/{run_name}")
    if not (run / "scores" / "player_scores.parquet").exists():
        _fail(f"no scores in {run}; run `nba-sim backtest` first")
    paths = build_reports(cfg, run, resolve(output))
    for k, p in paths.items():
        typer.echo(f"{k:18s} -> {p}")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
