"""One-click refresh: download -> clean -> load new events -> rebuild features -> retrain."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from . import db, model, parse
from .features import build_features, load_inputs
from .http import RAW_DIR, download_tables

DATA_DIR = RAW_DIR.parent
ARTIFACT_PATH = DATA_DIR / "models.joblib"
CALIBRATION_PLOT = DATA_DIR / "calibration.png"


def refresh_database(conn: sqlite3.Connection) -> dict[str, int]:
    """Download the latest CSVs and insert only events the db doesn't have yet."""
    download_tables()
    raw = {
        name: pd.read_csv(RAW_DIR / f"ufc_{name}.csv")
        for name in ("event_details", "fight_results", "fight_stats", "fighter_tott")
    }
    events = parse.clean_events(raw["event_details"])
    fighters = parse.clean_fighters(raw["fighter_tott"])
    fights = parse.clean_fights(raw["fight_results"], events, fighters)
    stats = parse.clean_fight_stats(raw["fight_stats"], fights)

    db.init_db(conn)
    return db.load_tables(conn, {"events": events, "fighters": fighters, "fights": fights, "fight_stats": stats})


def retrain(conn: sqlite3.Connection, path: Path = ARTIFACT_PATH) -> dict:
    """Rebuild features, evaluate on the last year, refit on everything and save."""
    features = build_features(*load_inputs(conn))
    metrics, _ = model.run(features, plot_path=CALIBRATION_PLOT)
    artifacts = {
        "models": model.train_final(features),
        "metrics": metrics,
        "split": metrics.attrs["split"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_fights": len(features),
        "latest_fight": features["date"].max(),
    }
    joblib.dump(artifacts, path)
    return artifacts


_ready_lock = threading.Lock()


def ensure_ready() -> None:
    """Build the db and model if missing (first launch on a new host); one visitor at a time."""
    with _ready_lock:
        if not ARTIFACT_PATH.exists():
            update_all()


def update_all(conn: sqlite3.Connection | None = None) -> tuple[dict[str, int], dict]:
    """Refresh the db then retrain; returns (rows added per table, saved artifacts)."""
    conn = conn or db.connect()
    added = refresh_database(conn)
    return added, retrain(conn)


if __name__ == "__main__":
    added, artifacts = update_all()
    print("added:", added)
    print(artifacts["split"])
    print(artifacts["metrics"].round(4).to_string())
