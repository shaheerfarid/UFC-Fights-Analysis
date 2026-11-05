"""SQLite schema and loading, with incremental updates by event date."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "ufc.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    height_in   REAL,
    reach_in    REAL,
    weight_lbs  INTEGER,
    stance      TEXT,
    dob         TEXT                      -- ISO date YYYY-MM-DD
);

CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    date        TEXT NOT NULL,            -- ISO date, so MAX(date) sorts correctly
    location    TEXT
);

CREATE TABLE IF NOT EXISTS fights (
    fight_id      TEXT PRIMARY KEY,
    event_id      TEXT NOT NULL REFERENCES events(event_id),
    fighter_1     TEXT NOT NULL,
    fighter_2     TEXT NOT NULL,
    fighter_1_id  TEXT REFERENCES fighters(fighter_id),   -- NULL if the name is ambiguous
    fighter_2_id  TEXT REFERENCES fighters(fighter_id),
    winner        TEXT,                   -- NULL for draws / no contests
    outcome       TEXT,
    weight_class  TEXT,
    method        TEXT,
    round         INTEGER,
    end_time_sec  INTEGER,
    duration_sec  INTEGER,
    time_format   TEXT,
    referee       TEXT
);
CREATE INDEX IF NOT EXISTS idx_fights_event ON fights(event_id);

CREATE TABLE IF NOT EXISTS fight_stats (
    fight_id    TEXT NOT NULL REFERENCES fights(fight_id),
    round       INTEGER NOT NULL,
    fighter     TEXT NOT NULL,
    kd          INTEGER,
    sub_att     INTEGER,
    rev         INTEGER,
    ctrl_sec    INTEGER,
    sig_str_landed INTEGER, sig_str_att INTEGER,
    total_str_landed INTEGER, total_str_att INTEGER,
    td_landed INTEGER, td_att INTEGER,
    head_landed INTEGER, head_att INTEGER,
    body_landed INTEGER, body_att INTEGER,
    leg_landed INTEGER, leg_att INTEGER,
    distance_landed INTEGER, distance_att INTEGER,
    clinch_landed INTEGER, clinch_att INTEGER,
    ground_landed INTEGER, ground_att INTEGER,
    sig_str_pct REAL,
    td_pct      REAL,
    PRIMARY KEY (fight_id, round, fighter)
);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open the database (creating the folder if needed) with foreign keys enforced."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't exist yet."""
    conn.executescript(SCHEMA)


def latest_event_date(conn: sqlite3.Connection) -> str | None:
    """Newest event date in the database as 'YYYY-MM-DD', or None if it's empty."""
    return conn.execute("SELECT MAX(date) FROM events").fetchone()[0]


def _rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """DataFrame -> plain Python tuples: dates as ISO strings, NA/NaN/NaT as None."""
    out = df[columns].copy()
    for col in out.select_dtypes("datetime").columns:
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.astype(object).where(out.notna(), None)
    return list(out.itertuples(index=False, name=None))


def insert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame, replace: bool = False) -> int:
    """Insert df into table using the table's own columns; return rows written.

    Existing primary keys are skipped, or overwritten when replace=True.
    Does not commit; call inside `with conn:`.
    """
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    sql = f"{verb} INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
    before = conn.total_changes
    conn.executemany(sql, _rows(df, columns))
    return conn.total_changes - before


def load_tables(conn: sqlite3.Connection, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Load cleaned tables ('fighters', 'events', 'fights', 'fight_stats') incrementally.

    Only events on or after the latest stored date that aren't already stored are
    added, along with their fights and stats. Fighter profiles are refreshed every
    time. Everything happens in one transaction, so a failure leaves the db unchanged.
    """
    latest = latest_event_date(conn)
    known = {row[0] for row in conn.execute("SELECT event_id FROM events")}

    events = tables["events"]
    new_events = events[~events["event_id"].isin(known)]
    if latest is not None:
        new_events = new_events[new_events["date"] >= pd.Timestamp(latest)]

    fights = tables["fights"]
    new_fights = fights[fights["event_id"].isin(new_events["event_id"])]
    stats = tables["fight_stats"]
    new_stats = stats[stats["fight_id"].isin(new_fights["fight_id"])]

    with conn:
        return {
            "fighters": insert_df(conn, "fighters", tables["fighters"], replace=True),
            "events": insert_df(conn, "events", new_events),
            "fights": insert_df(conn, "fights", new_fights),
            "fight_stats": insert_df(conn, "fight_stats", new_stats),
        }
