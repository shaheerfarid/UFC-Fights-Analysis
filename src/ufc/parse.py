"""Clean the raw ufcstats tables into typed frames ready for the database."""
from __future__ import annotations

import re

import pandas as pd

MISSING = {"--", "---", ""}
DEFAULT_ROUND_MINUTES = 5

# raw "x of y" column -> prefix of the <prefix>_landed / <prefix>_att columns
OF_COLUMNS = {
    "SIG.STR.": "sig_str",
    "TOTAL STR.": "total_str",
    "TD": "td",
    "HEAD": "head",
    "BODY": "body",
    "LEG": "leg",
    "DISTANCE": "distance",
    "CLINCH": "clinch",
    "GROUND": "ground",
}


# ---------- generic helpers ----------

def to_na(s: pd.Series) -> pd.Series:
    """Strip whitespace and turn '--', '---' and empty strings into NA."""
    s = s.astype("string").str.strip()
    return s.mask(s.isin(MISSING))


def id_from_url(s: pd.Series) -> pd.Series:
    """Last path segment of a ufcstats URL, e.g. '.../event-details/55e9...' -> '55e9...'."""
    return s.str.rstrip("/").str.rsplit("/", n=1).str[-1]


def parse_date(s: pd.Series, fmt: str = "%B %d, %Y") -> pd.Series:
    """Parse date strings to datetime64; unparseable or missing values become NaT."""
    return pd.to_datetime(to_na(s), format=fmt, errors="coerce")


def split_of(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split '9 of 37' into (landed, attempted) as nullable ints."""
    parts = to_na(s).str.extract(r"^(\d+)\s+of\s+(\d+)$")
    return (
        pd.to_numeric(parts[0]).astype("Int64"),
        pd.to_numeric(parts[1]).astype("Int64"),
    )


def safe_pct(landed: pd.Series, attempted: pd.Series) -> pd.Series:
    """landed / attempted as a float, NA (not a ZeroDivisionError) when attempted is 0 or NA."""
    return (landed / attempted.where(attempted > 0)).astype("Float64")


def parse_height(s: pd.Series) -> pd.Series:
    """Convert 5' 11" to total inches (71.0)."""
    parts = to_na(s).str.extract(r"^(\d+)'\s*(\d+)\"$")
    return (pd.to_numeric(parts[0]) * 12 + pd.to_numeric(parts[1])).astype("Float64")


def parse_reach(s: pd.Series) -> pd.Series:
    """Convert 66" to inches (66.0)."""
    return pd.to_numeric(to_na(s).str.extract(r"^(\d+(?:\.\d+)?)\"?$")[0]).astype("Float64")


def parse_mmss(s: pd.Series) -> pd.Series:
    """Convert '2:32' to seconds (152)."""
    parts = to_na(s).str.extract(r"^(\d+):(\d{2})$")
    return (pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1])).astype("Int64")


def _scheduled_minutes(time_format: str | None) -> list[int]:
    """'3 Rnd + OT (5-5-5-5)' -> [5, 5, 5, 5]; [] when the format has no round lengths."""
    m = re.search(r"\(([\d-]+)\)", time_format or "")
    return [int(x) for x in m.group(1).split("-")] if m else []


def fight_duration_sec(round_: int, end_sec: int, time_format: str | None) -> int | None:
    """Total fight time: full length of all earlier rounds plus the time in the last one."""
    if pd.isna(round_) or pd.isna(end_sec):
        return None
    mins = _scheduled_minutes(time_format)
    earlier = sum(mins[i] if i < len(mins) else DEFAULT_ROUND_MINUTES for i in range(int(round_) - 1))
    return earlier * 60 + int(end_sec)


# ---------- table cleaners ----------

def clean_events(raw: pd.DataFrame) -> pd.DataFrame:
    """event_id, name, date (datetime), location."""
    out = pd.DataFrame({
        "event_id": id_from_url(raw["URL"]),
        "name": raw["EVENT"].str.strip(),
        "date": parse_date(raw["DATE"]),
        "location": to_na(raw["LOCATION"]),
    })
    return out.dropna(subset=["date"]).drop_duplicates("event_id").reset_index(drop=True)


def clean_fighters(tott: pd.DataFrame) -> pd.DataFrame:
    """fighter_id, name, height_in, reach_in, weight_lbs, stance, dob (datetime)."""
    out = pd.DataFrame({
        "fighter_id": id_from_url(tott["URL"]),
        "name": tott["FIGHTER"].str.strip(),
        "height_in": parse_height(tott["HEIGHT"]),
        "reach_in": parse_reach(tott["REACH"]),
        "weight_lbs": pd.to_numeric(to_na(tott["WEIGHT"]).str.extract(r"^(\d+)")[0]).astype("Int64"),
        "stance": to_na(tott["STANCE"]),
        "dob": parse_date(tott["DOB"], "%b %d, %Y"),
    })
    return out.drop_duplicates("fighter_id").reset_index(drop=True)


def clean_fights(results: pd.DataFrame, events: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """One row per fight.

    Drops rows whose event name isn't in `events` (renamed events leave stale
    duplicates of the same fight URL). Fighter ids are filled only when the name
    is unique in `fighters`, otherwise NA. Names are stripped because the raw
    files contain trailing spaces. `event_name` and `bout` are kept so
    fight stats can be matched back to a fight_id.
    """
    event_ids = events.set_index("name")["event_id"]
    results = results.assign(EVENT=results["EVENT"].str.strip(), BOUT=results["BOUT"].str.strip())
    df = results.assign(event_id=results["EVENT"].map(event_ids))
    df = df.dropna(subset=["event_id"]).drop_duplicates("URL").reset_index(drop=True)

    names = df["BOUT"].str.split(" vs. ", n=1, expand=True)
    unique_ids = fighters.drop_duplicates("name", keep=False).set_index("name")["fighter_id"]
    end_sec = parse_mmss(df["TIME"])

    out = pd.DataFrame({
        "fight_id": id_from_url(df["URL"]),
        "event_id": df["event_id"],
        "event_name": df["EVENT"],
        "bout": df["BOUT"],
        "fighter_1": names[0],
        "fighter_2": names[1],
        "fighter_1_id": names[0].map(unique_ids),
        "fighter_2_id": names[1].map(unique_ids),
        # OUTCOME is from the first-listed fighter's view: W/L = fighter_1 won
        "winner": names[0].where(df["OUTCOME"] == "W/L", names[1].where(df["OUTCOME"] == "L/W")),
        "outcome": df["OUTCOME"],
        "weight_class": to_na(df["WEIGHTCLASS"]),
        "method": to_na(df["METHOD"]),
        "round": df["ROUND"].astype("Int64"),
        "end_time_sec": end_sec,
        "duration_sec": [
            fight_duration_sec(r, t, f) for r, t, f in zip(df["ROUND"], end_sec, df["TIME FORMAT"])
        ],
        "time_format": to_na(df["TIME FORMAT"]),
        "referee": to_na(df["REFEREE"]),
    })
    out["duration_sec"] = out["duration_sec"].astype("Int64")
    return out


def clean_fight_stats(stats: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter per round, with numeric columns.

    Matched to fights on (event name, bout). Bouts listed twice in one event are
    ambiguous and dropped, as are the empty rows in the raw file.
    """
    keys = fights[["fight_id", "event_name", "bout"]].drop_duplicates(["event_name", "bout"], keep=False)
    stats = stats.assign(
        EVENT=stats["EVENT"].str.strip(), BOUT=stats["BOUT"].str.strip(), FIGHTER=stats["FIGHTER"].str.strip()
    )
    df = stats.dropna(subset=["ROUND", "FIGHTER"]).merge(
        keys, left_on=["EVENT", "BOUT"], right_on=["event_name", "bout"],
        how="inner", validate="many_to_one",
    ).reset_index(drop=True)

    out = pd.DataFrame({
        "fight_id": df["fight_id"],
        "round": pd.to_numeric(df["ROUND"].str.extract(r"(\d+)")[0]).astype("Int64"),
        "fighter": df["FIGHTER"],
        "kd": df["KD"].astype("Int64"),
        "sub_att": df["SUB.ATT"].astype("Int64"),
        "rev": df["REV."].astype("Int64"),
        "ctrl_sec": parse_mmss(df["CTRL"]),
    })
    for raw_col, prefix in OF_COLUMNS.items():
        out[f"{prefix}_landed"], out[f"{prefix}_att"] = split_of(df[raw_col])
    out["sig_str_pct"] = safe_pct(out["sig_str_landed"], out["sig_str_att"])
    out["td_pct"] = safe_pct(out["td_landed"], out["td_att"])
    return out
