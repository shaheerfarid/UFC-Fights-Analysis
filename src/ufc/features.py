"""Point-in-time pre-fight features, expressed as fighter_1 minus fighter_2.

Every feature for a fight is computed only from fights on strictly earlier dates.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

FINISH_METHODS = {"KO/TKO", "Submission", "TKO - Doctor's Stoppage"}
DEFAULT_LAYOFF_DAYS = 180  # stand-in for "days since last fight" on a debut (flagged by is_debut)

# Pseudo-observations that pull small samples toward the league average at that date.
SHRINK_MINUTES = 10
SHRINK_ATTEMPTS = 5
SHRINK_FIGHTS = 3

# Per fighter-fight quantities that get accumulated over a career.
SUMS = [
    "fight", "win", "finish_win", "stat_minutes",
    "sig_landed", "sig_absorbed", "td_landed", "td_att", "td_absorbed", "opp_td_att",
]
FEATURES = [
    "age", "height", "reach",
    "sig_landed_pm", "sig_absorbed_pm", "td_acc", "td_def", "finish_rate",
    "win_streak", "days_since_last", "experience", "is_debut",
]
STATIC_FLAGS = ["age", "height", "reach"]


def load_inputs(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read (fights with event date, fight_stats, fighters) from the database."""
    fights = pd.read_sql_query(
        "SELECT f.*, e.date FROM fights f JOIN events e USING (event_id)", conn, parse_dates=["date"]
    )
    stats = pd.read_sql_query("SELECT * FROM fight_stats", conn)
    fighters = pd.read_sql_query("SELECT * FROM fighters", conn, parse_dates=["dob"])
    return fights, stats, fighters


def _long_table(fights: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter per fight (so two per fight), sorted by fighter then date.

    Holds what happened *in that fight*; it only becomes a feature for later fights.
    """
    per_fight = stats.groupby(["fight_id", "fighter"])[["sig_str_landed", "td_landed", "td_att"]]
    per_fight = per_fight.sum(min_count=1).reset_index()

    base = ["fight_id", "date", "outcome", "winner", "method", "duration_sec"]
    sides = []
    for side, me, other in ((1, "fighter_1", "fighter_2"), (2, "fighter_2", "fighter_1")):
        d = fights[base].copy()
        d["side"], d["fighter"], d["opponent"] = side, fights[me], fights[other]
        d["fighter_id"] = fights[f"{me}_id"]
        sides.append(d)
    long = pd.concat(sides, ignore_index=True)
    # ambiguous names have no id; fall back to the name (merges namesakes, rare)
    long["key"] = long["fighter_id"].fillna(long["fighter"])

    own = per_fight.rename(columns={"sig_str_landed": "sig_landed"})
    opp = per_fight.rename(columns={
        "fighter": "opponent", "sig_str_landed": "sig_absorbed",
        "td_landed": "td_absorbed", "td_att": "opp_td_att",
    })
    long = long.merge(own, on=["fight_id", "fighter"], how="left")
    long = long.merge(opp, on=["fight_id", "opponent"], how="left")

    won = long["winner"].eq(long["fighter"]).fillna(False).astype(bool)
    long["result"] = np.select(
        [won, long["winner"].notna(), long["outcome"] == "NC/NC"], ["W", "L", "NC"], default="D"
    )
    long["fight"] = 1
    long["win"] = won.astype(int)
    long["finish_win"] = (won & long["method"].isin(FINISH_METHODS)).astype(int)
    # minutes only count where we have strike data, so rates aren't diluted by missing stats
    long["stat_minutes"] = (long["duration_sec"] / 60).where(long["sig_landed"].notna(), 0)
    long[SUMS] = long[SUMS].fillna(0)
    # upcoming matchups have no result yet: they must not count toward anyone's history
    played = long["outcome"].notna()
    long.loc[~played, SUMS] = 0
    long["result"] = long["result"].where(played, "NC")  # NC leaves the streak unchanged
    return long.sort_values(["key", "date", "fight_id"]).reset_index(drop=True)


def _streak_after(long: pd.DataFrame) -> pd.Series:
    """Win streak right after each fight: win +1, loss/draw reset, no-contest unchanged."""
    out = np.zeros(len(long), dtype=int)
    prev_key, streak = None, 0
    for i, (key, result) in enumerate(zip(long["key"], long["result"])):
        if key != prev_key:
            prev_key, streak = key, 0
        streak = streak + 1 if result == "W" else streak if result == "NC" else 0
        out[i] = streak
    return pd.Series(out, index=long.index)


def _prior_state(long: pd.DataFrame) -> pd.DataFrame:
    """Each fighter's career totals strictly before each fight's date.

    "Before the previous row" is not enough: a fighter can fight twice on one
    date (early tournaments). So every row in a (fighter, date) group takes the
    state of the group's first row, which only includes earlier dates.
    """
    by_key = long.groupby("key")
    before = by_key[SUMS].cumsum() - long[SUMS]  # includes earlier rows, excludes this one
    before["win_streak"] = _streak_after(long).groupby(long["key"]).shift(1).fillna(0)
    # last date actually fought, skipping result-less (upcoming) rows
    played_date = long["date"].where(long["outcome"].notna())
    before["last_date"] = played_date.groupby(long["key"]).shift(1).groupby(long["key"]).ffill()

    group = long.groupby(["key", "date"], sort=False).ngroup()
    is_first = ~group.duplicated()
    first_rows = before[is_first].set_index(group[is_first])
    return first_rows.loc[group].set_axis(long.index)


def _fighter_features(long: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Pre-fight features for every fighter-fight row of `long`."""
    prior = _prior_state(long)

    # league-wide totals before each date, used as the baseline for shrinkage (also leak-free)
    daily = long.groupby("date")[SUMS].sum().sort_index().cumsum().shift(1)
    league = daily.reindex(long["date"]).set_axis(long.index)

    def rate(num: str, den: str, k: float) -> pd.Series:
        baseline = (league[num] / league[den]).fillna(0)
        return (prior[num] + k * baseline) / (prior[den] + k)

    f = pd.DataFrame(index=long.index)
    f["sig_landed_pm"] = rate("sig_landed", "stat_minutes", SHRINK_MINUTES)
    f["sig_absorbed_pm"] = rate("sig_absorbed", "stat_minutes", SHRINK_MINUTES)
    f["td_acc"] = rate("td_landed", "td_att", SHRINK_ATTEMPTS)
    f["td_def"] = 1 - rate("td_absorbed", "opp_td_att", SHRINK_ATTEMPTS)
    f["finish_rate"] = rate("finish_win", "fight", SHRINK_FIGHTS)
    f["win_streak"] = prior["win_streak"]
    f["experience"] = prior["fight"]
    f["is_debut"] = (prior["fight"] == 0).astype(int)
    f["days_since_last"] = (long["date"] - prior["last_date"]).dt.days.fillna(DEFAULT_LAYOFF_DAYS)

    profile = fighters.set_index("fighter_id")
    f["age"] = (long["date"] - long["fighter_id"].map(profile["dob"])).dt.days / 365.25
    f["height"] = long["fighter_id"].map(profile["height_in"]).astype(float)
    f["reach"] = long["fighter_id"].map(profile["reach_in"]).astype(float)

    f["fight_id"], f["side"] = long["fight_id"], long["side"]
    return f


def _diff_table(fights: pd.DataFrame, stats: pd.DataFrame, fighters: pd.DataFrame,
                with_sides: bool = False) -> pd.DataFrame:
    """Every fight's <feature>_diff columns (fighter_1 - fighter_2) and <name>_unknown flags.

    Missing age/height/reach make that diff 0 and set the flag (symmetric).
    with_sides=True also adds each fighter's raw values as a_<feature> / b_<feature>.
    """
    feats = _fighter_features(_long_table(fights, stats), fighters)
    a = feats[feats["side"] == 1].set_index("fight_id")
    b = feats[feats["side"] == 2].set_index("fight_id").reindex(a.index)

    diff = (a[FEATURES] - b[FEATURES]).add_suffix("_diff")
    flags = pd.DataFrame({f"{c}_unknown": diff[f"{c}_diff"].isna().astype(int) for c in STATIC_FLAGS})
    diff = diff.fillna(0)

    meta = fights.set_index("fight_id")[["date", "fighter_1", "fighter_2", "winner"]]
    out = meta.join(diff).join(flags)
    if with_sides:
        out = out.join(a[FEATURES].add_prefix("a_")).join(b[FEATURES].add_prefix("b_"))
    return out


def build_features(fights: pd.DataFrame, stats: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """One row per decided fight: <feature>_diff columns, <name>_unknown flags and target y.

    y is 1 if fighter_1 won. Draws and no-contests are dropped.
    """
    out = _diff_table(fights, stats, fighters)
    out["y"] = (out["winner"] == out["fighter_1"]).astype(int)
    out = out[out["winner"].notna()].drop(columns="winner")
    return out.sort_values("date").reset_index()


def matchup_features(fights: pd.DataFrame, stats: pd.DataFrame, fighters: pd.DataFrame,
                     matchups: pd.DataFrame) -> pd.DataFrame:
    """Pre-fight features for fights that haven't happened yet.

    `matchups` needs fighter_1, fighter_2, fighter_1_id, fighter_2_id (ids may be
    NA for fighters not in the db, e.g. debutants) and date. Each matchup is added
    as a result-less fight on that date and run through the same pipeline as the
    training data, so it only sees fights strictly before its date. Returns diffs,
    flags and each side's raw values (a_*, b_*), in the order of `matchups`.
    """
    upcoming = matchups[["fighter_1", "fighter_2", "fighter_1_id", "fighter_2_id", "date"]].assign(
        fight_id=[f"upcoming-{i}" for i in range(len(matchups))],
        winner=None, outcome=None, method=None, duration_sec=np.nan,
    )
    upcoming["date"] = pd.to_datetime(upcoming["date"])
    table = _diff_table(pd.concat([fights, upcoming], ignore_index=True), stats, fighters, with_sides=True)
    return table.loc[upcoming["fight_id"]].drop(columns="winner").reset_index()


def random_corner_swap(df: pd.DataFrame, seed: int = 0, p: float = 0.5) -> pd.DataFrame:
    """Swap fighter_1/fighter_2 in a random share p of rows.

    Swapped rows get every *_diff negated and y inverted; the *_unknown flags are
    symmetric so they stay. Rows are independent, so this can't leak across the
    time split; apply it after splitting.
    """
    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < p
    out = df.copy()
    diff_cols = [c for c in out.columns if c.endswith("_diff")]
    out.loc[flip, diff_cols] = -out.loc[flip, diff_cols]
    out.loc[flip, "y"] = 1 - out.loc[flip, "y"]
    out.loc[flip, ["fighter_1", "fighter_2"]] = out.loc[flip, ["fighter_2", "fighter_1"]].to_numpy()
    out["swapped"] = flip
    return out
