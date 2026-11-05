"""Correctness checks for the pipeline.  Run from the project root:  python -m pytest -q

Needs data/ufc.db and data/models.joblib (run `python -m src.ufc.update` first).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ufc import db, parse
from src.ufc import predict as P
from src.ufc.features import build_features, load_inputs, matchup_features, random_corner_swap
from src.ufc.model import feature_columns, time_split

SAMPLES = Path(__file__).resolve().parents[1] / "samples"

pytestmark = pytest.mark.skipif(not db.DB_PATH.exists(), reason="run `python -m src.ufc.update` first")


@pytest.fixture(scope="module")
def data():
    return load_inputs(db.connect())


@pytest.fixture(scope="module")
def features(data):
    return build_features(*data)


# ---------- parsing ----------

def test_parse_helpers():
    s = pd.Series(["9 of 37", "---", "0 of 0"])
    landed, att = parse.split_of(s)
    assert landed.tolist()[0] == 9 and att.tolist()[0] == 37 and pd.isna(landed[1])
    assert pd.isna(parse.safe_pct(landed, att)[2])  # 0 attempts -> NA, not a crash
    assert parse.parse_height(pd.Series(["5' 11\"", "--"])).tolist()[0] == 71
    assert parse.parse_reach(pd.Series(['66"'])).tolist() == [66]
    assert parse.parse_mmss(pd.Series(["2:32"])).tolist() == [152]


# ---------- features ----------

def test_no_leakage(data, features):
    """Rewriting every result from a cutoff onward must not change any feature up to that day."""
    fights, stats, fighters = data
    cut = fights.loc[fights["date"] >= "2024-01-01", "date"].min()
    late = fights["date"] >= cut
    tampered = fights.copy()
    tampered.loc[late, "winner"] = tampered.loc[late, "fighter_2"]
    tampered.loc[late, ["method", "duration_sec"]] = ["KO/TKO", 7]
    stats2 = stats.copy()
    stats2.loc[stats2["fight_id"].isin(fights.loc[late, "fight_id"]), ["sig_str_landed", "td_landed"]] *= 5

    cols = feature_columns(features)
    before = features[features["date"] <= cut].set_index("fight_id")[cols].sort_index()
    after = build_features(tampered, stats2, fighters)
    after = after[after["date"] <= cut].set_index("fight_id")[cols].sort_index()
    pd.testing.assert_frame_equal(before, after)


def test_features_clean(features):
    cols = feature_columns(features)
    assert not features[cols].isna().any().any()
    assert np.isfinite(features[cols].to_numpy(float)).all()


def test_corner_swap_balances_target(features):
    swapped = random_corner_swap(features, seed=0)
    assert abs(swapped["y"].mean() - 0.5) < 0.03
    flipped = swapped[swapped["swapped"]].iloc[0]
    original = features.set_index("fight_id").loc[flipped["fight_id"]]
    assert flipped["age_diff"] == -original["age_diff"] and flipped["y"] == 1 - original["y"]


def test_time_split_is_strict(features):
    train, test = time_split(features)
    assert train["date"].max() < test["date"].min()


def test_upcoming_path_matches_training_rows(data, features):
    """A past fight fed through the upcoming-fight path must get identical features."""
    fights = data[0].set_index("fight_id")
    rows = features.tail(20)
    past = fights.loc[rows["fight_id"], ["fighter_1", "fighter_2", "fighter_1_id", "fighter_2_id", "date"]]
    upcoming = matchup_features(*data, past.reset_index(drop=True))
    cols = feature_columns(features)
    np.testing.assert_allclose(upcoming[cols].to_numpy(float), rows[cols].to_numpy(float))


# ---------- predictions ----------

@pytest.fixture(scope="module")
def models():
    artifacts = P.load_artifacts()
    if artifacts is None:
        pytest.skip("no trained model")
    return artifacts["models"]


def test_prediction_is_symmetric(data, models):
    card = P.parse_card_json((SAMPLES / "sample_card.json").read_text(), data[2])
    m = card.bouts.assign(date=card.date)
    flipped = m.rename(columns={"fighter_1": "fighter_2", "fighter_2": "fighter_1",
                                "fighter_1_id": "fighter_2_id", "fighter_2_id": "fighter_1_id"})
    p = P.predict_matchups(models, *data, m)["p"]
    p_flipped = P.predict_matchups(models, *data, flipped)["p"]
    np.testing.assert_allclose(p, 1 - p_flipped)
    assert ((p > 0) & (p < 1)).all()


def test_sample_html_card(data, models):
    card = P.read_card("sample_card.html", (SAMPLES / "sample_card.html").read_bytes(), data[2])
    assert len(card.bouts) == 6 and card.date == pd.Timestamp("2026-11-21")
    assert card.bouts["fighter_2_id"].isna().sum() == 1  # the fake debut fighter
    preds = P.predict_matchups(models, *data, card.bouts.assign(date=card.date))
    assert preds.loc[5, "b_is_debut"] == 1


def test_ambiguous_name_is_not_guessed(data):
    assert P.resolve_fighter("Bruno Silva", data[2]) is None


def test_names_match_without_accents(data):
    assert P.normalise_name("Uroš Medić") == "uros medic"
    assert P.resolve_fighter("CIRYL GANE", data[2]) == P.resolve_fighter("Ciryl Gane", data[2]) is not None
