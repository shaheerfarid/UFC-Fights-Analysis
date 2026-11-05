"""Predictions for upcoming or hand-picked matchups, and reading saved fight cards."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from bs4 import BeautifulSoup

from .features import matchup_features
from .model import predict
from .update import ARTIFACT_PATH


# Checked in order: women's first, and Light Heavyweight before Heavyweight, since names contain each other.
DIVISIONS = [
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
    "Light Heavyweight", "Heavyweight", "Middleweight", "Welterweight",
    "Lightweight", "Featherweight", "Bantamweight", "Flyweight",
]
# upper weight limit (lbs) -> division, for fighters with no usable fight
WEIGHT_LIMITS = [(125, "Flyweight"), (135, "Bantamweight"), (145, "Featherweight"), (155, "Lightweight"),
                 (170, "Welterweight"), (185, "Middleweight"), (205, "Light Heavyweight"), (265, "Heavyweight")]


def normalise_division(weight_class: str | None) -> str | None:
    """'UFC Interim Lightweight Title Bout' -> 'Lightweight'; catch/open weight -> None."""
    if not isinstance(weight_class, str):
        return None
    text = weight_class.lower()
    return next((d for d in DIVISIONS if d.lower() in text), None)


def division_as_of(fights: pd.DataFrame, fighters: pd.DataFrame, fighter_id: str | None,
                   name: str, before: pd.Timestamp) -> str | None:
    """Division of the fighter's latest fight before `before`, else from profile weight."""
    if isinstance(fighter_id, str):
        mine = (fights["fighter_1_id"] == fighter_id) | (fights["fighter_2_id"] == fighter_id)
    else:
        mine = (fights["fighter_1"] == name) | (fights["fighter_2"] == name)
    past = fights.loc[mine & (fights["date"] < before)].sort_values("date", ascending=False)
    for weight_class in past["weight_class"]:
        if division := normalise_division(weight_class):
            return division

    weight = fighters.loc[fighters["fighter_id"] == fighter_id, "weight_lbs"]
    if len(weight) and pd.notna(weight.iloc[0]):
        return next((d for limit, d in WEIGHT_LIMITS if weight.iloc[0] <= limit), "Heavyweight")
    return None


@dataclass
class Card:
    """An upcoming event: name, date and bouts (fighter_1, fighter_2, fighter_1_id, fighter_2_id)."""
    event: str
    date: pd.Timestamp
    bouts: pd.DataFrame
    headline: str = ""
    url: str = ""


def load_artifacts(path: Path = ARTIFACT_PATH) -> dict | None:
    """Trained models and metrics saved by update.retrain, or None if not trained yet."""
    return joblib.load(path) if Path(path).exists() else None


def fighter_options(fighters: pd.DataFrame) -> pd.DataFrame:
    """Fighters sorted by name with a dropdown label; namesakes get weight and id to tell them apart."""
    out = fighters.sort_values("name").copy()
    dup = out["name"].duplicated(keep=False)
    weight = out["weight_lbs"].astype("Int64").astype("string").fillna("?")
    out["label"] = out["name"].where(~dup, out["name"] + " (" + weight + " lbs, " + out["fighter_id"].str[:6] + ")")
    return out.reset_index(drop=True)


def normalise_name(name: str) -> str:
    """'Uroš Medić' -> 'uros medic': no accents, case or punctuation, single spaces."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", plain.lower()).strip()


def resolve_fighter(name: str, fighters: pd.DataFrame) -> str | None:
    """fighter_id for a name match ignoring accents/case/punctuation; None if unknown or ambiguous."""
    hits = fighters.loc[fighters["name"].map(normalise_name) == normalise_name(name), "fighter_id"]
    return hits.iloc[0] if len(hits) == 1 else None


def _bouts_from_names(pairs: list[tuple[str, str]], fighters: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"fighter_1": a.strip(), "fighter_2": b.strip(),
         "fighter_1_id": resolve_fighter(a, fighters), "fighter_2_id": resolve_fighter(b, fighters)}
        for a, b in pairs
    ]
    return pd.DataFrame(rows, columns=["fighter_1", "fighter_2", "fighter_1_id", "fighter_2_id"])


def cards_from_upcoming(data: dict, fighters: pd.DataFrame) -> list[Card]:
    """Cards cached by ufccom.load_upcoming, with fighter ids resolved; bouts keep weight_class."""
    cards = []
    for c in data["cards"]:
        bouts = _bouts_from_names([(b["fighter_1"], b["fighter_2"]) for b in c["bouts"]], fighters)
        bouts["weight_class"] = [b.get("weight_class") for b in c["bouts"]]
        cards.append(Card(c["name"], pd.Timestamp(c["date"]), bouts, c.get("headline", ""), c.get("url", "")))
    return cards


def parse_card_json(text: str, fighters: pd.DataFrame) -> Card:
    """Read a hand-written card.

    {"event": "UFC 334", "date": "2026-11-14",
     "bouts": [["Fighter A", "Fighter B"], {"fighter_1": "C", "fighter_2": "D"}]}
    """
    data = json.loads(text)
    pairs = [(b["fighter_1"], b["fighter_2"]) if isinstance(b, dict) else (b[0], b[1]) for b in data["bouts"]]
    return Card(data.get("event", "Upcoming event"), pd.Timestamp(data["date"]), _bouts_from_names(pairs, fighters))


def parse_card_html(html: str, fighters: pd.DataFrame) -> Card:
    """Read an event page saved from ufcstats.com in your browser (File > Save Page As).

    Fighters are taken from the fighter-details links in page order, two per bout,
    and identified by the id in the link, so namesakes are never mixed up.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select('a[href*="fighter-details/"]'):
        href = a["href"].rstrip("/")
        if not links or links[-1][0] != href:  # skip a repeated link to the same fighter
            links.append((href, a.get_text(strip=True)))
    if len(links) < 2 or len(links) % 2:
        raise ValueError(f"expected pairs of fighter links, found {len(links)}")

    known = set(fighters["fighter_id"])
    rows = []
    for (href_1, name_1), (href_2, name_2) in zip(links[::2], links[1::2]):
        id_1, id_2 = href_1.rsplit("/", 1)[-1], href_2.rsplit("/", 1)[-1]
        rows.append({"fighter_1": name_1, "fighter_2": name_2,
                     "fighter_1_id": id_1 if id_1 in known else None,
                     "fighter_2_id": id_2 if id_2 in known else None})

    text = soup.get_text(" ", strip=True)
    date = re.search(r"Date:\s*([A-Z][a-z]+\.? \d{1,2}, \d{4})", text)
    if not date:
        raise ValueError("couldn't find 'Date: <Month DD, YYYY>' on the page")
    title = soup.select_one("h2") or soup.title
    event = title.get_text(strip=True) if title else "Upcoming event"
    return Card(event, pd.Timestamp(date.group(1)), pd.DataFrame(rows))


def read_card(filename: str, data: bytes, fighters: pd.DataFrame) -> Card:
    """Parse an uploaded .json or .html card file."""
    text = data.decode("utf-8", errors="replace")
    if filename.lower().endswith(".json"):
        return parse_card_json(text, fighters)
    return parse_card_html(text, fighters)


def predict_matchups(models: dict, fights: pd.DataFrame, stats: pd.DataFrame, fighters: pd.DataFrame,
                     matchups: pd.DataFrame) -> pd.DataFrame:
    """Features, p_<model> per model, p (their average) = P(fighter_1 wins),
    each side's current division (a_division / b_division) and a different_division flag."""
    feats = matchup_features(fights, stats, fighters, matchups)
    feats["fighter_1_id"] = matchups["fighter_1_id"].to_numpy()
    feats["fighter_2_id"] = matchups["fighter_2_id"].to_numpy()
    when = pd.to_datetime(matchups["date"]).tolist()
    for side, n in (("a", 1), ("b", 2)):
        feats[f"{side}_division"] = [
            division_as_of(fights, fighters, fid, name, d)
            for fid, name, d in zip(matchups[f"fighter_{n}_id"], matchups[f"fighter_{n}"], when)
        ]
    feats["different_division"] = (
        feats["a_division"].notna() & feats["b_division"].notna() & (feats["a_division"] != feats["b_division"])
    )
    for name, m in models.items():
        feats[f"p_{name}"] = predict(m, feats)
    feats["p"] = feats[[f"p_{name}" for name in models]].mean(axis=1)
    return feats
