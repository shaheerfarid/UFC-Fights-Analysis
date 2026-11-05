"""Upcoming fight cards from ufc.com, fetched politely and cached on disk.

ufc.com's robots.txt allows /events and /event/* with a 15 s crawl delay, and its
terms allow automated access at a human-like rate. So: one request at a time,
15 s apart, and the result is cached for a day so visitors don't trigger refetches.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

BASE = "https://www.ufc.com"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "upcoming_cards.json"
CRAWL_DELAY = 15        # seconds, from ufc.com/robots.txt
MAX_AGE_HOURS = 24
MAX_EVENTS = 5
HEADERS = {"User-Agent": "UFC-Fights-Analysis/1.0 (personal fight-prediction project)"}
EVENT_TZ = ZoneInfo("America/New_York")  # card dates as ufc.com shows them

_lock = threading.Lock()
_last_request = 0.0


def _get(path: str) -> str:
    """GET a ufc.com page, waiting so requests are at least CRAWL_DELAY apart."""
    global _last_request
    wait = _last_request + CRAWL_DELAY - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    try:
        response = requests.get(BASE + path, headers=HEADERS, timeout=30)
    finally:
        _last_request = time.monotonic()
    response.raise_for_status()
    return response.text


def _event_name(slug: str) -> str:
    """'/event/ufc-334' -> 'UFC 334', fight nights -> 'UFC Fight Night'."""
    number = re.fullmatch(r"/event/ufc-(\d+)", slug)
    return f"UFC {number.group(1)}" if number else "UFC Fight Night"


def parse_events_list(html: str) -> list[dict]:
    """Upcoming events from ufc.com/events: slug, name, headline, date (YYYY-MM-DD)."""
    soup = BeautifulSoup(html, "html.parser")
    now = datetime.now(timezone.utc).timestamp()
    events = []
    for card in soup.select("#events-list-upcoming .c-card-event--result"):
        link = card.select_one(".c-card-event--result__headline a")
        stamp = card.select_one("[data-main-card-timestamp]")
        if not (link and stamp):
            continue
        ts = int(stamp["data-main-card-timestamp"])
        if ts < now:
            continue
        slug = link["href"].split("#")[0]
        events.append({
            "slug": slug,
            "name": _event_name(slug),
            "headline": link.get_text(" ", strip=True),
            "date": datetime.fromtimestamp(ts, EVENT_TZ).date().isoformat(),
        })
    return sorted(events, key=lambda e: e["date"])


def parse_event_page(html: str) -> list[dict]:
    """Bouts on an event page, main event first: fighter_1 (red), fighter_2 (blue), weight_class."""
    soup = BeautifulSoup(html, "html.parser")
    bouts = []
    for fight in soup.select(".c-listing-fight"):
        red = fight.select_one(".c-listing-fight__corner-name--red")
        blue = fight.select_one(".c-listing-fight__corner-name--blue")
        weight = fight.select_one(".c-listing-fight__class-text")
        if not (red and blue):
            continue
        names = red.get_text(" ", strip=True), blue.get_text(" ", strip=True)
        if "TBD" in names:
            continue
        bouts.append({"fighter_1": names[0], "fighter_2": names[1],
                      "weight_class": weight.get_text(" ", strip=True) if weight else None})
    return bouts


def fetch_upcoming(max_events: int = MAX_EVENTS, progress=None) -> dict:
    """Fetch the next `max_events` cards (skipping ones with no bouts announced).

    progress(i, total, text) is called before each event page, for a UI progress bar.
    """
    events = parse_events_list(_get("/events"))[:max_events]
    cards = []
    for i, event in enumerate(events):
        if progress:
            progress(i, len(events), f"{event['name']}: {event['headline']}")
        bouts = parse_event_page(_get(event["slug"]))
        if bouts:
            cards.append({**event, "url": BASE + event["slug"], "bouts": bouts})
    return {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cards": cards}


def read_cache(path: Path = CACHE_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cache_age_hours(data: dict | None) -> float:
    if not data:
        return float("inf")
    fetched = datetime.fromisoformat(data["fetched_at"])
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600


def load_upcoming(force: bool = False, max_age_hours: float = MAX_AGE_HOURS,
                  path: Path = CACHE_PATH, progress=None) -> dict:
    """Cached cards if fresh enough, otherwise fetch (one fetch at a time across visitors)."""
    cached = read_cache(path)
    if not force and cache_age_hours(cached) < max_age_hours:
        return cached
    with _lock:
        cached = read_cache(path)  # another visitor may have just refreshed it
        if not force and cache_age_hours(cached) < max_age_hours:
            return cached
        data = fetch_upcoming(progress=progress)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return data
