from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
TABLES = [
    "ufc_event_details",
    "ufc_fight_details",
    "ufc_fight_results",
    "ufc_fight_stats",
    "ufc_fighter_details",
    "ufc_fighter_tott",
]
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def download_tables(dest=RAW_DIR):
    """Download every historical table to dest as CSV, return the paths."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in TABLES:
        path = dest / f"{name}.csv"
        path.write_text(fetch(f"{BASE_URL}/{name}.csv"), encoding="utf-8")
        paths.append(path)
    return paths
