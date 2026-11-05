"""UFC fight predictor.  Run from the project root:  streamlit run app/streamlit_app.py"""
import html
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import pandas as pd
import requests
import streamlit as st

from src.ufc import db, ufccom
from src.ufc import predict as P
from src.ufc.features import load_inputs
from src.ufc.model import contributions
from src.ufc.update import ARTIFACT_PATH, CALIBRATION_PLOT, ensure_ready, refresh_database, retrain

st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="wide")

RED, BLUE = "#d20a0a", "#2563c9"  # corner colours: fighter_1 / fighter_2
st.markdown(f"""<style>
/* A calmer canvas with stronger hierarchy and consistent spacing. */
[data-testid="stAppViewContainer"] {{ background: #f8f7f5; }}
[data-testid="stHeader"] {{ background: #fff; border-bottom: 1px solid #e7e9ed; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1180px; }}
h1 {{ letter-spacing: -0.045em; font-weight: 780 !important; color: #171b22; }}
h2, h3 {{ letter-spacing: -0.025em; color: #202631; }}
[data-testid="stCaptionContainer"] {{ color: #657080; }}
[data-testid="stMetric"] {{ background: white; border: 1px solid #e7e9ed; border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 2px 8px #18233308; }}
[data-testid="stMetricLabel"] {{ color: #687383; }}
[data-testid="stMetricValue"] {{ font-weight: 750; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: #e6e8ec; border-radius: 16px; background: white; box-shadow: 0 3px 12px #18233308; }}
[data-testid="stTabs"] button {{ font-weight: 650; }}
[data-testid="stBaseButton-primary"] {{ border-radius: 9px; font-weight: 650; }}
[data-testid="stBaseButton-secondary"] {{ border-radius: 9px; font-weight: 600; }}
[data-testid="stTopNav"] {{ gap: .2rem; }}
[data-testid="stTopNav"] a, [data-testid="stTopNav"] button {{
  background:transparent !important; border:0 !important; border-bottom:2px solid transparent !important;
  border-radius:0 !important; color:#535b66 !important; font-weight:650; padding:.8rem 1rem !important;
}}
[data-testid="stTopNav"] a[aria-current="page"],
[data-testid="stTopNav"] button[aria-current="page"],
[data-testid="stTopNav"] [aria-selected="true"] {{ color:#171b22 !important; border-bottom-color:{RED} !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid #e7e9ed; border-radius: 10px; overflow: hidden; }}
.page-intro {{ color: #667180; font-size: 1.02rem; margin-top: -.6rem; margin-bottom: 1.2rem; }}
.event-heading {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:1.6rem 0 .65rem; }}
.event-title {{ border-left:5px solid {RED}; padding-left:.65rem; color:#17191d; font-family:Impact, "Arial Narrow", sans-serif; font-size:clamp(1.35rem, 2.5vw, 2rem); font-weight:900; letter-spacing:-.025em; line-height:1.05; text-transform:uppercase; }}
.event-date {{ color:#4b5058; font-size:1rem; white-space:nowrap; }}
.event-subtitle {{ color:#69717d; font-size:.85rem; margin:.45rem 0 .75rem .9rem; }}
.corner-legend {{ display:flex; gap:1.1rem; margin:.3rem 0 .9rem; color:#59616d; font-size:.82rem; }}
.corner-legend .red {{ color:{RED}; font-weight:700; }}
.corner-legend .blue {{ color:{BLUE}; font-weight:700; }}
.prediction-row {{ padding:.8rem .15rem .25rem; }}
.bout-head {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:.55rem; }}
.bout-names {{ color:#1c1d20; font-size:clamp(1rem, 1.7vw, 1.35rem); }}
.bout-names strong {{ font-weight:750; }}
.bout-names .favorite {{ font-weight:850; }}
.bout-vs {{ color:#53565c; padding:0 .35em; }}
.bout-odds {{ color:#54575e; font-size:clamp(.98rem, 1.5vw, 1.25rem); white-space:nowrap; font-variant-numeric:tabular-nums; }}
.bout-odds .probability-a {{ color:{RED}; }}
.bout-odds .probability-b {{ color:{BLUE}; }}
.bout-odds .favorite {{ font-weight:850; }}
.prob-track {{ display:flex; height:8px; width:100%; overflow:hidden; }}
.prob-track .corner-a {{ display:block; height:100%; background:{RED}; }}
.prob-track .corner-b {{ display:block; height:100%; background:{BLUE}; }}
.bout-notes {{ color:#727c89; font-size:.76rem; margin-top:.35rem; }}
.bout-notes:empty {{ display:none; }}
.fight-separator {{ border:0; border-top:1px solid #dedfdf; margin:.15rem 0; }}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding:1.5rem 1rem 3rem; }}
  .event-heading {{ align-items:flex-start; flex-direction:column; gap:.35rem; }}
  .event-date {{ font-size:.9rem; }}
  .bout-head {{ align-items:flex-start; flex-direction:column; gap:.2rem; }}
  .bout-odds {{ font-size:.95rem; }}
}}
.split {{ display:flex; height:10px; border-radius:5px; overflow:hidden; margin:4px 0 2px; }}
.split .r {{ background:{RED}; }} .split .b {{ background:{BLUE}; flex:1; }}
.fighter {{ line-height:1.25; }}
.fighter .name {{ font-weight:500; }} .fighter.fav .name {{ font-weight:700; }}
.fighter .pct {{ font-size:1.4rem; font-weight:800; letter-spacing:-.04em; }}
.fighter.red .pct {{ color:{RED}; }} .fighter.blue .pct {{ color:{BLUE}; }}
.fighter.blue {{ text-align:right; }}
.wc {{ text-align:center; font-size:0.8rem; opacity:0.7; }}
</style>""", unsafe_allow_html=True)


def page_heading(title: str, description: str) -> None:
    st.title(title)
    st.markdown(f'<div class="page-intro">{html.escape(description)}</div>', unsafe_allow_html=True)


def split_bar(p: float) -> str:
    """Red share = P(fighter_1), blue = P(fighter_2)."""
    return f'<div class="split"><div class="r" style="width:{p * 100:.1f}%"></div><div class="b"></div></div>'


def fighter_html(name: str, pct: float, corner: str, favourite: bool) -> str:
    return (f'<div class="fighter {corner}{" fav" if favourite else ""}">'
            f'<div class="name">{html.escape(name)}</div><div class="pct">{pct:.0%}</div></div>')


def prediction_row_html(row: pd.Series) -> str:
    """Show corner-consistent probabilities and the split bar for one bout."""
    p = min(max(float(row["p"]), 0.0), 1.0)
    a, b = html.escape(str(row["fighter_1"])), html.escape(str(row["fighter_2"]))
    fav_a = " favorite" if p > .5 else ""
    fav_b = " favorite" if p < .5 else ""
    weight = row.get("weight_class")
    notes = [weight] if isinstance(weight, str) else []
    if row["different_division"]:
        notes.append(f"⚠ Different weight classes ({row['a_division']} vs {row['b_division']})")
    unknown = [name for name, fid in ((row["fighter_1"], row["fighter_1_id"]),
                                      (row["fighter_2"], row["fighter_2_id"]))
               if not isinstance(fid, str)]
    if unknown:
        notes.append("Not in our data, treated as debut: " + ", ".join(unknown))
    note_markup = html.escape(" · ".join(notes))
    return (
        '<div class="prediction-row">'
        f'<div class="bout-head"><div class="bout-names"><strong class="{fav_a.strip()}">{a}</strong>'
        f'<span class="bout-vs">vs</span><strong class="{fav_b.strip()}">{b}</strong></div>'
        f'<div class="bout-odds"><span class="probability-a{fav_a}">{p:.0%}</span> / '
        f'<span class="probability-b{fav_b}">{1-p:.0%}</span></div></div>'
        f'<div class="prob-track" role="img" aria-label="{a}: {p:.0%}; {b}: {1-p:.0%}">'
        f'<span class="corner-a" style="width:{p * 100:.1f}%"></span>'
        f'<span class="corner-b" style="width:{(1-p) * 100:.1f}%"></span></div>'
        f'<div class="bout-notes">{note_markup}</div></div>'
    )

# feature -> (label, format); formats take the raw per-fighter value
STAT_ROWS = {
    "age": ("Age", "{:.1f}"),
    "height": ("Height (in)", "{:.0f}"),
    "reach": ("Reach (in)", "{:.0f}"),
    "experience": ("UFC fights", "{:.0f}"),
    "win_streak": ("Win streak", "{:.0f}"),
    "days_since_last": ("Days since last fight", "{:.0f}"),
    "sig_landed_pm": ("Sig. strikes landed / min", "{:.2f}"),
    "sig_absorbed_pm": ("Sig. strikes absorbed / min", "{:.2f}"),
    "td_acc": ("Takedown accuracy", "{:.0%}"),
    "td_def": ("Takedown defence", "{:.0%}"),
    "finish_rate": ("Finishes per fight", "{:.0%}"),
}
LEAGUE_AVG_FOR_DEBUT = {"sig_landed_pm", "sig_absorbed_pm", "td_acc", "td_def", "finish_rate"}


# ---------- cached loading (keyed on file mtimes so an update invalidates them) ----------

def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading fight history…")
def get_data(db_mtime: float):
    conn = db.connect()
    db.init_db(conn)
    return load_inputs(conn)


@st.cache_resource(show_spinner="Loading models…")
def get_artifacts(mtime: float):
    return P.load_artifacts()


@st.cache_data(show_spinner="Computing features…")
def cached_predictions(db_mtime: float, model_mtime: float, matchups: pd.DataFrame) -> pd.DataFrame:
    fights, stats, fighters = get_data(db_mtime)
    return P.predict_matchups(get_artifacts(model_mtime)["models"], fights, stats, fighters, matchups)


def load_everything():
    if not ARTIFACT_PATH.exists():  # fresh install / freshly booted host
        with st.status("First run: downloading fight data and training the model (about 30 s)…") as status:
            ensure_ready()
            status.update(label="Ready", state="complete")
        st.cache_data.clear()
        st.cache_resource.clear()
    db_mtime, model_mtime = _mtime(db.DB_PATH), _mtime(ARTIFACT_PATH)
    fights, stats, fighters = get_data(db_mtime)
    return fights, stats, fighters, get_artifacts(model_mtime), (db_mtime, model_mtime)


# ---------- prediction display ----------

def confidence_label(p: float) -> str:
    edge = abs(p - 0.5)
    if edge < 0.05:
        return "Coin flip"
    if edge < 0.15:
        return "Slight edge"
    if edge < 0.25:
        return "Clear favourite"
    return "Strong favourite"


def stat_table(row: pd.Series) -> pd.DataFrame:
    a, b = row["fighter_1"], row["fighter_2"]
    rows = [{"Stat": "Weight class", a: row["a_division"] or "—", b: row["b_division"] or "—"}]
    for feat, (label, fmt) in STAT_ROWS.items():
        cells = {}
        for side, name in (("a", a), ("b", b)):
            value, debut = row[f"{side}_{feat}"], row[f"{side}_is_debut"] == 1
            if pd.isna(value) or (debut and feat == "days_since_last"):
                cells[name] = "—"
            else:
                cells[name] = fmt.format(value) + (" (league avg)" if debut and feat in LEAGUE_AVG_FOR_DEBUT else "")
        rows.append({"Stat": label, **cells})
    return pd.DataFrame(rows).set_index("Stat")


def show_matchup(row: pd.Series, models: dict) -> None:
    a, b, p = row["fighter_1"], row["fighter_2"], float(row["p"])
    with st.container(border=True):
        left, mid, right = st.columns([2, 3, 2], vertical_alignment="center")
        left.markdown(fighter_html(a, p, "red", p >= 0.5), unsafe_allow_html=True)
        right.markdown(fighter_html(b, 1 - p, "blue", p < 0.5), unsafe_allow_html=True)
        mid.markdown(split_bar(p) + f'<div class="wc">{confidence_label(p)}: {html.escape(a if p >= 0.5 else b)}</div>',
                     unsafe_allow_html=True)

    model_ps = {name: float(row[f"p_{name}"]) for name in models}
    spread = max(model_ps.values()) - min(model_ps.values())
    st.caption(
        " · ".join(f"{name}: {v:.0%}" for name, v in model_ps.items())
        + (" — models disagree, treat with caution" if spread > 0.1 else " — models agree")
    )
    if row["different_division"]:
        st.warning(
            f"Different weight classes: {a} last fought at **{row['a_division']}**, "
            f"{b} at **{row['b_division']}**. The model doesn't account for weight class, "
            "so treat this prediction with caution."
        )
    for side, name in (("a", a), ("b", b)):
        if row[f"{side}_is_debut"] == 1:
            st.warning(f"{name} has no UFC fights in the data, so their stats are league averages.")

    stats_col, why_col = st.columns([3, 2])
    with stats_col:
        st.subheader("Pre-fight stats")
        st.dataframe(stat_table(row), width="stretch")
    with why_col:
        st.subheader("What drives the pick")
        contrib = contributions(models["logreg"], row.to_frame().T)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        labels = [STAT_ROWS.get(c.removesuffix("_diff"), (c.removesuffix("_diff").replace("_", " ").title(),))[0]
                  for c in top.index]
        chart_df = pd.DataFrame({
            "Factor": labels,
            "Push": -top.to_numpy(),  # negative = towards fighter_1, drawn on the left like their corner
            "Favours": [a if v > 0 else b for v in top],
            "Log-odds": top.abs().round(3).to_numpy(),
        })
        reach = max(float(top.abs().max()), 0.05) * 1.1
        chart = alt.Chart(chart_df).mark_bar(cornerRadius=3).encode(
            x=alt.X("Push:Q", title=f"← {a}   ·   {b} →", scale=alt.Scale(domain=[-reach, reach]),
                    axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("Factor:N", sort=labels, title=None),
            color=alt.Color("Favours:N", scale=alt.Scale(domain=[a, b], range=[RED, BLUE]), legend=None),
            tooltip=["Factor", "Favours", "Log-odds"],
        ).properties(height=34 * len(labels) + 40)
        st.altair_chart(chart, width="stretch")
        st.caption("Bar length = how much that difference moves the logistic regression's odds.")


# ---------- pages ----------

def missing_model(artifacts, fighters) -> bool:
    if artifacts is None or fighters.empty:
        st.info("No data or model yet. Open **Data & model** and click **Update data & retrain**.")
        return True
    return False


def open_head_to_head(row: pd.Series, labels_by_id: dict) -> None:
    """Prefill the head-to-head page with this bout and go there."""
    st.session_state["fighter_a"] = labels_by_id[row["fighter_1_id"]]
    st.session_state["fighter_b"] = labels_by_id[row["fighter_2_id"]]
    st.session_state["fight_date"] = row["date"].date()
    st.switch_page(HEAD_TO_HEAD)


def bout_row(row: pd.Series, key: str, labels_by_id: dict) -> None:
    """Table-like bout row with corner probabilities and a stats/details action."""
    unknown = [name for name, fid in ((row["fighter_1"], row["fighter_1_id"]), (row["fighter_2"], row["fighter_2_id"]))
               if not isinstance(fid, str)]
    display, action = st.columns([10, 1.7], vertical_alignment="center")
    display.markdown(prediction_row_html(row), unsafe_allow_html=True)
    if action.button("Details →", key=key, disabled=bool(unknown), width="stretch",
                     help="View pre-fight stats and what drives the model's pick" if not unknown
                     else "Fighter not in our data"):
        open_head_to_head(row, labels_by_id)
    st.markdown('<hr class="fight-separator">', unsafe_allow_html=True)


def upload_card_section(fighters: pd.DataFrame, artifacts: dict, mtimes: tuple) -> None:
    """Fallback: predict a card from a saved ufcstats HTML page or a JSON file."""
    st.write("Upload an event page saved from ufcstats.com (*File → Save Page As*, HTML) or a JSON card.")
    st.code('{"event": "UFC 334", "date": "2026-11-14",\n'
            ' "bouts": [["Fighter A", "Fighter B"], ["Fighter C", "Fighter D"]]}', language="json")
    upload = st.file_uploader("Fight card", type=["html", "htm", "json"])
    if not upload:
        return
    try:
        card = P.read_card(upload.name, upload.getvalue(), fighters)
    except (ValueError, KeyError, IndexError) as err:
        st.error(f"Couldn't read that card: {err}")
        return
    if card.bouts.empty:
        st.error("No bouts found in that file.")
        return
    preds = cached_predictions(*mtimes, card.bouts.assign(date=card.date))
    st.subheader(f"{card.event} — {card.date:%B %d, %Y}")
    options = P.fighter_options(fighters)
    labels_by_id = dict(zip(options["fighter_id"], options["label"]))
    for i, row in preds.assign(date=card.date).iterrows():
        bout_row(row, f"upload-{i}", labels_by_id)


def load_cards() -> dict | None:
    """Cached ufc.com cards, refetching (with a progress bar) when older than a day."""
    data = ufccom.read_cache()
    if ufccom.cache_age_hours(data) < ufccom.MAX_AGE_HOURS:
        return data
    bar = st.progress(0.0, text="Fetching upcoming cards from ufc.com (one page every 15 s, about a minute)…")
    try:
        return ufccom.load_upcoming(progress=lambda i, n, text: bar.progress(i / n, text=f"Fetching {text}…"))
    except requests.RequestException as err:
        if data:
            st.warning(f"Couldn't refresh from ufc.com ({err}); showing the last saved cards.")
            return data
        st.error(f"Couldn't reach ufc.com: {err}")
        return None
    finally:
        bar.empty()


def page_upcoming() -> None:
    page_heading("Upcoming fights", "Explore the next cards and compare each fighter’s win probability.")
    fights, stats, fighters, artifacts, mtimes = load_everything()
    if missing_model(artifacts, fighters):
        return

    data = load_cards()
    cards = [c for c in P.cards_from_upcoming(data, fighters) if c.date.date() >= date.today()] if data else []

    if data:
        age = ufccom.cache_age_hours(data)
        info, refresh = st.columns([4, 1], vertical_alignment="center")
        info.caption(f"Cards from [ufc.com](https://www.ufc.com/events), updated "
                     f"{'just now' if age < 1 / 60 else f'{age:.0f} h ago' if age >= 1 else f'{age * 60:.0f} min ago'}. "
                     "Probability = our model's chance each fighter wins.")
        if refresh.button("Refresh", disabled=age < 1, help="Available once an hour, to go easy on ufc.com"):
            ufccom.load_upcoming(force=True)
            st.rerun()
        st.markdown(
            f'<div class="corner-legend"><span class="red">● Red: first fighter</span>'
            f'<span class="blue">● Blue: second fighter</span></div>',
            unsafe_allow_html=True,
        )

    if cards:
        options = P.fighter_options(fighters)
        labels_by_id = dict(zip(options["fighter_id"], options["label"]))
        matchups = pd.concat([c.bouts.assign(date=c.date, card=i) for i, c in enumerate(cards)], ignore_index=True)
        preds = cached_predictions(*mtimes, matchups.drop(columns=["card", "weight_class"]))
        preds[["card", "weight_class", "date"]] = matchups[["card", "weight_class", "date"]]

        for i, card in enumerate(cards):
            st.markdown(
                f'<div class="event-heading"><div class="event-title">{html.escape(card.event)}'
                f': {html.escape(card.headline)}</div><div class="event-date">{card.date:%a, %b %d, %Y}</div></div>'
                f'<div class="event-subtitle">{len(card.bouts)} bouts · '
                f'<a href="{html.escape(card.url, quote=True)}">Official fight card</a></div>',
                unsafe_allow_html=True,
            )
            card_preds = preds[preds["card"] == i]
            for j, row in card_preds.iterrows():
                bout_row(row, f"bout-{j}", labels_by_id)
    elif data:
        st.info("No upcoming cards with announced bouts right now.")

    with st.expander("How the predictions work"):
        st.markdown(
            "- Each fighter's stats (strikes landed/absorbed per minute, takedown accuracy/defence, finish rate, "
            "win streak, layoff, age, reach…) are computed **only from fights before the event date**.\n"
            "- The model compares the two fighters' differences and outputs a **win probability**. It's the average "
            "of a logistic regression and an XGBoost model trained on ~8,800 UFC fights since 1994.\n"
            "- On the most recent year of fights (not used for training) it picked the winner **~61%** of the time, "
            "and its probabilities are well calibrated: 70% favourites won about 70% of the time.\n"
            "- It doesn't know about injuries, short-notice replacements, betting odds or weight cuts. "
            "Treat it as one opinion, not a sure thing."
        )
    with st.expander("Or upload a fight card file"):
        upload_card_section(fighters, artifacts, mtimes)


def page_head_to_head() -> None:
    page_heading("Head to head", "Choose two fighters and a date to see the forecast, key stats, and what moves the model’s pick.")
    fights, stats, fighters, artifacts, mtimes = load_everything()
    if missing_model(artifacts, fighters):
        return

    options = P.fighter_options(fighters)
    labels = options["label"].tolist()
    st.session_state.setdefault("fight_date", date.today())
    c1, c2, c3 = st.columns([2, 2, 1])
    name_a = c1.selectbox("Fighter A", labels, index=None, key="fighter_a", placeholder="Search a fighter…")
    name_b = c2.selectbox("Fighter B", labels, index=None, key="fighter_b", placeholder="Search a fighter…")
    fight_date = c3.date_input("Fight date", key="fight_date")
    if not (name_a and name_b):
        st.info("Start by searching for two fighters above. Their stats will be calculated using only fights before the date you choose.")
        return
    if name_a == name_b:
        st.warning("Pick two different fighters.")
        return
    ra = options.loc[options["label"] == name_a].iloc[0]
    rb = options.loc[options["label"] == name_b].iloc[0]
    matchups = pd.DataFrame([{
        "fighter_1": ra["name"], "fighter_2": rb["name"],
        "fighter_1_id": ra["fighter_id"], "fighter_2_id": rb["fighter_id"],
        "date": pd.Timestamp(fight_date),
    }])
    st.divider()
    show_matchup(cached_predictions(*mtimes, matchups).iloc[0], artifacts["models"])


def page_data() -> None:
    page_heading("Data & model", "Check the dataset, refresh fight history, and review held-out model performance.")
    fights, _, fighters, artifacts, _ = load_everything()

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest event", f"{fights['date'].max():%b %d, %Y}" if len(fights) else "—")
    c2.metric("Fights", f"{len(fights):,}")
    c3.metric("Fighters", f"{len(fighters):,}")

    if st.button("Update data & retrain", type="primary"):
        with st.status("Updating…", expanded=True) as status:
            conn = db.connect()
            st.write("Downloading and loading new events…")
            added = refresh_database(conn)
            st.write(f"Added {added['events']} events, {added['fights']} fights, "
                     f"{added['fight_stats']} stat rows.")
            st.write("Rebuilding features and retraining…")
            retrain(conn)
            status.update(label="Up to date", state="complete")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if artifacts is None:
        st.info("No model trained yet.")
        return
    st.subheader("Model")
    st.caption(f"Trained {artifacts['trained_at']} on {artifacts['n_fights']:,} fights. "
               f"Held-out evaluation: {artifacts['split']}")
    st.dataframe(artifacts["metrics"].style.format("{:.3f}"), width="stretch")
    st.caption("Log loss: 0.693 = coin flip, lower is better. Brier: 0.25 = coin flip, lower is better.")
    if CALIBRATION_PLOT.exists():
        st.image(str(CALIBRATION_PLOT), caption="Calibration on the held-out year", width=480)


UPCOMING = st.Page(page_upcoming, title="Upcoming fights", default=True)
HEAD_TO_HEAD = st.Page(page_head_to_head, title="Head to head", url_path="head-to-head")
DATA = st.Page(page_data, title="Data & model", url_path="data")
st.navigation([UPCOMING, HEAD_TO_HEAD, DATA], position="top").run()
