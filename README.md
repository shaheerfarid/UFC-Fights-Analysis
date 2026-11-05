# UFC Fight Predictor

**Try it live: https://ufc-fights-predictor.streamlit.app/**

I've wanted to build this since I started watching the UFC. It predicts who wins upcoming fights using every UFC fight since 1994.

The app has three pages:

- **Upcoming fights**: the next few cards from ufc.com, with a prediction for every bout. Click a fight to see the full breakdown.
- **Head to head**: pick any two fighters and see their stats side by side, plus what's pushing the prediction one way or the other.
- **Data & model**: update the data, retrain, and see how the model did on fights it hadn't seen.

## How it works

**Data.** Fight results, round-by-round stats and fighter profiles come from the
[Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) dataset, which is scraped from ufcstats.com.
Everything goes into a SQLite database, and updates only add new events. Upcoming cards are pulled from ufc.com at most
once a day, waiting 15 seconds between requests as their robots.txt asks.

**Features.** For each fight, I work out where both fighters stood going into it: strikes landed and absorbed per minute,
takedown accuracy and defence, finish rate, win streak, time since their last fight, number of fights, age, height and reach.
Only fights before that date are used, so the model can't cheat by looking ahead. Fighters with only a few fights get
their stats pulled toward the league average, so one good night doesn't make someone look unbeatable. The model works
with the difference between the two fighters (A minus B).

**Models.** There are two: a logistic regression and an XGBoost model. The app shows the average of the two. During
training I randomly swap the red and blue corners, so it doesn't matter which fighter you put first.

**Results.** I trained on older fights and tested on the most recent year. It picks the winner about 61% of the time,
with a log loss of 0.648 (a coin flip gets 0.693). The probabilities are well calibrated too: when it says 70%, that
fighter wins roughly 70% of the time.

## Running it yourself

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py   # first run downloads the data and trains, takes about 30 seconds
python -m pytest -q                  # checks for data leakage, symmetry and parsing
```

## What's where

```
app/streamlit_app.py   the Streamlit app
src/ufc/               http (downloads) · parse (cleaning) · db (SQLite) · features · model
                       update (refresh + retrain) · predict (matchups, cards) · ufccom (upcoming cards)
tests/                 pytest tests
samples/               example fight cards for the upload option
```

## Tech stack

Python, pandas, SQLite, scikit-learn, XGBoost, BeautifulSoup, Streamlit, Altair, pytest

## A note

This is a fun project, not betting advice. The model doesn't know about injuries, short-notice replacements or the odds.
