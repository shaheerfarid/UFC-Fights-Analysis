# UFC Fight Predictor

Win probabilities for upcoming UFC fights, from a model trained on every UFC fight since 1994.

- **Upcoming fights**: the next cards from ufc.com with a prediction for every bout
- **Head to head**: any two fighters, with pre-fight stats and what drives the pick
- **Data & model**: one-click data update and the model's held-out performance

## How it works

1. **Data**: fight results, per-round stats and fighter profiles from the
   [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) dataset (sourced from ufcstats.com),
   stored in SQLite and updated incrementally. Upcoming cards come from ufc.com, fetched at most once a day with a
   15 s delay between requests as its robots.txt asks.
2. **Features**: for every fight, each fighter's stats (strikes landed/absorbed per minute, takedown accuracy/defence,
   finish rate, win streak, layoff, age, height, reach) are computed **only from fights before that date**, with small
   samples shrunk toward the league average. The model sees fighter A minus fighter B.
3. **Models**: logistic regression and XGBoost, trained with random corner swaps so the order of the names can't matter.
   The app shows their average.
4. **Evaluation**: trained on older fights and tested on the most recent year: ~61% of winners picked, log loss 0.648
   vs 0.693 for a coin flip, and well-calibrated probabilities.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py   # first launch downloads data and trains (~30 s)
python -m pytest -q                  # leakage, symmetry and parsing checks
```

## Layout

```
app/streamlit_app.py   Streamlit UI
src/ufc/               http (download) · parse (cleaning) · db (SQLite) · features · model
                       update (refresh + retrain) · predict (matchups, cards) · ufccom (upcoming cards)
tests/                 pytest suite
samples/               example fight-card files for the upload option
```

Predictions are for fun and learning. The model knows nothing about injuries, short-notice replacements or betting markets.
