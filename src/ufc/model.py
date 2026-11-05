"""Train and evaluate win-probability models on the point-in-time feature table.

Run as a script for a full train/evaluate report:
    python -m src.ufc.model
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import random_corner_swap

XGB_PARAMS = dict(
    learning_rate=0.02, max_depth=3, min_child_weight=10, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
)
MAX_TREES = 3000


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: the *_diff features plus the symmetric *_unknown flags."""
    return [c for c in df.columns if c.endswith(("_diff", "_unknown"))]


def time_split(df: pd.DataFrame, test_start: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = fights strictly before test_start, test = on/after it.

    Default test_start is one year before the newest fight.
    """
    test_start = pd.Timestamp(test_start) if test_start else df["date"].max() - pd.DateOffset(years=1)
    return df[df["date"] < test_start].copy(), df[df["date"] >= test_start].copy()


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Same fights seen from the other corner: diffs negated, flags unchanged."""
    out = df.copy()
    diffs = [c for c in out.columns if c.endswith("_diff")]
    out[diffs] = -out[diffs]
    return out


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """P(fighter_1 wins), averaged with 1 - P(fighter_2 wins) so corner order can't matter."""
    cols = feature_columns(df)
    p = model.predict_proba(df[cols])[:, 1]
    p_mirror = model.predict_proba(mirror(df)[cols])[:, 1]
    return (p + 1 - p_mirror) / 2


def fit_logreg(train: pd.DataFrame, C: float = 1.0) -> Pipeline:
    """Standardised logistic regression baseline."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return model.fit(train[feature_columns(train)], train["y"])


def fit_xgb(train: pd.DataFrame, valid_frac: float = 0.15) -> XGBClassifier:
    """XGBoost with the tree count picked by early stopping on the latest slice of train.

    The validation slice comes from the end of the *training* period, never the
    test set, then the model is refit on all of train with that tree count.
    """
    cols = feature_columns(train)
    train = train.sort_values("date")
    cut = int(len(train) * (1 - valid_frac))
    fit_part, valid_part = train.iloc[:cut], train.iloc[cut:]

    probe = XGBClassifier(**XGB_PARAMS, n_estimators=MAX_TREES, early_stopping_rounds=200)
    probe.fit(fit_part[cols], fit_part["y"], eval_set=[(valid_part[cols], valid_part["y"])], verbose=False)

    model = XGBClassifier(**XGB_PARAMS, n_estimators=probe.best_iteration + 1)
    return model.fit(train[cols], train["y"])


def train_final(features: pd.DataFrame, seed: int = 0) -> dict:
    """Fit both models on every decided fight, for predicting upcoming ones."""
    train = random_corner_swap(features, seed=seed)
    return {"logreg": fit_logreg(train), "xgboost": fit_xgb(train)}


def contributions(logreg: Pipeline, row: pd.DataFrame) -> pd.Series:
    """Log-odds each *_diff feature adds toward fighter_1 (+) or fighter_2 (-) for one row.

    coef * value / scale: measured from an even matchup (all diffs 0), not the
    training mean, so the sign always means "favours fighter_1".
    """
    scaler, lr = logreg[0], logreg[-1]
    cols = feature_columns(row)
    contrib = pd.Series(lr.coef_[0] * row[cols].iloc[0].to_numpy(float) / scaler.scale_, index=cols)
    return contrib[[c for c in cols if c.endswith("_diff")]]


def implied_probs(decimal_odds_a: pd.Series, decimal_odds_b: pd.Series) -> pd.Series:
    """Bookmaker probability that A wins, with the margin (vig) removed by normalising."""
    raw_a, raw_b = 1 / decimal_odds_a, 1 / decimal_odds_b
    return raw_a / (raw_a + raw_b)


def evaluate(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    """Probability metrics (lower is better for log loss / Brier) plus accuracy and AUC."""
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, p >= 0.5),
        "auc": roc_auc_score(y, p) if y.nunique() == 2 else float("nan"),
    }


def plot_calibration(y: pd.Series, probs: dict[str, np.ndarray], path: Path | str, n_bins: int = 10) -> Path:
    """Reliability diagram: predicted probability vs. observed win rate per bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, hist) = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfectly calibrated")
    for name, p in probs.items():
        frac_won, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_won, marker="o", label=name)
        hist.hist(p, bins=20, range=(0, 1), histtype="step", label=name)
    ax.set(ylabel="observed win rate", title="Calibration on test set")
    ax.legend(loc="upper left")
    hist.set(xlabel="predicted P(fighter_1 wins)", ylabel="fights")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run(features: pd.DataFrame, test_start: str | None = None, seed: int = 0,
        plot_path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Split by time, corner-swap, train both models, return (metrics table, models).

    Train and test are swapped with different seeds so test's base rate is ~50%
    and no model can score by learning "first-listed fighter wins".
    """
    train, test = time_split(features, test_start)
    train = random_corner_swap(train, seed=seed)
    test = random_corner_swap(test, seed=seed + 1)

    models = {"logreg": fit_logreg(train), "xgboost": fit_xgb(train)}
    probs = {
        "naive (train base rate)": np.full(len(test), train["y"].mean()),
        **{name: predict(m, test) for name, m in models.items()},
    }
    table = pd.DataFrame({name: evaluate(test["y"], p) for name, p in probs.items()}).T
    table.attrs["split"] = (
        f"train {len(train)} fights {train.date.min().date()}..{train.date.max().date()} | "
        f"test {len(test)} fights {test.date.min().date()}..{test.date.max().date()}"
    )
    if plot_path:
        plot_calibration(test["y"], {k: v for k, v in probs.items() if k in models}, plot_path)
    return table, models


if __name__ == "__main__":
    from . import db
    from .features import build_features, load_inputs

    conn = db.connect()
    X = build_features(*load_inputs(conn))
    out = Path(__file__).resolve().parents[2] / "data" / "calibration.png"
    table, _ = run(X, plot_path=out)
    print(table.attrs["split"])
    print(table.round(4).to_string())
    print(f"calibration plot: {out}")
