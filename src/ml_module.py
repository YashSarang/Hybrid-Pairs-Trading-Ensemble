"""
ml_module.py – Label generation, model training, inference utilities
===================================================================
This module provides everything needed to **train, validate, persist, and
load** a Random-Forest (RF) classifier that predicts whether a stock pair is
likely to mean-revert within a user-defined horizon.  You can later swap the RF
for any scikit-learn-compatible estimator—or even wrap a deep-learning model
behind the same API—without changing the rest of the pipeline.

Public API (import-safe)
------------------------
- `generate_labels_backtest(prices, half_life=20, lookahead=60) -> pd.Series`
- `train_rf(features, labels, *, model_path="models/rf.pkl", **rf_kwargs)`
- `load_model(model_path="models/rf.pkl") -> RandomForestClassifier`
- `predict_proba(model, feature_row) -> float`   # convenience helper

The **label** is 1 when the absolute price spread reverts to its rolling mean
within `lookahead` trading days, approximating a profitable mean-reversion
opportunity.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Label generator
# ---------------------------------------------------------------------------


def _reversion_within_window(spread: pd.Series, window: int) -> pd.Series:
    """Return boolean Series: True if spread crosses its rolling mean in *window* days."""
    mean = spread.rolling(window, min_periods=window).mean()
    # we compare the deviation at t with future path; pad with False at the end
    out = pd.Series(False, index=spread.index)
    for t in range(len(spread) - window):
        dev = spread.iloc[t] - mean.iloc[t]
        if np.isnan(dev):
            continue
        future = spread.iloc[t + 1: t + 1 + window] - mean.iloc[t]
        # sign change implies mean-crossing (strict reversal)
        out.iloc[t] = np.any(np.sign(dev) != np.sign(future))
    return out


def generate_labels_backtest(
    prices: pd.DataFrame,
    half_life: int = 20,
    lookahead: int = 60,
) -> pd.Series:
    """Produce binary labels for *each pair* on each date.

    A label =1 indicates the pair's *price spread* reverted to its rolling mean
    (half-life window) sometime during the next `lookahead` days.

    Returns
    -------
    pd.Series with MultiIndex (date, pair) of booleans/ints.
    """
    labels = []
    for s1, s2 in combinations(prices.columns, 2):
        pair = f"{s1}-{s2}"
        spread = prices[s1] - prices[s2]
        hit = _reversion_within_window(spread, lookahead).astype(int)
        hit.name = pair
        labels.append(hit)
    lab = pd.concat(labels, axis=1)
    lab = lab.stack()  # index = (date, pair)
    lab.index.names = ["date", "pair"]
    return lab.sort_index()

# ---------------------------------------------------------------------------
# Model training / persistence
# ---------------------------------------------------------------------------


def train_rf(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    model_path: str | Path | None = MODELS_DIR / "rf.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    rf_kwargs: Dict[str, Any] | None = None,
    use_gpu: bool = False,
):
    """Fit a Random-Forest on features aligned with labels and persist to disk."""
    if rf_kwargs is None:
        rf_kwargs = {
            "n_estimators": 200,
            "max_depth": 6,
            "class_weight": "balanced",  # note: dropped in GPU path
        }

    # --- Align features with labels
    df = features.merge(labels.rename("y"), on=["date", "pair"]).dropna()
    if df.empty:
        raise ValueError("No training rows after merging features and labels.")
    feature_cols = [c for c in df.columns if c not in {"date", "pair", "y"}]
    # keep as DataFrame for sklearn (preserves names)
    X = df[feature_cols]
    y = df["y"].values

    # --- Train/val split (time-ordered)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = None
    used_gpu = False

    # ---- GPU path: cuML RF (convert to NumPy arrays)
    if use_gpu:
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF  # type: ignore
            gpu_params = dict(rf_kwargs)
            gpu_params.pop("class_weight", None)
            gpu_params.setdefault("random_state", random_state)

            X_train_np = X_train.to_numpy()
            X_val_np = X_val.to_numpy()

            model = cuRF(**gpu_params)
            model.fit(X_train_np, y_train)
            proba_val = model.predict_proba(X_val_np)
            used_gpu = True
        except Exception:
            model = None
            used_gpu = False

    # ---- CPU path: sklearn RF (train with DataFrame so names are stored)
    if model is None:
        from sklearn.ensemble import RandomForestClassifier
        cpu_params = dict(rf_kwargs)
        cpu_params.setdefault("random_state", random_state)
        model = RandomForestClassifier(**cpu_params)
        model.fit(X_train, y_train)
        proba_val = model.predict_proba(X_val)

    # Normalize potential CuPy output
    try:
        import cupy as cp  # type: ignore
        if hasattr(proba_val, "__class__") and isinstance(proba_val, cp.ndarray):
            proba_val = cp.asnumpy(proba_val)
    except Exception:
        pass

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, proba_val[:, 1])
    print(
        f"Validation ROC-AUC: {auc:.3f}  |  Positive rate val: {float(np.mean(y_val)):.3f}"
        + ("  |  (GPU)" if used_gpu else "")
    )

    if model_path:
        import joblib
        joblib.dump(model, model_path)
        print("Model saved →", Path(model_path).resolve())

    return model

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def load_model(model_path: str | Path = MODELS_DIR / "rf.pkl"):
    """Load a persisted RF model (or raise FileNotFoundError)."""
    return joblib.load(model_path)


def predict_proba(
    model,
    feature_row: pd.Series | pd.DataFrame,
) -> float:
    """Return probability of label==1 for a **single** row of features."""
    if isinstance(feature_row, pd.Series):
        feature_row = feature_row.to_frame().T

    # Align by column order if available (sklearn stores this)
    if hasattr(model, "feature_names_in_"):
        feature_row = feature_row.reindex(
            columns=model.feature_names_in_, fill_value=0.0)
        X_in = feature_row
    else:
        X_in = feature_row.to_numpy()

    proba = model.predict_proba(X_in)
    try:
        import cupy as cp  # type: ignore
        if isinstance(proba, cp.ndarray):
            proba = cp.asnumpy(proba)
    except Exception:
        pass
    return float(proba[0, 1])
