"""
ensemble.py – Fusion layer blending RF & Engle–Granger scores
=============================================================
This *clean* rebuild ensures **only one** `from __future__ import annotations`
statement positioned exactly after the module docstring (as required by
Python), eliminating the previous SyntaxError.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .stats_module import engle_granger_test
from .ml_module import load_model

# ---------------------------------------------------------------------------
# Globals & model cache
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rf.pkl"
_RF_MODEL = None  # lazy‑loaded singleton


def _get_model(model_path: str | Path | None = None):
    """Load RF from disk once and reuse."""
    global _RF_MODEL  # noqa: PLW0603
    if _RF_MODEL is None or model_path is not None:
        _RF_MODEL = load_model(model_path or DEFAULT_MODEL_PATH)
    return _RF_MODEL

# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def score_one_pair(
    prices1: pd.Series,
    prices2: pd.Series,
    feat_row: pd.Series | pd.DataFrame,
    *,
    model=None,
    alpha_rf: float = 0.6,
    alpha_stat: float = 0.4,
    coint_window: int = 252,
) -> Dict[str, float]:
    """Compute ML confidence, statistical score, and blended ensemble value."""
    # Ensure 2‑D DataFrame for scikit‑learn
    if isinstance(feat_row, pd.Series):
        X_ml = feat_row[["corr", "z_spread"]].to_frame().T
    else:
        if len(feat_row) != 1:
            raise ValueError("feat_row DataFrame must have exactly one row.")
        X_ml = feat_row[["corr", "z_spread"]]

    model = model or _get_model()
    ml_conf = float(model.predict_proba(X_ml)[0, 1])

    pval, _ = engle_granger_test(prices1.tail(
        coint_window), prices2.tail(coint_window))
    stat_score = 1.0 - pval  # higher = stronger cointegration

    w_sum = alpha_rf + alpha_stat
    alpha_rf, alpha_stat = alpha_rf / w_sum, alpha_stat / w_sum
    ensemble_score = alpha_rf * ml_conf + alpha_stat * stat_score

    return {"ml_conf": ml_conf, "stat_score": stat_score, "ensemble": ensemble_score}

# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------


def rank_pairs(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    *,
    top_k: int = 10,
    model=None,
    **score_kwargs: Any,
) -> pd.DataFrame:
    """Return a DataFrame (possibly empty) of the top‑K pairs by ensemble score."""
    latest_rows = features.groupby("pair").tail(1).set_index("pair")
    records: list[dict[str, float | str]] = []

    for pair, row in latest_rows.iterrows():
        t1, t2 = pair.split("-")
        if {t1, t2}.issubset(prices.columns):
            try:
                scores = score_one_pair(
                    prices[t1], prices[t2], row, model=model, **score_kwargs)
                scores["pair"] = pair
                records.append(scores)
            except Exception:
                # Skip pairs where stats fail (e.g. too short history)
                continue

    df = pd.DataFrame(records, columns=[
                      "pair", "ml_conf", "stat_score", "ensemble"])
    if not df.empty:
        df = df.sort_values("ensemble", ascending=False).head(
            top_k).reset_index(drop=True)
    return df
