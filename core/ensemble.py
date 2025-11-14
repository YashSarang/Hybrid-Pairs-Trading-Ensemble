"""Ensembling utilities for pair selection scores and entry/exit signals.

This module is UI-agnostic and depends only on numpy/pandas and
`core.selectors` for the `Pair`/`PairScore` dataclasses.

Functions
---------
- normalize_weights(weights)
- ensemble_pair_scores(scores_by_model, weights, top_k=None)
- ensemble_signals(signals_by_model, weights, neutral_band=0.15)
- scores_to_frame(aggregated) -> pd.DataFrame (A,B,Score,Details)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .selectors import Pair, PairScore


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in weights.items()}
    s = float(sum(abs(v) for v in w.values()))
    if s <= 0:
        return {k: 0.0 for k in w}
    return {k: (v / s) for k, v in w.items()}


# ---------------------------------------------
# Pair score ensembling
# ---------------------------------------------

def ensemble_pair_scores(
    scores_by_model: Dict[str, List[PairScore]],
    weights: Dict[str, float],
    top_k: int | None = None,
) -> List[PairScore]:
    """Combine per-model `PairScore` lists into a single ranked list.

    Parameters
    ----------
    scores_by_model : dict of {model_name -> list[PairScore]}
    weights         : dict of {model_name -> weight}
    top_k           : optional cap on number of pairs to return

    Returns
    -------
    list[PairScore]: aggregated scores with merged details per model
    """
    wnorm = normalize_weights(weights)
    bucket: Dict[Tuple[str, str], Dict] = {}

    for model_name, scores in scores_by_model.items():
        w = float(wnorm.get(model_name, 0.0))
        if w == 0 or scores is None:
            continue
        for s in scores:
            key = s.pair.key()
            if key not in bucket:
                bucket[key] = {"pair": s.pair, "score": 0.0, "details": {}}
            bucket[key]["score"] += w * float(s.score)
            bucket[key]["details"][model_name] = s.details

    aggregated = [PairScore(v["pair"], float(v["score"]), v["details"]) for v in bucket.values()]
    aggregated.sort(key=lambda x: x.score, reverse=True)
    return aggregated[:top_k] if top_k else aggregated


def scores_to_frame(aggregated: List[PairScore]) -> pd.DataFrame:
    """Represent aggregated scores as a tidy DataFrame."""
    rows = []
    for ps in aggregated:
        rows.append({"A": ps.pair.a, "B": ps.pair.b, "Score": ps.score, "Details": ps.details})
    return pd.DataFrame(rows)


# ---------------------------------------------
# Signal ensembling
# ---------------------------------------------

def ensemble_signals(
    signals_by_model: Dict[str, pd.Series],
    weights: Dict[str, float],
    neutral_band: float = 0.15,
) -> pd.Series:
    """Blend multiple model signals into a single {-1,0,1} series.

    Steps:
    1) Align to a common DatetimeIndex
    2) Clip each model's signal to [-1, 1]
    3) Weight-average using normalized absolute weights (sign preserved)
    4) Hard sign() to {-1, 0, 1}; additionally zero out small magnitudes within `neutral_band`
    """
    if not signals_by_model:
        raise ValueError("signals_by_model is empty")

    # Align
    aligned = pd.concat(signals_by_model.values(), axis=1)
    aligned.columns = list(signals_by_model.keys())

    # Sanitize & clip
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)

    # Normalize weights
    wnorm = normalize_weights(weights)
    w = np.array([wnorm.get(c, 0.0) for c in aligned.columns], dtype=float)
    if np.allclose(w, 0):
        # If all zero, equal-weight
        w = np.ones(len(aligned.columns), dtype=float)
        w = w / w.sum()

    blended = aligned.values @ w
    out = np.sign(blended)
    # Neutral band to avoid whipsaws
    out[np.abs(blended) < float(neutral_band)] = 0

    return pd.Series(out.astype(int), index=aligned.index, name="signal")
