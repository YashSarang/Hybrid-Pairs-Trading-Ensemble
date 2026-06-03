"""Base models and helpers for Pair Selection.

Includes:
- Data classes: Pair, PairScore
- Abstract base: PairSelector
- Helpers _annualize_days, _hurst_rs, _halflife
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------
# Data classes
# ---------------------------------------------

@dataclass(frozen=True)
class Pair:
    a: str
    b: str

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.a, self.b)))


@dataclass
class PairScore:
    pair: Pair
    score: float
    details: dict


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def _annualize_days(index: pd.DatetimeIndex) -> int:
    # Rough mapper for periods per year based on spacing
    if len(index) < 2:
        return 252
    dt = np.median(np.diff(index.values).astype(
        "timedelta64[s]").astype(float))
    if dt <= 120:  # roughly minute-level
        return 252 * 6 * 60
    if dt <= 4000:  # hourly-ish
        return 252 * 24
    return 252


def _hurst_rs(x: pd.Series) -> float:
    """Very small, rough Hurst exponent estimator based on R/S.
    Alternatives: DFA or periodogram. For screening only.
    """
    x = x.dropna().values
    n = len(x)
    if n < 100:
        return 0.5
    max_k = min(100, n // 2)
    lags = np.arange(2, max_k)
    tau = []
    for lag in lags:
        y = x[lag:] - x[:-lag]
        tau.append(np.sqrt(np.std(y)))
    m = np.polyfit(np.log(lags), np.log(np.maximum(tau, 1e-12)), 1)
    return float(m[0])


def _halflife(spread: pd.Series) -> float:
    x = spread.dropna()
    if len(x) < 20:
        return np.inf
    x_lag = x.shift(1).dropna()
    y = x.loc[x_lag.index]
    var = x_lag.var()
    beta = (x_lag.cov(y) / (var + 1e-12)) if var != 0 else 0.0
    if abs(beta) >= 1 or abs(beta) <= 1e-6:
        return np.inf
    return float(-1.0 / np.log(abs(beta)))


# ---------------------------------------------
# Base class
# ---------------------------------------------

class PairSelector:
    name: str = "base"

    def fit(self, prices: pd.DataFrame) -> "PairSelector":
        return self

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        raise NotImplementedError
