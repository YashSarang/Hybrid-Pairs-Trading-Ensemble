"""Statistical Pair Selection Models.

Includes:
- CorrelationSelector: Pearson correlation-based selection
- DistanceSelector: Gatev et al. (2006) normalized distance method
- CointegrationSelector: Engle-Granger cointegration testing
- CombinedCriteriaSelector: Sarmento & Horta (2021) multi-criteria approach
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .selectors_base import Pair, PairScore, PairSelector, _hurst_rs, _halflife

# Optional dependencies
try:  # statsmodels for cointegration
    from statsmodels.tsa.stattools import coint
except Exception:  # pragma: no cover
    coint = None

try:
    from joblib import Parallel, delayed as _delayed
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    _HAS_JOBLIB = False


class CorrelationSelector(PairSelector):
    # class RollingCorrelationSelector(PairSelector):
    """Select pairs by Rolling Pearson Correlation Coefficient (RPCC)."""
    name = "Correlation"

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        rets = prices.pct_change()

        out: List[PairScore] = []
        for p in candidates:
            if p.a not in rets.columns or p.b not in rets.columns:
                s = 0.0
                meta = {"rpcc_last": 0.0, "lookback": self.lookback}
            else:
                # rolling correlation time series
                rpcc = rets[p.a].rolling(self.lookback).corr(rets[p.b])
                s = float(rpcc.iloc[-1]) if len(rpcc) else 0.0
                if pd.isna(s):
                    s = 0.0
                meta = {"rpcc_last": s, "lookback": self.lookback}

            out.append(PairScore(p, s, meta))

        return out


class DistanceSelector(PairSelector):
    """Gatev et al. (2006) **distance method**.

    mode="zscore": z-normalize each price series over the lookback, then
                    score = -||z_a - z_b||_2 (higher is better)
    Alt mode="cumret": compare cumulative returns similarity: P/P0 - 1
    """
    name = "Distance (Gatev)"

    def __init__(self, lookback: int = 252, mode: str = "zscore"):
        self.lookback = lookback
        self.mode = mode

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        window = prices.tail(self.lookback)
        if self.mode == "zscore":
            z = (window - window.mean()) / (window.std(ddof=0) + 1e-9)
        else:
            z = window / window.iloc[0] - 1
        out: List[PairScore] = []
        for p in candidates:
            if p.a in z and p.b in z:
                dist = float(np.linalg.norm(z[p.a].values - z[p.b].values))
                out.append(PairScore(
                    p, -dist, {"neg_l2": -dist, "mode": self.mode, "lookback": self.lookback}))
            else:
                out.append(PairScore(p, -np.inf, {}))
        return out


class CointegrationSelector(PairSelector):
    """Engle–Granger cointegration test selector.

    Defaults
    --------
    lookback: 2Y (≈ 504 daily bars)
    pvalue_threshold: 0.05 (alternative: 0.01 for stricter)
    """
    name = "Cointegration (Engle–Granger)"

    def __init__(self, lookback: int = 252 * 2, pvalue_threshold: float = 0.05):
        self.lookback = lookback
        self.pvalue_threshold = pvalue_threshold

    def _score_one(self, window: pd.DataFrame, p: Pair) -> PairScore:
        if p.a not in window or p.b not in window or coint is None:
            return PairScore(p, 0.0, {})
        try:
            score, pval, _ = coint(window[p.a], window[p.b])
        except Exception:
            pval, score = np.nan, np.nan
        s = (1 - pval) if pval == pval else 0.0
        return PairScore(
            p,
            s if (pval == pval and pval < self.pvalue_threshold) else 0.0,
            {"coint_stat": float(score) if score == score else None,
             "pvalue": float(pval) if pval == pval else None,
             "lookback": self.lookback, "p_thr": self.pvalue_threshold},
        )

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        window = prices.tail(self.lookback)
        if _HAS_JOBLIB:
            return Parallel(n_jobs=-1, backend="threading")(
                _delayed(self._score_one)(window, p) for p in candidates
            )
        return [self._score_one(window, p) for p in candidates]


class CombinedCriteriaSelector(PairSelector):
    """Sarmento & Horta (2021)-style **combined filters**:

    Conditions (defaults):
    - pvalue < 0.05 (cointegration)
    - Hurst(spread) < 0.5 (mean-reverting)
    - half-life < 60 bars
    - hits >= 3 (number of 2σ excursions within lookback)

    You can tune thresholds per your market/frequency.
    """
    name = "Combined Criteria (Sarmento–Horta)"

    def __init__(self, p_thr: float = 0.05, hurst_max: float = 0.5, halflife_max: int = 60, min_hits: int = 3, lookback: int = 252):
        self.p_thr = p_thr
        self.hurst_max = hurst_max
        self.halflife_max = halflife_max
        self.min_hits = min_hits
        self.lookback = lookback

    def _score_one(self, window: pd.DataFrame, p: Pair) -> PairScore:
        if p.a not in window or p.b not in window:
            return PairScore(p, 0.0, {})
        a, b = window[p.a], window[p.b]
        if coint is None:
            pval = np.nan
        else:
            try:
                _, pval, _ = coint(a, b)
            except Exception:
                pval = np.nan
        spread = a - b
        hurst = _hurst_rs(spread)
        hl = _halflife(spread)
        s_mean = spread.rolling(60).mean()
        s_std = spread.rolling(60).std(ddof=0)
        hits = int(((spread - s_mean).abs() > 2 * (s_std + 1e-9)).sum())
        ok = (pval == pval and pval < self.p_thr) and (hurst < self.hurst_max) and (
            hl < self.halflife_max) and (hits >= self.min_hits)
        return PairScore(
            p, float(1.0 if ok else 0.0),
            {"pvalue": float(pval) if pval == pval else None,
             "hurst": float(hurst),
             "halflife": float(hl) if np.isfinite(hl) else None,
             "hits": hits, "lookback": self.lookback},
        )

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        window = prices.tail(self.lookback)
        if _HAS_JOBLIB:
            return Parallel(n_jobs=-1, backend="threading")(
                _delayed(self._score_one)(window, p) for p in candidates
            )
        return [self._score_one(window, p) for p in candidates]
