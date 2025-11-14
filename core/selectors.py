"""Stage 1 (Pair Selection) models for the Pairs Trading app.

Includes:
- Data classes: Pair, PairScore
- Abstract base: PairSelector
- Implementations:
  * CorrelationSelector
  * DistanceSelector (Gatev et al., 2006; default z-score L2 distance)
  * CointegrationSelector (Engle–Granger)
  * CombinedCriteriaSelector (Sarmento & Horta, 2021): cointegration + Hurst + half-life + hits
  * MLSelector (optional): simple supervised model with year-based Train/Val/Test split

Notes
-----
* Keep this module free of Streamlit/UI code.
* Statsmodels/scikit-learn are optional; classes degrade gracefully if unavailable.
* See docstrings & __init__ defaults for literature-backed specs and alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependencies
try:  # statsmodels for cointegration
    from statsmodels.tsa.stattools import coint
except Exception:  # pragma: no cover
    coint = None

try:  # supervised model
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # pragma: no cover
    GradientBoostingClassifier = None


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
    dt = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
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
    x_lag = x.shift(1).dropna(); y = x.loc[x_lag.index]
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


# ---------------------------------------------
# Implementations
# ---------------------------------------------

class CorrelationSelector(PairSelector):
    """Select pairs by **historical return correlation**.

    Defaults
    --------
    lookback: 252 trading days (≈ 1Y). Literature varies 60–504.
    """
    name = "Correlation"
    def __init__(self, lookback: int = 252):
        self.lookback = lookback
    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        window = prices.tail(self.lookback)
        corr = window.pct_change().corr().fillna(0.0)
        out: List[PairScore] = []
        for p in candidates:
            s = corr.loc[p.a, p.b] if p.a in corr.index and p.b in corr.columns else 0.0
            out.append(PairScore(p, float(s), {"corr": float(s), "lookback": self.lookback}))
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
                out.append(PairScore(p, -dist, {"neg_l2": -dist, "mode": self.mode, "lookback": self.lookback}))
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
    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        window = prices.tail(self.lookback)
        out: List[PairScore] = []
        for p in candidates:
            if p.a in window and p.b in window and coint is not None:
                try:
                    score, pval, _ = coint(window[p.a], window[p.b])
                except Exception:
                    pval, score = np.nan, np.nan
                s = (1 - pval) if pval == pval else 0.0
                out.append(PairScore(p, s if (pval == pval and pval < self.pvalue_threshold) else 0.0,
                                     {"coint_stat": float(score) if score==score else None,
                                      "pvalue": float(pval) if pval==pval else None,
                                      "lookback": self.lookback, "p_thr": self.pvalue_threshold}))
            else:
                out.append(PairScore(p, 0.0, {}))
        return out


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
    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        window = prices.tail(self.lookback)
        out: List[PairScore] = []
        for p in candidates:
            if p.a not in window or p.b not in window:
                out.append(PairScore(p, 0.0, {})); continue
            a, b = window[p.a], window[p.b]
            # Cointegration p-value
            if coint is None:
                pval = np.nan
            else:
                try:
                    _, pval, _ = coint(a, b)
                except Exception:
                    pval = np.nan
            spread = a - b
            # Hurst & half-life
            hurst = _hurst_rs(spread)
            hl = _halflife(spread)
            # Count 2σ excursions as a proxy for observable mean-reversion opportunities
            s_mean = spread.rolling(60).mean()
            s_std = spread.rolling(60).std(ddof=0)
            hits = int(((spread - s_mean).abs() > 2 * (s_std + 1e-9)).sum())
            ok = (pval == pval and pval < self.p_thr) and (hurst < self.hurst_max) and (hl < self.halflife_max) and (hits >= self.min_hits)
            out.append(PairScore(p, float(1.0 if ok else 0.0),
                                 {"pvalue": float(pval) if pval==pval else None,
                                  "hurst": float(hurst),
                                  "halflife": float(hl) if np.isfinite(hl) else None,
                                  "hits": hits,
                                  "lookback": self.lookback}))
        return out


# ---------------------------------------------
# Fallback model for ML selector
# ---------------------------------------------

@dataclass
class TrivialSelectorModel:
    """Fallback when y has <2 classes. Predicts constant proba = prior."""
    p1: float = 0.5  # prior P(y=1)
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        p1 = np.clip(float(self.p1), 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([np.full(len(X), p0), np.full(len(X), p1)])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------
# ML-based selector
# ---------------------------------------------

class MLSelector(PairSelector):
    """Supervised selector trained to predict profitable pairs.

    Features (fast baseline): corr20, corr60, vol_a, vol_b, ratio momentum, 1-pval
    Label: forward spread profitability over horizon bars (very simple proxy)

    Temporal split: Train (years[:-2]) / Val (year[-2]) / Test (year[-1]) if >= 4 distinct years,
    else 60/20/20 split by index length. This avoids look-ahead where possible.
    """
    name = "Supervised ML"
    def __init__(self, horizon: int = 20, rebalance_if_ratio_gt: float = 5.0):
        self.horizon = horizon
        self.model: Optional[object] = None
        self.features_: List[str] = []
        self.rebalance_if_ratio_gt = float(rebalance_if_ratio_gt)

    def _pair_features(self, a: pd.Series, b: pd.Series) -> Dict[str, float]:
        r_a = a.pct_change(); r_b = b.pct_change()
        corr20 = r_a.rolling(20).corr(r_b).iloc[-1]
        corr60 = r_a.rolling(60).corr(r_b).iloc[-1]
        vol_a = r_a.rolling(60).std(ddof=0).iloc[-1]
        vol_b = r_b.rolling(60).std(ddof=0).iloc[-1]
        ratio = (a / b).dropna()
        mom_ratio = ratio.pct_change(20).iloc[-1] if len(ratio) > 20 else 0.0
        pval = np.nan
        if coint is not None:
            try:
                _, pval, _ = coint(a, b)
            except Exception:
                pval = np.nan
        return {
            "corr20": float(corr20 if corr20 == corr20 else 0.0),
            "corr60": float(corr60 if corr60 == corr60 else 0.0),
            "vol_a": float(vol_a if vol_a == vol_a else 0.0),
            "vol_b": float(vol_b if vol_b == vol_b else 0.0),
            "mom_ratio20": float(mom_ratio if mom_ratio == mom_ratio else 0.0),
            "coint_1mp": float(0.0 if pval != pval else (1 - pval)),
        }

    def _label(self, a: pd.Series, b: pd.Series) -> int:
        # Simple forward profitability proxy over horizon bars
        r_a = a.pct_change(); r_b = b.pct_change()
        spread_ret = (r_a - r_b).shift(-1).rolling(self.horizon).sum().iloc[-1]
        return int(1 if spread_ret == spread_ret and spread_ret > 0 else 0)

    def _year_splits(self, idx: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        years = pd.to_datetime(idx).year
        uniq = list(dict.fromkeys(years))  # preserve order
        if len(uniq) >= 4:
            train_years = uniq[:-2]; val_year = uniq[-2]; test_year = uniq[-1]
            train_mask = years.isin(train_years)
            val_mask = years == val_year
            test_mask = years == test_year
        else:
            n = len(idx)
            train_mask = np.zeros(n, dtype=bool); train_mask[: int(0.6 * n)] = True
            val_mask = np.zeros(n, dtype=bool); val_mask[int(0.6 * n): int(0.8 * n)] = True
            test_mask = np.zeros(n, dtype=bool); test_mask[int(0.8 * n):] = True
        return train_mask, val_mask, test_mask

    def fit(self, prices: pd.DataFrame) -> "MLSelector":
        idx = prices.index
        if len(idx) < 400:
            self.model = None
            self.features_ = []
            return self

        train_mask, val_mask, test_mask = self._year_splits(idx)

        # Build training set on train_mask end of window (last values)
        feats: List[Dict[str, float]] = []
        labels: List[int] = []
        cols = list(prices.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = prices.loc[train_mask, cols[i]].dropna()
                b = prices.loc[train_mask, cols[j]].dropna()
                common = a.index.intersection(b.index)
                a, b = a.reindex(common).ffill(), b.reindex(common).ffill()
                if len(common) < 260:
                    continue
                feats.append(self._pair_features(a, b))
                labels.append(self._label(a, b))

        if not feats:
            self.model = None
            self.features_ = []
            return self

        X = pd.DataFrame(feats).fillna(0.0)
        y = np.asarray(labels, dtype=int)
        self.features_ = list(X.columns)

        # ---- NEW: class balance + safe fallback --------------------------------
        uniq, counts = np.unique(y, return_counts=True)
        counts_map = {int(k): int(v) for k, v in zip(uniq, counts)}
        print(f"[MLSelector.fit] y class counts: {counts_map}")

        if len(uniq) < 2:
            # Fall back to constant-probability model to avoid sklearn error
            p1 = float(y.mean()) if len(y) else 0.5
            self.model = TrivialSelectorModel(p1=max(1e-6, min(1.0 - 1e-6, p1)))
            self.model.fit(X, y)
            return self

        # Optional: very light rebalancing if super-skewed (e.g., > 5:1)
        c0 = counts_map.get(0, 0); c1 = counts_map.get(1, 0)
        maj_label = 0 if c0 >= c1 else 1
        min_label = 1 - maj_label
        maj_count, min_count = (c0, c1) if maj_label == 0 else (c1, c0)
        if min_count > 0 and maj_count / max(1, min_count) > self.rebalance_if_ratio_gt:
            # Undersample majority to at most ratio 2:1 to keep signal & speed
            target_maj = int(2.0 * min_count)
            maj_idx = np.where(y == maj_label)[0]
            min_idx = np.where(y == min_label)[0]
            keep_maj = np.random.RandomState(42).choice(maj_idx, size=target_maj, replace=False)
            keep_idx = np.sort(np.concatenate([keep_maj, min_idx]))
            X = X.iloc[keep_idx].reset_index(drop=True)
            y = y[keep_idx]
            print(f"[MLSelector.fit] Rebalanced from {maj_count}:{min_count} -> {np.sum(y==maj_label)}:{np.sum(y==min_label)}")

        # ------------------------------------------------------------------------

        # Choose model
        if _HAS_XGB:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=0,
            )
        else:
            if GradientBoostingClassifier is None:
                self.model = None
                return self
            model = GradientBoostingClassifier(random_state=42)

        model.fit(X, y)
        self.model = model
        return self

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        out: List[PairScore] = []
        if self.model is None or not self.features_:
            for p in candidates:
                out.append(PairScore(p, 0.0, {"ml": False}))
            return out
        for p in candidates:
            a = prices[p.a].dropna(); b = prices[p.b].dropna()
            idx = a.index.intersection(b.index)
            a, b = a.reindex(idx).ffill(), b.reindex(idx).ffill()
            if len(idx) < 260:
                out.append(PairScore(p, 0.0, {})); continue
            f = self._pair_features(a, b)
            X = pd.DataFrame([f])[self.features_].fillna(0.0)
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(X)[0, 1])
            elif hasattr(self.model, "decision_function"):
                proba = float(self.model.decision_function(X))
            else:
                proba = float(self.model.predict(X))
            out.append(PairScore(p, proba, {"ml_proba": proba}))
        return out
