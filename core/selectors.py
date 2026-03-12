"""Stage 1 (Pair Selection) models for the Pairs Trading app.

This module implements multiple pair selection methodologies based on academic research
and industry best practices. Each selector evaluates potential stock pairs for their
suitability in pairs trading strategies.

Key Features:
- Multiple selection criteria (correlation, distance, cointegration, ML)
- Ensemble-ready scoring system with detailed metadata
- Graceful degradation when optional dependencies are unavailable
- Literature-backed parameter defaults with customization options

Academic Sources:
- Gatev et al. (2006): Distance-based pair selection methodology
- Engle & Granger (1987): Cointegration testing framework  
- Sarmento & Horta (2021): Multi-criteria decision making approach
- Various ML approaches: Feature engineering for pair selection

Includes:
- Data classes: Pair, PairScore
- Abstract base: PairSelector
- Implementations:
  * CorrelationSelector: Pearson correlation-based selection
  * DistanceSelector: Gatev et al. (2006) normalized distance method
  * CointegrationSelector: Engle-Granger cointegration testing
  * CombinedCriteriaSelector: Sarmento & Horta (2021) multi-criteria approach
  * MLSelector: Supervised learning with engineered features

Notes:
- Keep this module free of Streamlit/UI code for modularity
- Statsmodels/scikit-learn are optional; classes degrade gracefully if unavailable
- See docstrings & __init__ defaults for literature-backed specifications
- All selectors return PairScore objects for consistent ensemble integration
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

try:
    import os as _os
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Bidirectional, Dense, Dropout,
        MultiHeadAttention, LayerNormalization,
        GlobalAveragePooling1D, Add, Lambda, Input,
    )
    from tensorflow.keras.callbacks import EarlyStopping
    _HAS_TF = True
except Exception:  # pragma: no cover
    _HAS_TF = False


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


# ---------------------------------------------
# Implementations
# ---------------------------------------------

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
                                     {"coint_stat": float(score) if score == score else None,
                                      "pvalue": float(pval) if pval == pval else None,
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
                out.append(PairScore(p, 0.0, {}))
                continue
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
            ok = (pval == pval and pval < self.p_thr) and (hurst < self.hurst_max) and (
                hl < self.halflife_max) and (hits >= self.min_hits)
            out.append(PairScore(p, float(1.0 if ok else 0.0),
                                 {"pvalue": float(pval) if pval == pval else None,
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
        r_a = a.pct_change()
        r_b = b.pct_change()
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
        r_a = a.pct_change()
        r_b = b.pct_change()
        spread_ret = (r_a - r_b).shift(-1).rolling(self.horizon).sum().iloc[-1]
        return int(1 if spread_ret == spread_ret and spread_ret > 0 else 0)

    def _year_splits(self, idx: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        years = pd.to_datetime(idx).year
        uniq = list(dict.fromkeys(years))  # preserve order
        if len(uniq) >= 4:
            train_years = uniq[:-2]
            val_year = uniq[-2]
            test_year = uniq[-1]
            train_mask = years.isin(train_years)
            val_mask = years == val_year
            test_mask = years == test_year
        else:
            n = len(idx)
            train_mask = np.zeros(n, dtype=bool)
            train_mask[: int(0.6 * n)] = True
            val_mask = np.zeros(n, dtype=bool)
            val_mask[int(0.6 * n): int(0.8 * n)] = True
            test_mask = np.zeros(n, dtype=bool)
            test_mask[int(0.8 * n):] = True
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
            self.model = TrivialSelectorModel(
                p1=max(1e-6, min(1.0 - 1e-6, p1)))
            self.model.fit(X, y)
            return self

        # Optional: very light rebalancing if super-skewed (e.g., > 5:1)
        c0 = counts_map.get(0, 0)
        c1 = counts_map.get(1, 0)
        maj_label = 0 if c0 >= c1 else 1
        min_label = 1 - maj_label
        maj_count, min_count = (c0, c1) if maj_label == 0 else (c1, c0)
        if min_count > 0 and maj_count / max(1, min_count) > self.rebalance_if_ratio_gt:
            # Undersample majority to at most ratio 2:1 to keep signal & speed
            target_maj = int(2.0 * min_count)
            maj_idx = np.where(y == maj_label)[0]
            min_idx = np.where(y == min_label)[0]
            keep_maj = np.random.RandomState(42).choice(
                maj_idx, size=target_maj, replace=False)
            keep_idx = np.sort(np.concatenate([keep_maj, min_idx]))
            X = X.iloc[keep_idx].reset_index(drop=True)
            y = y[keep_idx]
            print(
                f"[MLSelector.fit] Rebalanced from {maj_count}:{min_count} -> {np.sum(y==maj_label)}:{np.sum(y==min_label)}")

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
            a = prices[p.a].dropna()
            b = prices[p.b].dropna()
            idx = a.index.intersection(b.index)
            a, b = a.reindex(idx).ffill(), b.reindex(idx).ffill()
            if len(idx) < 260:
                out.append(PairScore(p, 0.0, {}))
                continue
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


# ---------------------------------------------
# LSTM / BiLSTM selector
# ---------------------------------------------

class LSTMSelector(PairSelector):
    """LSTM / BiLSTM pair selector using temporal spread features.

    Trains a sequence model on sliding windows of multivariate pair features
    to predict whether each pair will be profitable over a forward horizon.
    The Bidirectional variant (bidirectional=True) processes the sequence in
    both temporal directions, capturing patterns visible from both ends of a
    window (e.g., divergence followed by convergence).

    Architecture
    ------------
    Input → [Bi]LSTM(units) → Dropout → Dense(16, relu) → Dense(1, sigmoid)

    Features per timestep (6 total, computed for each pair A, B):
        corr_20       : 20-bar rolling Pearson correlation of returns
        corr_60       : 60-bar rolling Pearson correlation of returns
        spread_z      : z-score of price spread (A − B), 60-bar window
        vol_ratio     : rolling volatility ratio σ_A / σ_B (20-bar)
        price_ratio_z : z-score of price ratio A / B (60-bar)
        beta          : rolling OLS β of A on B (60-bar cov/var)

    Training label: 1 if Σ(r_A − r_B) over the next `horizon` bars > 0, else 0.
    Sliding windows are built per pair; pairs are mixed and shuffled for training.

    Temporal integrity: training uses all data except the last 252 bars,
    keeping the most recent year for out-of-sample scoring — consistent with
    the split used in MLSelector.

    Parameters
    ----------
    seq_len       : int   — input window length in bars (default 60 ≈ 3 months)
    bidirectional : bool  — BiLSTM if True, plain LSTM if False
    units         : int   — LSTM hidden units (default 32, kept small for speed)
    epochs        : int   — maximum training epochs; EarlyStopping may halt earlier
    batch_size    : int   — mini-batch size
    horizon       : int   — forward bars defining the profitability label
    dropout       : float — dropout rate after the LSTM layer

    Sources
    -------
    - Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
      Neural Computation, 9(8), 1735–1780.
    - Schuster, M. & Paliwal, K.K. (1997). "Bidirectional recurrent neural
      networks." IEEE Transactions on Signal Processing, 45(11), 2673–2681.
    - Fischer, T. & Krauss, C. (2018). "Deep learning with long short-term
      memory networks for financial market predictions." European Journal of
      Operational Research, 270(2), 654–669.
      [Establishes the sliding-window, binary-label LSTM approach used here]
    """

    name = "LSTM/BiLSTM"
    _N_FEATURES = 6

    def __init__(
        self,
        seq_len: int = 60,
        bidirectional: bool = True,
        units: int = 32,
        epochs: int = 20,
        batch_size: int = 32,
        horizon: int = 20,
        dropout: float = 0.2,
    ):
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.dropout = dropout
        self.model: Optional[object] = None

    def _pair_feature_series(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """Return a (T × 6) DataFrame of temporal features for pair (A, B)."""
        r_a = a.pct_change()
        r_b = b.pct_change()

        corr_20 = r_a.rolling(20).corr(r_b)
        corr_60 = r_a.rolling(60).corr(r_b)

        spread = a - b
        sp_m = spread.rolling(60).mean()
        sp_s = spread.rolling(60).std(ddof=0)
        spread_z = (spread - sp_m) / (sp_s + 1e-9)

        vol_ratio = r_a.rolling(20).std(ddof=0) / (r_b.rolling(20).std(ddof=0) + 1e-9)

        ratio = a / (b + 1e-9)
        rat_m = ratio.rolling(60).mean()
        rat_s = ratio.rolling(60).std(ddof=0)
        price_ratio_z = (ratio - rat_m) / (rat_s + 1e-9)

        beta = r_a.rolling(60).cov(r_b) / (r_b.rolling(60).var() + 1e-9)

        df = pd.DataFrame({
            "corr_20":       corr_20,
            "corr_60":       corr_60,
            "spread_z":      spread_z,
            "vol_ratio":     vol_ratio,
            "price_ratio_z": price_ratio_z,
            "beta":          beta,
        })
        return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    def _make_sequences(
        self, a: pd.Series, b: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sliding-window (X, y) arrays for one pair."""
        feats = self._pair_feature_series(a, b).values
        r_spread = (a.pct_change() - b.pct_change()).fillna(0.0).values
        T = len(feats)
        xs, ys = [], []
        for t in range(self.seq_len, T - self.horizon):
            xs.append(feats[t - self.seq_len: t])
            ys.append(1 if r_spread[t: t + self.horizon].sum() > 0 else 0)
        if not xs:
            empty_x = np.empty((0, self.seq_len, self._N_FEATURES), dtype=np.float32)
            return empty_x, np.empty(0, dtype=np.int32)
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)

    def _build_model(self) -> object:
        model = Sequential()
        if self.bidirectional:
            model.add(Bidirectional(
                LSTM(self.units),
                input_shape=(self.seq_len, self._N_FEATURES),
            ))
        else:
            model.add(LSTM(self.units, input_shape=(self.seq_len, self._N_FEATURES)))
        model.add(Dropout(self.dropout))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def fit(self, prices: pd.DataFrame) -> "LSTMSelector":
        if not _HAS_TF:
            print("[LSTMSelector] TensorFlow unavailable; will return neutral scores.")
            self.model = None
            return self

        min_len = self.seq_len + self.horizon + 60
        if len(prices) < min_len:
            print(f"[LSTMSelector] Insufficient rows ({len(prices)} < {min_len}); skipping fit.")
            self.model = None
            return self

        # Temporal split: train on all data except last 252 bars (out-of-sample)
        split = max(min_len, len(prices) - 252)
        train = prices.iloc[:split]
        cols = list(train.columns)

        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = train[cols[i]].dropna()
                b = train[cols[j]].dropna()
                common = a.index.intersection(b.index)
                if len(common) < min_len:
                    continue
                X, y = self._make_sequences(
                    a.reindex(common).ffill(), b.reindex(common).ffill()
                )
                if len(X):
                    all_X.append(X)
                    all_y.append(y)

        if not all_X:
            print("[LSTMSelector] No training sequences produced; skipping fit.")
            self.model = None
            return self

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        unique, counts = np.unique(y, return_counts=True)
        print(f"[LSTMSelector] {len(X)} sequences, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

        if len(unique) < 2:
            print("[LSTMSelector] Single-class labels; skipping fit.")
            self.model = None
            return self

        # Shuffle pairs (within-pair temporal order was respected when building windows)
        idx = np.random.RandomState(42).permutation(len(X))
        X, y = X[idx], y[idx]

        tf.random.set_seed(42)
        self.model = self._build_model()
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        self.model.fit(
            X, y.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=[es],
            verbose=0,
        )
        print("[LSTMSelector] Training complete.")
        return self

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        out: List[PairScore] = []

        if not _HAS_TF or self.model is None:
            for p in candidates:
                out.append(PairScore(p, 0.0, {"lstm": False}))
            return out

        for p in candidates:
            if p.a not in prices.columns or p.b not in prices.columns:
                out.append(PairScore(p, 0.0, {}))
                continue

            a = prices[p.a].dropna()
            b = prices[p.b].dropna()
            common = a.index.intersection(b.index)
            if len(common) < self.seq_len + 60:
                out.append(PairScore(p, 0.0, {"lstm": False, "reason": "insufficient data"}))
                continue

            feats = self._pair_feature_series(
                a.reindex(common).ffill(), b.reindex(common).ffill()
            ).values

            # Score using the most recent seq_len window
            window = feats[-self.seq_len:].astype(np.float32)
            if window.shape[0] < self.seq_len:
                out.append(PairScore(p, 0.0, {}))
                continue

            X = window[np.newaxis, ...]  # shape (1, seq_len, n_features)
            proba = float(self.model.predict(X, verbose=0)[0, 0])
            out.append(PairScore(p, proba, {
                "lstm_proba": proba,
                "bidirectional": self.bidirectional,
                "seq_len": self.seq_len,
            }))

        return out


# ---------------------------------------------
# Transformer selector
# ---------------------------------------------

class TransformerSelector(PairSelector):
    """Transformer encoder pair selector using temporal spread features.

    Applies a multi-head self-attention encoder to the same 6-feature time
    series used by LSTMSelector. Unlike the LSTM, the Transformer attends
    directly to any position in the window in O(1) sequential depth, making
    it effective at capturing non-local patterns (e.g., a spread that peaked
    120 bars ago still influences the current reading).

    Architecture
    ------------
    Input (seq_len × 6)
    → Dense(embed_dim)                      [linear feature projection]
    → + Sinusoidal Positional Encoding      [inject temporal order]
    → TransformerBlock × num_layers:
          MultiHeadAttention → Add & Norm
          Feed-Forward (Dense-ReLU-Dense) → Add & Norm
    → GlobalAveragePooling1D                [aggregate temporal axis]
    → Dense(16, relu) → Dense(1, sigmoid)

    Positional encoding uses the fixed sinusoidal scheme from Vaswani et al.
    (2017), which generalises to unseen sequence lengths and requires no
    additional parameters.

    Features per timestep (6, identical to LSTMSelector):
        corr_20, corr_60, spread_z, vol_ratio, price_ratio_z, beta

    Training label, temporal split, and sliding-window construction follow
    exactly the same protocol as LSTMSelector for fair ensemble comparison.

    Parameters
    ----------
    seq_len    : int   — input window length in bars (default 60)
    embed_dim  : int   — projection dimension; must be divisible by num_heads
    num_heads  : int   — number of attention heads
    ff_dim     : int   — inner dimension of the position-wise feed-forward layer
    num_layers : int   — number of stacked Transformer encoder blocks
    dropout    : float — dropout rate inside attention and feed-forward layers
    epochs     : int   — maximum training epochs
    batch_size : int   — mini-batch size
    horizon    : int   — forward bars for the profitability label

    Sources
    -------
    - Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 30.
      [Original Transformer architecture and sinusoidal positional encoding]
    - Zerveas, G. et al. (2021). "A Transformer-based Framework for
      Multivariate Time Series Representation Learning." KDD 2021, 2114–2124.
      [Establishes Transformer encoder + global pooling for time-series
       classification; directly informs the architecture used here]
    - Wen, Q. et al. (2023). "Transformers in Time Series: A Survey."
      IJCAI 2023. [Survey situating the design choices made in this module]
    """

    name = "Transformer"
    _N_FEATURES = 6

    def __init__(
        self,
        seq_len: int = 60,
        embed_dim: int = 32,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 32,
        horizon: int = 20,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.model: Optional[object] = None

    # ------------------------------------------------------------------
    # Feature engineering (identical to LSTMSelector for fair comparison)
    # ------------------------------------------------------------------

    def _pair_feature_series(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """Return a (T × 6) DataFrame of temporal features for pair (A, B)."""
        r_a = a.pct_change()
        r_b = b.pct_change()

        corr_20 = r_a.rolling(20).corr(r_b)
        corr_60 = r_a.rolling(60).corr(r_b)

        spread = a - b
        sp_m = spread.rolling(60).mean()
        sp_s = spread.rolling(60).std(ddof=0)
        spread_z = (spread - sp_m) / (sp_s + 1e-9)

        vol_ratio = r_a.rolling(20).std(ddof=0) / (r_b.rolling(20).std(ddof=0) + 1e-9)

        ratio = a / (b + 1e-9)
        rat_m = ratio.rolling(60).mean()
        rat_s = ratio.rolling(60).std(ddof=0)
        price_ratio_z = (ratio - rat_m) / (rat_s + 1e-9)

        beta = r_a.rolling(60).cov(r_b) / (r_b.rolling(60).var() + 1e-9)

        df = pd.DataFrame({
            "corr_20":       corr_20,
            "corr_60":       corr_60,
            "spread_z":      spread_z,
            "vol_ratio":     vol_ratio,
            "price_ratio_z": price_ratio_z,
            "beta":          beta,
        })
        return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    def _make_sequences(
        self, a: pd.Series, b: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding-window (X, y) arrays for one pair."""
        feats = self._pair_feature_series(a, b).values
        r_spread = (a.pct_change() - b.pct_change()).fillna(0.0).values
        T = len(feats)
        xs, ys = [], []
        for t in range(self.seq_len, T - self.horizon):
            xs.append(feats[t - self.seq_len: t])
            ys.append(1 if r_spread[t: t + self.horizon].sum() > 0 else 0)
        if not xs:
            empty_x = np.empty((0, self.seq_len, self._N_FEATURES), dtype=np.float32)
            return empty_x, np.empty(0, dtype=np.int32)
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    @staticmethod
    def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
        """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Returns shape (1, seq_len, d_model) ready to broadcast over batches.
        """
        positions = np.arange(seq_len)[:, np.newaxis]          # (seq_len, 1)
        dims = np.arange(d_model)[np.newaxis, :]                # (1, d_model)
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        return angles[np.newaxis, :, :].astype(np.float32)      # (1, seq_len, d_model)

    def _build_model(self) -> object:
        """Construct the Transformer encoder model using the Keras functional API."""
        pos_enc = self._positional_encoding(self.seq_len, self.embed_dim)
        pos_const = tf.constant(pos_enc, dtype=tf.float32)

        inputs = Input(shape=(self.seq_len, self._N_FEATURES))

        # Project raw features into the embedding space
        x = Dense(self.embed_dim)(inputs)

        # Inject positional information (fixed, non-trainable)
        x = Lambda(lambda t: t + pos_const)(x)

        # Stack Transformer encoder blocks
        for _ in range(self.num_layers):
            # --- Multi-head self-attention sublayer ---
            attn = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout,
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

            # --- Position-wise feed-forward sublayer ---
            ff = Dense(self.ff_dim, activation="relu")(x)
            ff = Dense(self.embed_dim)(ff)
            x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))

        # Aggregate the temporal axis → scalar representation per sample
        x = GlobalAveragePooling1D()(x)
        x = Dense(16, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    # ------------------------------------------------------------------
    # fit / score_pairs  (same protocol as LSTMSelector)
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame) -> "TransformerSelector":
        if not _HAS_TF:
            print("[TransformerSelector] TensorFlow unavailable; will return neutral scores.")
            self.model = None
            return self

        min_len = self.seq_len + self.horizon + 60
        if len(prices) < min_len:
            print(f"[TransformerSelector] Insufficient rows ({len(prices)} < {min_len}); skipping fit.")
            self.model = None
            return self

        # Reserve last 252 bars as out-of-sample; train on everything before
        split = max(min_len, len(prices) - 252)
        train = prices.iloc[:split]
        cols = list(train.columns)

        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = train[cols[i]].dropna()
                b = train[cols[j]].dropna()
                common = a.index.intersection(b.index)
                if len(common) < min_len:
                    continue
                X, y = self._make_sequences(
                    a.reindex(common).ffill(), b.reindex(common).ffill()
                )
                if len(X):
                    all_X.append(X)
                    all_y.append(y)

        if not all_X:
            print("[TransformerSelector] No training sequences produced; skipping fit.")
            self.model = None
            return self

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        unique, counts = np.unique(y, return_counts=True)
        print(f"[TransformerSelector] {len(X)} sequences, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

        if len(unique) < 2:
            print("[TransformerSelector] Single-class labels; skipping fit.")
            self.model = None
            return self

        idx = np.random.RandomState(42).permutation(len(X))
        X, y = X[idx], y[idx]

        tf.random.set_seed(42)
        self.model = self._build_model()
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        self.model.fit(
            X, y.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=[es],
            verbose=0,
        )
        print("[TransformerSelector] Training complete.")
        return self

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        out: List[PairScore] = []

        if not _HAS_TF or self.model is None:
            for p in candidates:
                out.append(PairScore(p, 0.0, {"transformer": False}))
            return out

        for p in candidates:
            if p.a not in prices.columns or p.b not in prices.columns:
                out.append(PairScore(p, 0.0, {}))
                continue

            a = prices[p.a].dropna()
            b = prices[p.b].dropna()
            common = a.index.intersection(b.index)
            if len(common) < self.seq_len + 60:
                out.append(PairScore(p, 0.0, {"transformer": False, "reason": "insufficient data"}))
                continue

            feats = self._pair_feature_series(
                a.reindex(common).ffill(), b.reindex(common).ffill()
            ).values

            window = feats[-self.seq_len:].astype(np.float32)
            if window.shape[0] < self.seq_len:
                out.append(PairScore(p, 0.0, {}))
                continue

            X = window[np.newaxis, ...]  # shape (1, seq_len, n_features)
            proba = float(self.model.predict(X, verbose=0)[0, 0])
            out.append(PairScore(p, proba, {
                "transformer_proba": proba,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
            }))

        return out


# ---------------------------------------------
# Graph Neural Network selector
# ---------------------------------------------

class GNNSelector(PairSelector):
    """Graph Convolutional Network (GCN) pair selector.

    Models the stock universe as a weighted correlation graph and learns node
    embeddings that capture each stock's relationship to the full universe.
    Pair quality is then scored via a link-prediction head over those embeddings.

    Unlike LSTMSelector and TransformerSelector which operate on the spread
    time series for each pair in isolation, the GNN processes the entire
    universe simultaneously, capturing multi-hop relationships (A relates to C
    through a common neighbour B) and universe-level co-movement patterns.

    Graph construction
    ------------------
    Nodes : one per stock, with 6 statistical node features
    Edges : non-negative Pearson correlation of returns over the lookback window
    Â     : symmetrically normalised adjacency with self-loops
              Â = D^{-½}(A + I)D^{-½}

    GCN forward pass (Kipf & Welling, 2017)
    ----------------------------------------
    H⁽¹⁾ = ReLU(Â X  W₁)          shape (N, hidden_dim)
    H⁽²⁾ = ReLU(Â H⁽¹⁾ W₂)        shape (N, embed_dim)   ← node embeddings

    Link-prediction head (Zhang et al., 2018)
    ------------------------------------------
    For pair (i, j):  f = [hᵢ ‖ hⱼ ‖ hᵢ⊙hⱼ]          shape (3·embed_dim,)
                   score = σ(f W_link + b)

    The concatenation of the two embeddings plus their element-wise product
    captures directional asymmetry, magnitude, and interaction simultaneously.

    Node features (6 per stock, computed over the lookback window)
    --------------------------------------------------------------
    mean_ret  : mean daily return
    vol       : annualised return std
    skew      : return skewness
    kurt      : excess kurtosis
    momentum  : cumulative return over window
    price_z   : z-score of last price vs window mean

    Training
    --------
    n_snapshots evenly-spaced graph snapshots are taken across the training
    period. Each snapshot contributes N(N−1)/2 binary pair labels (forward
    spread profitability). All snapshots share a single weight set, providing
    inductive generalisation: the trained W₁, W₂ can be applied to any graph
    at inference — including universes of different size.

    Parameters
    ----------
    lookback    : int   — bars per snapshot for feature/adjacency computation
    hidden_dim  : int   — GCN first-layer width (W₁ columns)
    embed_dim   : int   — GCN output embedding width (W₂ columns)
    epochs      : int   — gradient-descent epochs over all snapshots
    lr          : float — Adam learning rate
    horizon     : int   — forward bars used to define the profitability label
    n_snapshots : int   — number of training graph snapshots

    Sources
    -------
    - Kipf, T.N. & Welling, M. (2017). "Semi-Supervised Classification with
      Graph Convolutional Networks." ICLR 2017.
      [GCN architecture and symmetrically normalised adjacency formula]
    - Zhang, M. & Chen, Y. (2018). "Link Prediction Based on Graph Neural
      Networks." NeurIPS 2018.
      [Link-prediction head: [hᵢ ‖ hⱼ ‖ hᵢ⊙hⱼ] feature construction]
    - Matsunaga, A., Suzumura, T., & Takahashi, T. (2019). "Exploring Graph
      Neural Networks for Stock Market Predictions with Rolling Window
      Analysis." NeurIPS 2019 Workshop on Robust AI in Financial Services.
      [Rolling-snapshot GNN training applied to stock-market prediction]
    """

    name = "GNN"
    _N_NODE_FEATURES = 6

    def __init__(
        self,
        lookback: int = 120,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        epochs: int = 50,
        lr: float = 0.01,
        horizon: int = 20,
        n_snapshots: int = 8,
    ):
        self.lookback    = lookback
        self.hidden_dim  = hidden_dim
        self.embed_dim   = embed_dim
        self.epochs      = epochs
        self.lr          = lr
        self.horizon     = horizon
        self.n_snapshots = n_snapshots
        # Trainable weights — set in fit()
        self.W1:     Optional[object] = None
        self.W2:     Optional[object] = None
        self.W_link: Optional[object] = None
        self.b_link: Optional[object] = None
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _node_features(self, prices: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """Return (N, 6) node feature matrix; one row per stock."""
        feats = []
        for c in cols:
            p = prices[c].dropna()
            r = p.pct_change().dropna()
            if len(r) < 5:
                feats.append(np.zeros(self._N_NODE_FEATURES, dtype=np.float32))
                continue
            mean_ret = float(r.mean())
            vol      = float(r.std(ddof=0) * np.sqrt(252))
            skew     = float(r.skew())   if len(r) >= 10 else 0.0
            kurt     = float(r.kurt())   if len(r) >= 10 else 0.0
            momentum = float((1 + r).prod() - 1)
            price_z  = float((p.iloc[-1] - p.mean()) / (p.std(ddof=0) + 1e-9))
            feats.append([mean_ret, vol, skew, kurt, momentum, price_z])
        return np.clip(np.array(feats, dtype=np.float32), -10.0, 10.0)

    def _adjacency(self, prices: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """Symmetrically normalised adjacency Â = D^{-½}(A+I)D^{-½}.

        Edge weights are clipped to [0, 1] so only positive correlations
        propagate information (negative correlations are not meaningful
        neighbours in the pairs trading context).
        """
        rets = prices[cols].pct_change().dropna()
        A = np.clip(rets.corr().values, 0.0, 1.0).astype(np.float32)
        A = A + np.eye(len(cols), dtype=np.float32)          # self-loops
        d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1) + 1e-9)
        D = np.diag(d_inv_sqrt)
        return (D @ A @ D).astype(np.float32)

    def _pair_labels(
        self,
        prices: pd.DataFrame,
        cols: List[str],
        fwd_start: int,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Binary profitability label for every pair from a forward window."""
        fwd = prices.iloc[fwd_start: fwd_start + self.horizon]
        pair_idx: List[Tuple[int, int]] = []
        labels: List[int] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i] not in fwd.columns or cols[j] not in fwd.columns:
                    continue
                ri = fwd[cols[i]].pct_change().dropna().values
                rj = fwd[cols[j]].pct_change().dropna().values
                n  = min(len(ri), len(rj))
                if n == 0:
                    continue
                pair_idx.append((i, j))
                labels.append(1 if (ri[:n] - rj[:n]).sum() > 0 else 0)
        return pair_idx, np.array(labels, dtype=np.float32)

    # ------------------------------------------------------------------
    # GCN forward (graph-size independent — works for any N)
    # ------------------------------------------------------------------

    def _gcn_forward(self, A_hat: np.ndarray, X: np.ndarray) -> "tf.Tensor":
        """Two-layer GCN: H⁽²⁾ = ReLU(Â·ReLU(Â·X·W₁)·W₂)."""
        A = tf.constant(A_hat, dtype=tf.float32)
        H = tf.constant(X,     dtype=tf.float32)
        H = tf.nn.relu(A @ H  @ self.W1)   # (N, hidden_dim)
        H = tf.nn.relu(A @ H  @ self.W2)   # (N, embed_dim)
        return H

    def _link_logits(
        self, H: "tf.Tensor", pair_idx: List[Tuple[int, int]]
    ) -> "tf.Tensor":
        """Score each pair via [hᵢ ‖ hⱼ ‖ hᵢ⊙hⱼ] → sigmoid."""
        feats = []
        for i, j in pair_idx:
            hi, hj = H[i], H[j]
            feats.append(tf.concat([hi, hj, hi * hj], axis=0))
        F = tf.stack(feats)                              # (n_pairs, 3·embed_dim)
        return tf.sigmoid(F @ self.W_link + self.b_link) # (n_pairs, 1)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame) -> "GNNSelector":
        if not _HAS_TF:
            print("[GNNSelector] TensorFlow unavailable; will return neutral scores.")
            self._trained = False
            return self

        min_len = self.lookback + self.horizon + 5
        if len(prices) < min_len:
            print(f"[GNNSelector] Insufficient rows ({len(prices)} < {min_len}); skipping fit.")
            self._trained = False
            return self

        # Same temporal split used by all other selectors
        train_end   = max(min_len, len(prices) - 252)
        train       = prices.iloc[:train_end]
        cols        = list(train.columns)
        N           = len(cols)

        if N < 2:
            print("[GNNSelector] Universe too small; skipping fit.")
            self._trained = False
            return self

        # Initialise trainable weights
        init        = tf.initializers.glorot_uniform(seed=42)
        self.W1     = tf.Variable(init((self._N_NODE_FEATURES, self.hidden_dim)), dtype=tf.float32)
        self.W2     = tf.Variable(init((self.hidden_dim,        self.embed_dim)),  dtype=tf.float32)
        self.W_link = tf.Variable(init((3 * self.embed_dim,    1)),                dtype=tf.float32)
        self.b_link = tf.Variable(tf.zeros((1,), dtype=tf.float32))
        trainable   = [self.W1, self.W2, self.W_link, self.b_link]

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Build evenly-spaced graph snapshots over the training period
        usable  = len(train) - self.horizon
        step    = max(1, (usable - self.lookback) // max(1, self.n_snapshots - 1))
        starts  = list(range(0, usable - self.lookback, step))[:self.n_snapshots]

        snapshots = []
        for s in starts:
            window                  = train.iloc[s: s + self.lookback]
            A_hat                   = self._adjacency(window, cols)
            X                       = self._node_features(window, cols)
            pair_idx, y             = self._pair_labels(train, cols, s + self.lookback)
            if len(pair_idx) == 0:
                continue
            snapshots.append((A_hat, X, pair_idx, y))

        if not snapshots:
            print("[GNNSelector] No valid snapshots; skipping fit.")
            self._trained = False
            return self

        all_y = np.concatenate([s[3] for s in snapshots])
        if len(np.unique(all_y)) < 2:
            print("[GNNSelector] Single-class labels across all snapshots; skipping fit.")
            self._trained = False
            return self

        print(f"[GNNSelector] Training on {len(snapshots)} snapshots, N={N} stocks.")
        tf.random.set_seed(42)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for A_hat, X, pair_idx, y in snapshots:
                y_t = tf.constant(y[:, np.newaxis], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    H      = self._gcn_forward(A_hat, X)
                    scores = self._link_logits(H, pair_idx)
                    loss   = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(y_t, scores)
                    )
                grads = tape.gradient(loss, trainable)
                optimizer.apply_gradients(zip(grads, trainable))
                epoch_loss += float(loss)

            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / len(snapshots)
                print(f"[GNNSelector] Epoch {epoch+1}/{self.epochs} — avg loss: {avg:.4f}")

        self._trained = True
        print("[GNNSelector] Training complete.")
        return self

    # ------------------------------------------------------------------
    # score_pairs
    # ------------------------------------------------------------------

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        out: List[PairScore] = []

        if not _HAS_TF or not self._trained:
            for p in candidates:
                out.append(PairScore(p, 0.0, {"gnn": False}))
            return out

        # Build current graph from the most recent `lookback` bars
        window  = prices.tail(self.lookback)
        cols    = list(window.columns)
        col_idx = {c: i for i, c in enumerate(cols)}

        A_hat   = self._adjacency(window, cols)
        X       = self._node_features(window, cols)
        H       = self._gcn_forward(A_hat, X).numpy()   # (N, embed_dim)

        for p in candidates:
            if p.a not in col_idx or p.b not in col_idx:
                out.append(PairScore(p, 0.0, {}))
                continue

            i, j    = col_idx[p.a], col_idx[p.b]
            hi, hj  = H[i], H[j]
            f       = np.concatenate([hi, hj, hi * hj]).astype(np.float32)
            f_t     = tf.constant(f[np.newaxis, :], dtype=tf.float32)  # (1, 3·embed_dim)
            proba   = float(
                tf.sigmoid(f_t @ self.W_link + self.b_link).numpy()[0, 0]
            )
            out.append(PairScore(p, proba, {
                "gnn_proba":   proba,
                "hidden_dim":  self.hidden_dim,
                "embed_dim":   self.embed_dim,
                "n_snapshots": self.n_snapshots,
            }))

        return out
