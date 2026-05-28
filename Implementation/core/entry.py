"""Stage 2 (Entry/Exit) models for the Pairs Trading app.

This module implements trading signal generation models that determine when to enter
and exit pairs trading positions. Each model applies different mathematical frameworks
to identify mean reversion opportunities in price spreads.

Academic Foundations:
- Mean Reversion Theory: Statistical tendency of prices to return to historical means
- Ornstein-Uhlenbeck Process: Mathematical model for mean-reverting stochastic processes
- Kalman Filtering: Optimal estimation theory for time-varying parameters
- Z-Score Methodology: Standard statistical approach for outlier detection

Signal Convention:
All models return discrete signals in {+1, 0, -1} format:
- +1: Long stock A, Short stock B (spread expected to narrow)
- -1: Short stock A, Long stock B (spread expected to widen)  
-  0: No position (neutral/exit signal)

Includes:
- Base class: EntryExitModel with standardized interface
- Implementations:
  * ZScoreThreshold: Classic mean-reversion bands on spread z-scores
  * OUThreshold: Ornstein-Uhlenbeck inspired thresholding with estimated parameters
  * KalmanHedge: Dynamic hedge ratio via Linear Kalman Filter state-space model
  * MLSignal: Gradient-boosted triclass classifier on 11 spread features

Key Features:
- Pure signal generation without portfolio management concerns
- Configurable parameters with literature-backed defaults
- Robust error handling and fallback mechanisms
- Support for both unit and dynamic hedge ratios

Sources:
- Uhlenbeck & Ornstein (1930): "On the theory of Brownian motion"
- Kalman (1960): "A new approach to linear filtering and prediction problems"
- Standard statistical literature for z-score methodology

Notes:
- Pure signal generation here; no portfolio sizing, costs, or stop rules
- Spread definition defaults to A - B (unit hedge) for simplicity and robustness
- For dynamic hedging, use KalmanHedge or extend with OLS beta estimation
- All models include generate_signals() method for consistency with prediction engine
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Optional ML dependencies -----------------------------------------------
try:
    from xgboost import XGBClassifier as _XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier as _GBMClassifier
    from sklearn.preprocessing import LabelEncoder as _LabelEncoder
    _HAS_GBM = True
except Exception:
    _HAS_GBM = False


# ---------------------------------------------
# Base
# ---------------------------------------------


class EntryExitModel:
    """Abstract entry/exit model producing discrete trading signals."""
    name: str = "base"

    def fit(self, a: pd.Series, b: pd.Series) -> "EntryExitModel":
        """Optional pre-fit step (e.g., estimate parameters)."""
        return self

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Compute a signal time-series in {+1, 0, -1} indexed like inputs."""
        raise NotImplementedError

    def generate_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Generate signals (alias for trade_signals for consistency)."""
        return self.trade_signals(a, b)


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def _zscore(x: pd.Series, lookback: int) -> pd.Series:
    m = x.rolling(lookback).mean()
    s = x.rolling(lookback).std(ddof=0)
    return (x - m) / (s + 1e-9)


def _rolling_beta(a: pd.Series, b: pd.Series, lookback: int = 100) -> pd.Series:
    """Rolling OLS beta of A on B (no intercept). Fallback to 1.0 when insufficient data."""
    cov = a.rolling(lookback).cov(b)
    var = b.rolling(lookback).var()
    beta = cov / (var + 1e-12)
    return beta.replace([np.inf, -np.inf], np.nan).fillna(1.0)


# ---------------------------------------------
# Implementations
# ---------------------------------------------

@dataclass
class ZScoreThreshold(EntryExitModel):
    """Mean-reversion bands on a simple spread.

    Parameters
    ----------
    lookback : int
        Rolling window for mean/std of spread (default 60 bars).
    entry_z : float
        Enter when |z| >= entry_z.
    exit_z : float
        Flatten when |z| < exit_z.
    hedge : str
        'unit' => spread = A - B
        'ols'  => spread = A - beta*B (rolling beta)
    beta_lb : int
        Lookback for rolling beta if hedge='ols'.
    """
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    hedge: str = "unit"  # or 'ols'
    beta_lb: int = 100

    name: str = "Mean Reversion (±2σ)"

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if self.hedge == "ols":
            beta = _rolling_beta(a, b, self.beta_lb)
            spread = a - beta * b
        else:
            spread = a - b
        z = _zscore(spread, self.lookback)
        sig = pd.Series(np.nan, index=spread.index)
        sig[z > self.entry_z] = -1  # spread high => short A, long B
        sig[z < -self.entry_z] = +1
        sig[(z.abs() < self.exit_z)] = 0
        return sig.ffill().fillna(0).astype(int)


@dataclass
class OUThreshold(EntryExitModel):
    """OU/AR(1)-inspired thresholding.

    We estimate a reversion speed via AR(1) on demeaned spread and
    scale deviations by that speed. Thresholds are set in std units
    of this metric.
    """
    lookback: int = 252
    entry_k: float = 1.5
    exit_k: float = 0.2
    hedge: str = "unit"  # or 'ols'
    beta_lb: int = 100

    name: str = "OU Model"

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if self.hedge == "ols":
            beta = _rolling_beta(a, b, self.beta_lb)
            spread = a - beta * b
        else:
            spread = a - b

        mu = spread.rolling(self.lookback).mean()
        x = spread - mu  # demeaned spread

        # Rolling AR(1) phi estimate: cov(x_t, x_{t-1}) / var(x_{t-1})
        # Using rolling windows ensures we only use past data at each bar (no look-ahead).
        x_lag = x.shift(1)
        rolling_cov = x.rolling(self.lookback).cov(x_lag)
        rolling_var = x_lag.rolling(self.lookback).var()
        phi = (rolling_cov / (rolling_var + 1e-12)).clip(-0.9999, 0.9999)

        # Half-life and mean-reversion speed k = 1/halflife from rolling phi
        # log(0) is undefined so clip phi away from 0 and ±1
        safe_phi = phi.abs().clip(1e-6, 0.9999)
        halflife = -1.0 / np.log(safe_phi)  # element-wise; >0 when phi in (0,1)
        k = (1.0 / halflife).fillna(0.0).clip(0.0)  # non-negative speed

        metric = (k * x).fillna(0)
        s = metric.rolling(self.lookback // 2).std(ddof=0).bfill().fillna(0)

        sig = pd.Series(np.nan, index=spread.index)
        sig[metric > self.entry_k * s] = -1
        sig[metric < -self.entry_k * s] = +1
        sig[(metric.abs() < self.exit_k * s)] = 0
        return sig.ffill().fillna(0).astype(int)


@dataclass
class KalmanHedge(EntryExitModel):
    """Dynamic hedge ratio estimation via a Linear Kalman Filter.

    State-space model:
        Observation:  P_a(t) = β(t)·P_b(t) + α(t) + ε(t),   ε ~ N(0, R)
        Transition:   [β(t), α(t)] = [β(t-1), α(t-1)] + η(t), η ~ N(0, Q)

    The state [β, α] is treated as a 2D random walk (F = I). The observation
    matrix H_t = [P_b(t), 1] is time-varying, making this a non-stationary
    dynamic linear regression solved exactly by the Kalman recursion.

    The Kalman innovation e_t = P_a(t) − H_t @ x̂_{t|t−1} is the model's
    spread estimate. Signals are derived from z_t = e_t / √S_t, where S_t is
    the innovation variance predicted by the filter.

    Parameters
    ----------
    delta : float
        Process noise magnitude. Controls how quickly β and α can adapt.
        Q = (delta / (1 − delta)) · I₂. Smaller δ → slower β adaptation.
        Typical range for daily equity pairs: 1e-5 to 1e-3 (Pole, 2007).
    R_noise : float
        Observation noise variance. Larger R → smoother hedge ratio estimates.
    entry_z : float
        Enter when |z| ≥ entry_z (spread significantly displaced).
    exit_z : float
        Exit when |z| < exit_z (spread reverted toward zero).

    Sources
    -------
    - Kalman, R.E. (1960). "A new approach to linear filtering and prediction
      problems." Journal of Basic Engineering, 82(1), 35–45.
    - Elliott, R.J., van der Hoek, J., & Malcolm, W.P. (2005). "Pairs trading."
      Quantitative Finance, 5(3), 271–276.
    - Pole, A. (2007). Statistical Arbitrage: Algorithmic Trading Insights and
      Techniques. Wiley Finance. [Chapter 4 — practical δ / R calibration]
    """
    delta: float = 1e-4
    R_noise: float = 1e-2
    entry_z: float = 2.0
    exit_z: float = 0.5

    name: str = "Kalman Hedge"

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        idx = a.index.intersection(b.index)
        a_vals = a.reindex(idx).ffill().values.astype(float)
        b_vals = b.reindex(idx).ffill().values.astype(float)
        n = len(a_vals)

        if n < 20:
            return pd.Series(0, index=idx)

        # Process noise covariance: Q = (δ / (1−δ)) · I₂
        Q = np.eye(2) * (self.delta / (1.0 - self.delta))
        R = float(self.R_noise)

        # Initial state [β=0, α=0] and a diffuse (large) prior covariance
        x = np.zeros(2)
        P = np.eye(2)

        innovations = np.empty(n)
        innov_stds = np.empty(n)

        for t in range(n):
            H = np.array([b_vals[t], 1.0])  # observation vector (1×2)

            # Predict (F = I, so x_pred = x, P_pred = P + Q)
            P_pred = P + Q

            # Innovation and its variance
            e = a_vals[t] - float(H @ x)
            S = float(H @ P_pred @ H) + R  # scalar innovation variance

            # Kalman gain (2×1)
            K = (P_pred @ H) / S

            # Update state and covariance
            x = x + K * e
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            innovations[t] = e
            innov_stds[t] = math.sqrt(max(S, 1e-12))

        z = pd.Series(innovations / innov_stds, index=idx)

        sig = pd.Series(np.nan, index=idx)
        sig[z > self.entry_z] = -1   # spread above mean → short A, long B
        sig[z < -self.entry_z] = 1   # spread below mean → long A, short B
        sig[z.abs() < self.exit_z] = 0

        return sig.ffill().fillna(0).astype(int)


class MLSignal(EntryExitModel):
    """Gradient-boosted machine for triclass spread-signal prediction.

    Constructs 11 hand-crafted spread features per bar and trains an
    XGBoost (or sklearn GradientBoosting fallback) classifier to predict
    whether the next *horizon* bars will produce a profitable long (+1),
    short (-1), or flat (0) spread trade.

    Features
    --------
    spread_z     : rolling z-score of spread (lookback)
    z_lag5       : spread_z lagged 5 bars
    z_lag20      : spread_z lagged 20 bars
    velocity     : 1-bar change in spread_z
    acceleration : 1-bar change in velocity
    abs_z        : |spread_z|  (magnitude without direction)
    corr_20      : 20-bar rolling Pearson correlation of A and B
    corr_60      : 60-bar rolling Pearson correlation of A and B
    vol_ratio    : rolling std(A) / rolling std(B)  (relative volatility)
    momentum_a   : 20-bar log-return of A
    momentum_b   : 20-bar log-return of B

    Label
    -----
    +1 if Σ(r_A − r_B)[t:t+horizon] > neutral_pct  → long spread
    -1 if Σ(r_A − r_B)[t:t+horizon] < −neutral_pct → short spread
     0 otherwise

    Train / Predict Split
    ---------------------
    The first *train_frac* of valid bars are used for training; signals
    are generated for all bars (NaN before sufficient history).

    Fallback
    --------
    If neither XGBoost nor sklearn is available, or if there are fewer
    than *min_train* training samples, falls back to ZScoreThreshold.

    Sources
    -------
    - Friedman, J.H. (2001). "Greedy function approximation: a gradient
      boosting machine." Annals of Statistics, 29(5), 1189–1232.
    - Chen, T. & Guestrin, C. (2016). "XGBoost: A scalable tree boosting
      system." Proceedings of KDD 2016, 785–794.
    - Krauss, C., Do, X.A., & Huck, N. (2017). "Deep neural networks,
      gradient-boosted trees, random forests: Statistical arbitrage on the
      S&P 500." European Journal of Operational Research, 259(2), 689–702.
    """

    name: str = "ML Signal"

    def __init__(
        self,
        lookback: int = 60,
        horizon: int = 5,
        neutral_pct: float = 0.0,
        train_frac: float = 0.70,
        min_train: int = 200,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon
        self.neutral_pct = neutral_pct
        self.train_frac = train_frac
        self.min_train = min_train
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model = None
        self._label_offset = 0
        self._label_encoder = None  # LabelEncoder for XGBoost (handles non-contiguous classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """Compute the 11-column feature matrix."""
        idx = a.index.intersection(b.index)
        a = a.reindex(idx).ffill()
        b = b.reindex(idx).ffill()

        spread = a - b
        z = _zscore(spread, self.lookback)
        vel = z.diff(1)
        acc = vel.diff(1)

        r_a = np.log(a / a.shift(20))
        r_b = np.log(b / b.shift(20))

        std_a = a.rolling(self.lookback).std(ddof=0)
        std_b = b.rolling(self.lookback).std(ddof=0)

        return pd.DataFrame(
            {
                "spread_z": z,
                "z_lag5": z.shift(5),
                "z_lag20": z.shift(20),
                "velocity": vel,
                "acceleration": acc,
                "abs_z": z.abs(),
                "corr_20": a.rolling(20).corr(b),
                "corr_60": a.rolling(60).corr(b),
                "vol_ratio": std_a / (std_b + 1e-9),
                "momentum_a": r_a,
                "momentum_b": r_b,
            },
            index=idx,
        )

    def _build_labels(self, a: pd.Series, b: pd.Series, idx) -> pd.Series:
        """Triclass label: +1 / -1 / 0 based on forward spread return."""
        r_a = a.reindex(idx).ffill().pct_change()
        r_b = b.reindex(idx).ffill().pct_change()
        fwd = (r_a - r_b).shift(-self.horizon).rolling(self.horizon).sum()
        label = pd.Series(0, index=idx)
        label[fwd > self.neutral_pct] = 1
        label[fwd < -self.neutral_pct] = -1
        return label

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, a: pd.Series, b: pd.Series) -> "MLSignal":
        """Train the classifier on the first *train_frac* of valid bars."""
        feat = self._build_features(a, b)
        labels = self._build_labels(a, b, feat.index)

        mask = feat.notna().all(axis=1) & labels.notna()
        X = feat[mask].values
        y = labels[mask].values

        n_train = int(len(X) * self.train_frac)
        log.debug(f"MLSignal.fit: n_valid={len(X)}, n_train={n_train}, min_train={self.min_train}, _HAS_XGB={_HAS_XGB}")
        if n_train < self.min_train:
            log.warning(f"MLSignal.fit: insufficient training samples (n_train={n_train} < min_train={self.min_train}) — falling back to ZScoreThreshold")
            self._model = None
            return self

        X_tr, y_tr = X[:n_train], y[:n_train]

        if _HAS_XGB:
            # Use LabelEncoder to handle any subset of {-1, 0, +1} as contiguous 0-based indices.
            # This is necessary when neutral_pct=0 (default): labels are {-1,+1} only, so
            # naive +1 remapping gives {0,2} which XGBoost rejects as non-contiguous.
            try:
                le = _LabelEncoder()
                y_encoded = le.fit_transform(y_tr)   # {-1,0,1} or subset → {0,...,n_classes-1}
                clf = _XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    eval_metric="mlogloss",
                    verbosity=0,
                )
                clf.fit(X_tr, y_encoded)
                self._model = clf
                self._label_encoder = le
                self._label_offset = 0   # not used; inverse_transform handles decoding
                log.debug(f"MLSignal.fit: XGBoost fitted successfully on {n_train} samples, classes={le.classes_}")
            except Exception as exc:
                log.warning(f"MLSignal.fit: XGBoost fit failed ({exc}) — falling back to ZScoreThreshold")
                self._model = None
        elif _HAS_GBM:
            try:
                clf = _GBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                )
                clf.fit(X_tr, y_tr)
                self._model = clf
                self._label_offset = 0
            except Exception as exc:
                log.warning(f"MLSignal.fit: GBM fit failed ({exc}) — falling back to ZScoreThreshold")
                self._model = None
        else:
            log.warning("MLSignal.fit: neither XGBoost nor GBM available — falling back to ZScoreThreshold")
            self._model = None

        return self

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if self._model is None:
            return ZScoreThreshold(lookback=self.lookback).trade_signals(a, b)

        feat = self._build_features(a, b)
        mask = feat.notna().all(axis=1)
        preds_raw = self._model.predict(feat[mask].values)
        if self._label_encoder is not None:
            preds = self._label_encoder.inverse_transform(preds_raw.astype(int))
        else:
            preds = preds_raw.astype(int) - self._label_offset  # GBM path: labels are original

        sig = pd.Series(0, index=feat.index)
        sig[mask] = preds
        return sig.fillna(0).astype(int)
