"""Machine Learning Pair Selection Models.

Includes:
- MLSelector: Supervised learning with engineered features
- LSTMSelector: LSTM/BiLSTM sequence model
- TransformerSelector: Transformer encoder model
- GNNSelector: Graph Convolutional Network link prediction
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .selectors_base import Pair, PairScore, PairSelector

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joblib for CPU-parallel pair loops
# ---------------------------------------------------------------------------
try:
    from joblib import Parallel, delayed as _delayed
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    _HAS_JOBLIB = False

# Optional dependencies
try:  # statsmodels for cointegration spread features
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
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
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

    # -----------------------------------------------------------------------
    # GPU setup — run once at import time
    # -----------------------------------------------------------------------
    _gpus = tf.config.list_physical_devices("GPU")
    if _gpus:
        # Allow incremental GPU memory growth (avoid allocating all VRAM upfront)
        for _gpu in _gpus:
            try:
                tf.config.experimental.set_memory_growth(_gpu, True)
            except RuntimeError:
                pass  # already initialised
        # Mixed precision: fp16 compute + fp32 master weights
        # 2-4x speedup on NVIDIA RTX/A-series; no accuracy loss for classification
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        _log.info(f"[GPU] {len(_gpus)} GPU(s) found. Mixed precision (float16) enabled.")
    else:
        _log.info("[GPU] No GPU detected — running on CPU.")
    # XLA JIT: fuses TF ops into optimised kernels (helps both CPU and GPU)
    tf.config.optimizer.set_jit(True)

except Exception:  # pragma: no cover
    _HAS_TF = False
    _gpus = []

# XGBoost device: use CUDA if a GPU is visible
_XGB_DEVICE: str = "cpu"
if _HAS_XGB:
    try:
        import subprocess as _sub
        _r = _sub.run(["nvidia-smi", "-L"], capture_output=True, timeout=3)
        if _r.returncode == 0 and b"GPU" in _r.stdout:
            _XGB_DEVICE = "cuda"
            _log.info("[GPU] XGBoost will use CUDA.")
    except Exception:
        pass


# ---------------------------------------------
# Fallback model for ML selector
# ---------------------------------------------

# ---------------------------------------------------------------------------
# Vectorized sliding-window sequence builder (shared by LSTM + Transformer)
# ---------------------------------------------------------------------------

def _make_sequences_fast(
    feats: np.ndarray,       # (T, F) float32
    r_spread: np.ndarray,    # (T,) float64
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) training pairs with zero-copy windows.

    Uses np.lib.stride_tricks.as_strided for O(1) window construction instead
    of a Python loop, then cumsum for O(T) label computation.
    """
    T, F = feats.shape
    n = T - seq_len - horizon
    if n <= 0:
        return (
            np.empty((0, seq_len, F), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    # Zero-copy sliding windows: shape (n, seq_len, F)
    item_bytes = feats.strides[0]
    feat_bytes = feats.strides[1]
    X = np.lib.stride_tricks.as_strided(
        feats,
        shape=(n, seq_len, F),
        strides=(item_bytes, item_bytes, feat_bytes),
    ).copy().astype(np.float32)   # .copy() owns its own memory

    # Vectorized labels via prefix-sum
    cs = np.empty(T + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(r_spread, out=cs[1:])
    roll = cs[seq_len + horizon : seq_len + horizon + n] - cs[seq_len : seq_len + n]
    y = (roll > 0).astype(np.int32)
    return X, y


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
    """Supervised selector trained to predict profitable pairs."""
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
        r_a = a.pct_change()
        r_b = b.pct_change()
        # Rolling horizon-bar sum of spread returns; drop NaN to avoid the
        # shift(-1) end-of-window NaN that caused all labels to be 0.
        spread_roll = (r_a - r_b).rolling(self.horizon).sum().dropna()
        if len(spread_roll) == 0:
            return 0
        spread_ret = spread_roll.iloc[-1]
        return int(1 if spread_ret > 0 else 0)

    def _year_splits(self, idx: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        years = pd.to_datetime(idx).year
        uniq = list(dict.fromkeys(years))
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

        uniq, counts = np.unique(y, return_counts=True)
        counts_map = {int(k): int(v) for k, v in zip(uniq, counts)}
        _log.info(f"[MLSelector.fit] y class counts: {counts_map}")

        if len(uniq) < 2:
            p1 = float(y.mean()) if len(y) else 0.5
            self.model = TrivialSelectorModel(p1=max(1e-6, min(1.0 - 1e-6, p1)))
            self.model.fit(X, y)
            return self

        c0 = counts_map.get(0, 0)
        c1 = counts_map.get(1, 0)
        maj_label = 0 if c0 >= c1 else 1
        min_label = 1 - maj_label
        maj_count, min_count = (c0, c1) if maj_label == 0 else (c1, c0)
        if min_count > 0 and maj_count / max(1, min_count) > self.rebalance_if_ratio_gt:
            target_maj = int(2.0 * min_count)
            maj_idx = np.where(y == maj_label)[0]
            min_idx = np.where(y == min_label)[0]
            keep_maj = np.random.RandomState(42).choice(maj_idx, size=target_maj, replace=False)
            keep_idx = np.sort(np.concatenate([keep_maj, min_idx]))
            X = X.iloc[keep_idx].reset_index(drop=True)
            y = y[keep_idx]
            _log.info(f"[MLSelector.fit] Rebalanced from {maj_count}:{min_count} -> {np.sum(y==maj_label)}:{np.sum(y==min_label)}")

        if _HAS_XGB:
            def _make_xgb(device: str) -> XGBClassifier:
                return XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                    tree_method="hist",   # histogram method: fast on both CPU and GPU
                    device=device,
                    n_jobs=-1,            # all CPU cores when device="cpu"
                )
            try:
                model = _make_xgb(_XGB_DEVICE)
                model.fit(X, y)
            except Exception:
                # CUDA unavailable or version mismatch — silently fall back to CPU
                _log.info("[MLSelector] CUDA XGBoost failed; falling back to CPU.")
                model = _make_xgb("cpu")
                model.fit(X, y)
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
    """LSTM / BiLSTM pair selector using temporal spread features."""
    name = "LSTM/BiLSTM"
    _N_FEATURES = 6

    def __init__(
        self,
        seq_len: int = 60,
        bidirectional: bool = True,
        units: int = 32,
        epochs: int = 20,
        batch_size: int = 256,
        horizon: int = 20,
        dropout: float = 0.2,
        max_sequences: int = 50_000,
    ):
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.dropout = dropout
        self.max_sequences = max_sequences
        self.model: Optional[object] = None

    def _pair_feature_series(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
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

    def _make_sequences(self, a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        feats = self._pair_feature_series(a, b).values.astype(np.float32)
        r_spread = (a.pct_change() - b.pct_change()).fillna(0.0).values
        return _make_sequences_fast(feats, r_spread, self.seq_len, self.horizon)

    def _build_model(self) -> object:
        # Output layer uses explicit float32 for numerical stability under mixed precision
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
        model.add(Dense(1, activation="sigmoid", dtype="float32"))  # fp32 output for stable loss
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def _seq_for_pair(
        self, cols: List[str], train: pd.DataFrame, min_len: int, i: int, j: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute sequences for one pair — called in parallel."""
        a = train[cols[i]].dropna()
        b = train[cols[j]].dropna()
        common = a.index.intersection(b.index)
        if len(common) < min_len:
            return None, None
        X, y = self._make_sequences(a.reindex(common).ffill(), b.reindex(common).ffill())
        return (X, y) if len(X) else (None, None)

    def fit(self, prices: pd.DataFrame) -> "LSTMSelector":
        if not _HAS_TF:
            _log.warning("[LSTMSelector] TensorFlow unavailable; will return neutral scores.")
            self.model = None
            return self

        min_len = self.seq_len + self.horizon + 60
        if len(prices) < min_len:
            _log.info(f"[LSTMSelector] Insufficient rows ({len(prices)} < {min_len}); skipping fit.")
            self.model = None
            return self

        split = max(min_len, len(prices) - 252)
        train = prices.iloc[:split]
        cols = list(train.columns)
        pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]

        # Build sequences in parallel across pairs (CPU-bound, releases GIL via numpy)
        n_jobs = -1 if _HAS_JOBLIB else 1
        if _HAS_JOBLIB:
            results = Parallel(n_jobs=n_jobs, backend="threading")(
                _delayed(self._seq_for_pair)(cols, train, min_len, i, j)
                for i, j in pairs
            )
        else:
            results = [self._seq_for_pair(cols, train, min_len, i, j) for i, j in pairs]

        all_X = [X for X, _ in results if X is not None]
        all_y = [y for _, y in results if y is not None]

        if not all_X:
            _log.info("[LSTMSelector] No training sequences produced; skipping fit.")
            self.model = None
            return self

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        unique, counts = np.unique(y, return_counts=True)
        _log.info(f"[LSTMSelector] {len(X)} sequences, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

        if len(unique) < 2:
            _log.info("[LSTMSelector] Single-class labels; skipping fit.")
            self.model = None
            return self

        idx = np.random.RandomState(42).permutation(len(X))
        X, y = X[idx], y[idx]

        if self.max_sequences and len(X) > self.max_sequences:
            X = X[: self.max_sequences]
            y = y[: self.max_sequences]
            _log.info(f"[LSTMSelector] Subsampled to {self.max_sequences} sequences.")

        # Split validation set manually (validation_split incompatible with tf.data)
        n_val = max(1, int(0.15 * len(X)))
        X_tr, X_val = X[n_val:], X[:n_val]
        y_tr, y_val = y[n_val:], y[:n_val]

        # tf.data pipeline: keeps GPU fed between batches
        _AUTO = tf.data.AUTOTUNE
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_tr, y_tr.astype(np.float32)))
            .shuffle(min(len(X_tr), 10_000), seed=42)
            .batch(self.batch_size)
            .prefetch(_AUTO)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val.astype(np.float32)))
            .batch(self.batch_size)
            .prefetch(_AUTO)
        )

        tf.random.set_seed(42)
        self.model = self._build_model()
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        self.model.fit(train_ds, validation_data=val_ds,
                       epochs=self.epochs, callbacks=[es], verbose=0)
        _log.info("[LSTMSelector] Training complete.")
        return self

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        if not _HAS_TF or self.model is None:
            return [PairScore(p, 0.0, {"lstm": False}) for p in candidates]

        # Build batch for all valid candidates in one pass (single GPU inference call)
        windows: List[np.ndarray] = []
        valid: List[bool] = []
        for p in candidates:
            if p.a not in prices.columns or p.b not in prices.columns:
                valid.append(False)
                continue
            a = prices[p.a].dropna()
            b = prices[p.b].dropna()
            common = a.index.intersection(b.index)
            if len(common) < self.seq_len + 60:
                valid.append(False)
                continue
            feats = self._pair_feature_series(
                a.reindex(common).ffill(), b.reindex(common).ffill()
            ).values[-self.seq_len:].astype(np.float32)
            if feats.shape[0] < self.seq_len:
                valid.append(False)
                continue
            windows.append(feats)
            valid.append(True)

        if not any(valid):
            return [PairScore(p, 0.0, {}) for p in candidates]

        X_batch = np.stack(windows, axis=0)           # (N_valid, seq_len, F)
        probas = self.model.predict(X_batch, batch_size=512, verbose=0).flatten()

        out: List[PairScore] = []
        prob_iter = iter(probas.tolist())
        for p, is_valid in zip(candidates, valid):
            if is_valid:
                proba = next(prob_iter)
                out.append(PairScore(p, proba, {"lstm_proba": proba}))
            else:
                out.append(PairScore(p, 0.0, {}))
        return out


# ---------------------------------------------
# Transformer selector
# ---------------------------------------------

if _HAS_TF:
    class _PositionalEncodingLayer(tf.keras.layers.Layer):
        """Non-trainable layer that adds sinusoidal positional encoding.

        Avoids the Lambda + captured-tf.constant pattern, which causes
        device-placement failures on GPU clusters (TF issue with Lambda.call).
        """
        def __init__(self, seq_len: int, embed_dim: int, **kwargs):
            super().__init__(trainable=False, **kwargs)
            angles = np.arange(seq_len)[:, None] / np.power(
                10000.0, (2 * (np.arange(embed_dim)[None, :] // 2)) / embed_dim
            )
            angles[:, 0::2] = np.sin(angles[:, 0::2])
            angles[:, 1::2] = np.cos(angles[:, 1::2])
            # Store as float32 numpy; converted to tensor on first call
            self._pe_np = angles[None, :, :].astype(np.float32)  # (1, seq_len, embed_dim)

        def call(self, x):
            pe = tf.constant(self._pe_np, dtype=x.dtype)
            return x + pe


class TransformerSelector(PairSelector):
    """Transformer encoder pair selector using temporal spread features."""
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
        batch_size: int = 256,
        horizon: int = 20,
        max_sequences: int = 50_000,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.max_sequences = max_sequences
        self.model: Optional[object] = None

    def _pair_feature_series(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
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

    def _make_sequences(self, a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        feats = self._pair_feature_series(a, b).values.astype(np.float32)
        r_spread = (a.pct_change() - b.pct_change()).fillna(0.0).values
        return _make_sequences_fast(feats, r_spread, self.seq_len, self.horizon)

    @staticmethod
    def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        return angles[np.newaxis, :, :].astype(np.float32)

    def _build_model(self) -> object:
        inputs = Input(shape=(self.seq_len, self._N_FEATURES))
        x = Dense(self.embed_dim)(inputs)
        x = _PositionalEncodingLayer(self.seq_len, self.embed_dim)(x)

        for _ in range(self.num_layers):
            attn = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout,
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

            ff = Dense(self.ff_dim, activation="relu")(x)
            ff = Dense(self.embed_dim)(ff)
            x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))

        x = GlobalAveragePooling1D()(x)
        x = Dense(16, activation="relu")(x)
        # dtype="float32" for numerical stability under mixed precision
        outputs = Dense(1, activation="sigmoid", dtype="float32")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def _seq_for_pair(
        self, cols: List[str], train: pd.DataFrame, min_len: int, i: int, j: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a = train[cols[i]].dropna()
        b = train[cols[j]].dropna()
        common = a.index.intersection(b.index)
        if len(common) < min_len:
            return None, None
        X, y = self._make_sequences(a.reindex(common).ffill(), b.reindex(common).ffill())
        return (X, y) if len(X) else (None, None)

    def fit(self, prices: pd.DataFrame) -> "TransformerSelector":
        if not _HAS_TF:
            _log.warning("[TransformerSelector] TensorFlow unavailable; will return neutral scores.")
            self.model = None
            return self

        min_len = self.seq_len + self.horizon + 60
        if len(prices) < min_len:
            _log.info(f"[TransformerSelector] Insufficient rows ({len(prices)} < {min_len}); skipping fit.")
            self.model = None
            return self

        split = max(min_len, len(prices) - 252)
        train = prices.iloc[:split]
        cols = list(train.columns)
        pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]

        n_jobs = -1 if _HAS_JOBLIB else 1
        if _HAS_JOBLIB:
            results = Parallel(n_jobs=n_jobs, backend="threading")(
                _delayed(self._seq_for_pair)(cols, train, min_len, i, j)
                for i, j in pairs
            )
        else:
            results = [self._seq_for_pair(cols, train, min_len, i, j) for i, j in pairs]

        all_X = [X for X, _ in results if X is not None]
        all_y = [y for _, y in results if y is not None]

        if not all_X:
            _log.info("[TransformerSelector] No training sequences produced; skipping fit.")
            self.model = None
            return self

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        unique, counts = np.unique(y, return_counts=True)
        _log.info(f"[TransformerSelector] {len(X)} sequences, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

        if len(unique) < 2:
            _log.info("[TransformerSelector] Single-class labels; skipping fit.")
            self.model = None
            return self

        idx = np.random.RandomState(42).permutation(len(X))
        X, y = X[idx], y[idx]

        if self.max_sequences and len(X) > self.max_sequences:
            X = X[: self.max_sequences]
            y = y[: self.max_sequences]
            _log.info(f"[TransformerSelector] Subsampled to {self.max_sequences} sequences.")

        n_val = max(1, int(0.15 * len(X)))
        X_tr, X_val = X[n_val:], X[:n_val]
        y_tr, y_val = y[n_val:], y[:n_val]

        _AUTO = tf.data.AUTOTUNE
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_tr, y_tr.astype(np.float32)))
            .shuffle(min(len(X_tr), 10_000), seed=42)
            .batch(self.batch_size)
            .prefetch(_AUTO)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val.astype(np.float32)))
            .batch(self.batch_size)
            .prefetch(_AUTO)
        )

        tf.random.set_seed(42)
        self.model = self._build_model()
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        self.model.fit(train_ds, validation_data=val_ds,
                       epochs=self.epochs, callbacks=[es], verbose=0)
        _log.info("[TransformerSelector] Training complete.")
        return self

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        if not _HAS_TF or self.model is None:
            return [PairScore(p, 0.0, {"transformer": False}) for p in candidates]

        windows: List[np.ndarray] = []
        valid: List[bool] = []
        for p in candidates:
            if p.a not in prices.columns or p.b not in prices.columns:
                valid.append(False)
                continue
            a = prices[p.a].dropna()
            b = prices[p.b].dropna()
            common = a.index.intersection(b.index)
            if len(common) < self.seq_len + 60:
                valid.append(False)
                continue
            feats = self._pair_feature_series(
                a.reindex(common).ffill(), b.reindex(common).ffill()
            ).values[-self.seq_len:].astype(np.float32)
            if feats.shape[0] < self.seq_len:
                valid.append(False)
                continue
            windows.append(feats)
            valid.append(True)

        if not any(valid):
            return [PairScore(p, 0.0, {}) for p in candidates]

        X_batch = np.stack(windows, axis=0)
        probas = self.model.predict(X_batch, batch_size=512, verbose=0).flatten()

        out: List[PairScore] = []
        prob_iter = iter(probas.tolist())
        for p, is_valid in zip(candidates, valid):
            if is_valid:
                proba = next(prob_iter)
                out.append(PairScore(p, proba, {"transformer_proba": proba}))
            else:
                out.append(PairScore(p, 0.0, {}))
        return out


# ---------------------------------------------
# Graph Neural Network selector
# ---------------------------------------------

class GNNSelector(PairSelector):
    """Graph Convolutional Network (GCN) pair selector."""
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
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.horizon = horizon
        self.n_snapshots = n_snapshots
        self.W1: Optional[object] = None
        self.W2: Optional[object] = None
        self.W_link: Optional[object] = None
        self.b_link: Optional[object] = None
        self._trained: bool = False

    def _node_features(self, prices: pd.DataFrame, cols: List[str]) -> np.ndarray:
        feats = []
        for c in cols:
            p = prices[c].dropna()
            r = p.pct_change().dropna()
            if len(r) < 5:
                feats.append(np.zeros(self._N_NODE_FEATURES, dtype=np.float32))
                continue
            mean_ret = float(r.mean())
            vol = float(r.std(ddof=0) * np.sqrt(252))
            skew = float(r.skew()) if len(r) >= 10 else 0.0
            kurt = float(r.kurt()) if len(r) >= 10 else 0.0
            momentum = float((1 + r).prod() - 1)
            price_z = float((p.iloc[-1] - p.mean()) / (p.std(ddof=0) + 1e-9))
            feats.append([mean_ret, vol, skew, kurt, momentum, price_z])
        return np.clip(np.array(feats, dtype=np.float32), -10.0, 10.0)

    def _adjacency(self, prices: pd.DataFrame, cols: List[str]) -> np.ndarray:
        rets = prices[cols].pct_change().dropna()
        A = np.clip(rets.corr().values, 0.0, 1.0).astype(np.float32)
        A = A + np.eye(len(cols), dtype=np.float32)
        d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1) + 1e-9)
        D = np.diag(d_inv_sqrt)
        return (D @ A @ D).astype(np.float32)

    def _pair_labels(self, prices: pd.DataFrame, cols: List[str], fwd_start: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        fwd = prices.iloc[fwd_start: fwd_start + self.horizon]
        pair_idx: List[Tuple[int, int]] = []
        labels: List[int] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i] not in fwd.columns or cols[j] not in fwd.columns:
                    continue
                ri = fwd[cols[i]].pct_change().dropna().values
                rj = fwd[cols[j]].pct_change().dropna().values
                n = min(len(ri), len(rj))
                if n == 0:
                    continue
                pair_idx.append((i, j))
                labels.append(1 if (ri[:n] - rj[:n]).sum() > 0 else 0)
        return pair_idx, np.array(labels, dtype=np.float32)

    def _gcn_forward(self, A_hat: np.ndarray, X: np.ndarray) -> "tf.Tensor":
        A = tf.constant(A_hat, dtype=tf.float32)
        H = tf.constant(X, dtype=tf.float32)
        H = tf.nn.relu(A @ H @ self.W1)
        H = tf.nn.relu(A @ H @ self.W2)
        return H

    def _link_logits(self, H: "tf.Tensor", pair_idx: List[Tuple[int, int]]) -> "tf.Tensor":
        feats = []
        for i, j in pair_idx:
            hi, hj = H[i], H[j]
            feats.append(tf.concat([hi, hj, hi * hj], axis=0))
        F = tf.stack(feats)
        return tf.sigmoid(F @ self.W_link + self.b_link)

    def fit(self, prices: pd.DataFrame) -> "GNNSelector":
        if not _HAS_TF:
            _log.warning("[GNNSelector] TensorFlow unavailable; will return neutral scores.")
            self._trained = False
            return self

        min_len = self.lookback + self.horizon + 5
        if len(prices) < min_len:
            _log.info(f"[GNNSelector] Insufficient rows ({len(prices)} < {min_len}); skipping fit.")
            self._trained = False
            return self

        train_end = max(min_len, len(prices) - 252)
        train = prices.iloc[:train_end]
        cols = list(train.columns)
        N = len(cols)

        if N < 2:
            _log.info("[GNNSelector] Universe too small; skipping fit.")
            self._trained = False
            return self

        init = tf.initializers.glorot_uniform(seed=42)
        self.W1 = tf.Variable(init((self._N_NODE_FEATURES, self.hidden_dim)), dtype=tf.float32)
        self.W2 = tf.Variable(init((self.hidden_dim, self.embed_dim)), dtype=tf.float32)
        self.W_link = tf.Variable(init((3 * self.embed_dim, 1)), dtype=tf.float32)
        self.b_link = tf.Variable(tf.zeros((1,), dtype=tf.float32))
        trainable = [self.W1, self.W2, self.W_link, self.b_link]

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        usable = len(train) - self.horizon
        step = max(1, (usable - self.lookback) // max(1, self.n_snapshots - 1))
        starts = list(range(0, usable - self.lookback, step))[:self.n_snapshots]

        snapshots = []
        for s in starts:
            window = train.iloc[s: s + self.lookback]
            A_hat = self._adjacency(window, cols)
            X = self._node_features(window, cols)
            pair_idx, y = self._pair_labels(train, cols, s + self.lookback)
            if len(pair_idx) == 0:
                continue
            snapshots.append((A_hat, X, pair_idx, y))

        if not snapshots:
            _log.info("[GNNSelector] No valid snapshots; skipping fit.")
            self._trained = False
            return self

        all_y = np.concatenate([s[3] for s in snapshots])
        if len(np.unique(all_y)) < 2:
            _log.info("[GNNSelector] Single-class labels across all snapshots; skipping fit.")
            self._trained = False
            return self

        _log.info(f"[GNNSelector] Training on {len(snapshots)} snapshots, N={N} stocks.")
        tf.random.set_seed(42)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for A_hat, X, pair_idx, y in snapshots:
                y_t = tf.constant(y[:, np.newaxis], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    H = self._gcn_forward(A_hat, X)
                    scores = self._link_logits(H, pair_idx)
                    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_t, scores))
                grads = tape.gradient(loss, trainable)
                optimizer.apply_gradients(zip(grads, trainable))
                epoch_loss += float(loss)

            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / len(snapshots)
                _log.debug(f"[GNNSelector] Epoch {epoch+1}/{self.epochs} — avg loss: {avg:.4f}")

        self._trained = True
        _log.info("[GNNSelector] Training complete.")
        return self

    def score_pairs(self, prices: pd.DataFrame, candidates: List[Pair]) -> List[PairScore]:
        out: List[PairScore] = []

        if not _HAS_TF or not self._trained:
            for p in candidates:
                out.append(PairScore(p, 0.0, {"gnn": False}))
            return out

        window = prices.tail(self.lookback)
        cols = list(window.columns)
        col_idx = {c: i for i, c in enumerate(cols)}

        A_hat = self._adjacency(window, cols)
        X = self._node_features(window, cols)
        H = self._gcn_forward(A_hat, X).numpy()

        for p in candidates:
            if p.a not in col_idx or p.b not in col_idx:
                out.append(PairScore(p, 0.0, {}))
                continue

            i, j = col_idx[p.a], col_idx[p.b]
            hi, hj = H[i], H[j]
            f = np.concatenate([hi, hj, hi * hj]).astype(np.float32)
            f_t = tf.constant(f[np.newaxis, :], dtype=tf.float32)
            proba = float(tf.sigmoid(f_t @ self.W_link + self.b_link).numpy()[0, 0])
            out.append(PairScore(p, proba, {
                "gnn_proba": proba,
                "hidden_dim": self.hidden_dim,
                "embed_dim": self.embed_dim,
                "n_snapshots": self.n_snapshots,
            }))

        return out
