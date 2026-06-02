#!/usr/bin/env python3
"""
run_training_diagnostics.py

Extracts training/validation loss curves + pair selection overlap from
LSTM, Transformer, GNN selectors on NSE Nifty 50 Fold 1 (train: 2020, test: 2021).

Outputs: results/nse_nifty50/training_diagnostics.json
"""

import sys, json, warnings, os, random
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import pandas as pd
import yaml

random.seed(42)
np.random.seed(42)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.selectors import (
    LSTMSelector, TransformerSelector, GNNSelector, Pair
)
from itertools import combinations

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "configs" / "nse_nifty50.yaml"
RESULTS_DIR = BASE / "results" / "nse_nifty50"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

cache_dir = Path(__file__).parent.parent.parent / config['data']['cache_dir']
start = config['data']['start_date']
end   = config['data']['end_date']
cache_file = cache_dir / f"prices_{start}_{end}.parquet"
prices = pd.read_parquet(cache_file)
print(f"Loaded prices: {prices.shape}")

train_prices = prices.loc["2020-01-01":"2020-12-31"]
test_prices  = prices.loc["2021-01-01":"2021-12-31"]
print(f"Train: {train_prices.shape}, Test: {test_prices.shape}")

tickers = list(prices.columns)
candidate_pairs = [Pair(a, b) for a, b in combinations(tickers, 2)]
print(f"Candidate pairs: {len(candidate_pairs)}")

diagnostics = {
    "experiment": "nse_nifty50_training_diagnostics",
    "timestamp": datetime.now().isoformat(),
    "fold": {"train": "2020-01-01 to 2020-12-31", "test": "2021-01-01 to 2021-12-31"},
    "n_candidate_pairs": len(candidate_pairs),
    "n_tickers": len(tickers),
    "train_trading_days": len(train_prices),
    "selectors": {}
}


def count_params(model):
    try:
        return int(model.count_params())
    except Exception:
        return None


for name, cls in [('lstm', LSTMSelector), ('transformer', TransformerSelector), ('gnn', GNNSelector)]:
    print(f"\n{'='*50}\nTraining {name} (run 1)...")
    diag = {"name": name, "status": "ok"}
    try:
        s1 = cls()
        s1.fit(train_prices)

        if hasattr(s1, 'model') and s1.model is not None:
            diag["n_params"] = count_params(s1.model)
        if hasattr(s1, 'lookback'):
            diag["lookback"] = int(s1.lookback)
            diag["n_usable_sequences"] = max(0, len(train_prices) - int(s1.lookback))
            diag["ratio_params_to_sequences"] = (
                round(diag["n_params"] / max(diag["n_usable_sequences"], 1), 1)
                if diag.get("n_params") else None
            )

        scores1 = s1.score_pairs(train_prices, candidate_pairs)
        valid1 = sorted([ps for ps in scores1 if ps.score > 0], key=lambda x: -x.score)
        top10_1 = set(f"{ps.pair.asset1}_{ps.pair.asset2}" for ps in valid1[:10])
        diag["n_scored_pairs_run1"] = len(valid1)

        print(f"  Run 1: params={diag.get('n_params')}, seqs={diag.get('n_usable_sequences')}, scored={len(valid1)}")
        print(f"  Training {name} (run 2)...")

        s2 = cls()
        s2.fit(train_prices)
        scores2 = s2.score_pairs(train_prices, candidate_pairs)
        valid2 = sorted([ps for ps in scores2 if ps.score > 0], key=lambda x: -x.score)
        top10_2 = set(f"{ps.pair.asset1}_{ps.pair.asset2}" for ps in valid2[:10])
        diag["n_scored_pairs_run2"] = len(valid2)

        union = top10_1 | top10_2
        intersect = top10_1 & top10_2
        diag["pair_overlap_top10"] = round(len(intersect) / len(union), 3) if union else 0.0
        diag["top_pairs_run1"] = sorted(list(top10_1))[:5]
        diag["top_pairs_run2"] = sorted(list(top10_2))[:5]
        diag["pairs_in_common"] = sorted(list(intersect))

        print(f"  Run 2: scored={len(valid2)}, overlap(top-10)={diag['pair_overlap_top10']:.1%}")

    except Exception as e:
        diag["status"] = f"error: {str(e)[:300]}"
        print(f"  ERROR: {e}")

    diagnostics["selectors"][name] = diag


# ── Keras loss curve for LSTM (representative fold) ────────────────────────
print("\n\n=== Capturing LSTM loss curves (Keras history) ===")
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    from tensorflow.keras.callbacks import EarlyStopping

    LOOKBACK = 60
    returns = train_prices.pct_change().dropna()
    cols = list(returns.columns)
    pairs_sample = list(combinations(range(len(cols)), 2))[:300]

    X_list, y_list = [], []
    for i, j in pairs_sample:
        r1 = returns.iloc[:, i].values
        r2 = returns.iloc[:, j].values
        spread = r1 - r2
        std = spread.std() + 1e-8
        for t in range(LOOKBACK, len(spread)):
            X_list.append(spread[t-LOOKBACK:t])
            future = spread[t:t+5].mean() if t + 5 < len(spread) else 0
            y_list.append(1 if abs(future) > std else 0)

    X = np.array(X_list)[..., np.newaxis]
    y = np.array(y_list)
    print(f"Sequences: {X.shape}, label balance: {y.mean():.3f}")

    split = int(0.8 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(LOOKBACK, 1)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    n_params = int(model.count_params())
    print(f"LSTM params: {n_params}, train seqs: {len(X_tr)}, ratio: {n_params/len(X_tr):.1f}")

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=20, batch_size=64, callbacks=[es], verbose=1
    )

    diagnostics["lstm_loss_curve"] = {
        "n_params": n_params,
        "n_train_sequences": len(X_tr),
        "n_val_sequences": len(X_val),
        "ratio_params_to_sequences": round(n_params / len(X_tr), 1),
        "lookback": LOOKBACK,
        "train_loss": [round(float(v), 4) for v in history.history['loss']],
        "val_loss":   [round(float(v), 4) for v in history.history['val_loss']],
        "epochs_run": len(history.history['loss']),
        "overfit_gap_final_epoch": round(
            float(history.history['val_loss'][-1]) - float(history.history['loss'][-1]), 4
        ),
        "min_val_loss": round(float(min(history.history['val_loss'])), 4),
        "min_train_loss": round(float(min(history.history['loss'])), 4),
    }
    print(f"train_loss: {diagnostics['lstm_loss_curve']['train_loss']}")
    print(f"val_loss:   {diagnostics['lstm_loss_curve']['val_loss']}")
    print(f"overfit_gap: {diagnostics['lstm_loss_curve']['overfit_gap_final_epoch']}")

except Exception as e:
    diagnostics["lstm_loss_curve"] = {"status": f"error: {str(e)[:300]}"}
    print(f"ERROR in loss curve: {e}")


# ── Save ────────────────────────────────────────────────────────────────────
out = RESULTS_DIR / "training_diagnostics.json"
with open(out, 'w') as f:
    json.dump(diagnostics, f, indent=2)
print(f"\n\nResults saved: {out}")
print(json.dumps(diagnostics, indent=2)[:3000])
