# Experiment Run Scripts

All commands assume you are in `C:\Code\Hybrid-Pairs-Trading-Ensemble` with the venv activated.

```bash
cd "C:\Code\Hybrid-Pairs-Trading-Ensemble"
.venv/Scripts/activate
```

---

## 0. GPU Setup (one-time, run before any experiment)

Your hardware: **RTX 4060 Laptop GPU (8 GB VRAM)**, Driver 595.79, max CUDA 13.2.

### Step 1 — Uninstall the CPU-only TensorFlow

```bash
pip uninstall tensorflow -y
```

### Step 2 — Install TensorFlow with bundled CUDA 12 libraries

`tensorflow[and-cuda]` bundles `nvidia-cuda-runtime`, `nvidia-cudnn`, and related packages
as Python wheels — no separate CUDA Toolkit installation required.

```bash
pip install "tensorflow[and-cuda]==2.21.0"
```

> **If the above fails on Windows** (NVIDIA wheels unavailable for your platform),
> use the DirectML backend instead:
> ```bash
> pip install tensorflow==2.10.0
> pip install tensorflow-directml-plugin
> ```
> DirectML uses DirectX 12 and works on all NVIDIA/AMD/Intel GPUs on Windows without CUDA.

### Step 3 — Install XGBoost with GPU support

XGBoost's `tree_method="hist"` with `device="cuda"` is already wired in.
The existing `xgboost>=2.0` package supports CUDA out of the box — no extra install needed.

### Step 4 — Install joblib (parallel pair loops)

```bash
pip install joblib
```

### Step 5 — Verify GPU is visible

```bash
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF GPUs:', gpus)
print('TF version:', tf.__version__)
import xgboost as xgb
print('XGBoost version:', xgb.__version__)
from core.selectors_ml import _HAS_TF, _gpus, _XGB_DEVICE, _HAS_JOBLIB
print('TF available:', _HAS_TF, '| GPUs seen by TF:', len(_gpus))
print('XGBoost device:', _XGB_DEVICE)
print('Joblib:', _HAS_JOBLIB)
"
```

Expected output when GPU is working:
```
TF GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
[GPU] 1 GPU(s) found. Mixed precision (float16) enabled.
XGBoost device: cuda
Joblib: True
```

---

## 1. E4 — Walk-Forward Validation (HEADLINE EXPERIMENT)

### 1a. Full mode + OU-only signal (the paper's main result)

Runs all 8 selectors (Correlation, Distance, Cointegration, Combined,
MLSelector, LSTM, Transformer, GNN) with OU-only Stage-2 signal.

Expected time with GPU: **~30–60 min total** (was 13+ hours without GPU).

```bash
python experiments/walk_forward.py --mode full --s2 ou_only
```

Watch for in the output:
- `[GPU] 1 GPU(s) found. Mixed precision (float16) enabled.` — confirms GPU in use
- `[LSTMSelector] Subsampled to 50000 sequences.` — confirms cap is active
- Per-fold: `OOS => Gross Sharpe=X  Net Sharpe=Y`
- Final: `FULL OOS (all test years stitched, 6 yrs): Net Sharpe = Z`

Compare final Net Sharpe against the stat_only baseline of **+0.359**.

### 1b. Stat-only + OU-only (fast baseline, ~2 min — re-run to confirm fixes work)

```bash
python experiments/walk_forward.py --mode stat_only --s2 ou_only
```

### 1c. Stat-ML + OU-only (XGBoost MLSelector, no DL, ~5 min)

```bash
python experiments/walk_forward.py --mode stat_ml --s2 ou_only
```

---

## 2. E3 — Ablation Study

### 2a. Stat-only ablation (fast, ~1 min — already done, re-run for reference)

```bash
python experiments/ablation.py --mode stat_only
```

### 2b. Stat-ML ablation (adds XGBoost MLSelector, ~5 min)

```bash
python experiments/ablation.py --mode stat_ml
```

### 2c. Full ablation (all 8 selectors — run after full WFV completes, ~60 min with GPU)

```bash
python experiments/ablation.py --mode full
```

---

## 3. E5 — Benchmark Comparison

Loads the latest WFV result and compares against Nifty 50 / Bank / IT.
Auto-detects the most recent `walk_forward_*_ou_only.json` result.

```bash
python experiments/benchmark_comparison.py --mode full --s2 ou_only
```

To compare a specific result file:

```bash
python experiments/benchmark_comparison.py --wfv walk_forward_20260402_230753.json
```

---

## 4. E6 — Statistical Significance

Bootstrap Sharpe CI + Newey-West HAC t-test + Bonferroni multiple-comparison.

```bash
python experiments/significance_tests.py --mode full --s2 ou_only
```

For the existing stat-only headline result:

```bash
python experiments/significance_tests.py --wfv walk_forward_20260402_230753.json
```

---

## 5. E1 — Frequency Comparison (daily vs hourly)

```bash
python experiments/freq_comparison.py
```

---

## 6. E2 — Hold Period Sweep

```bash
python experiments/hold_period_sweep.py
```

---

## 7. Streamlit App

```bash
streamlit run app.py
```

---

## Performance Notes (RTX 4060 8 GB)

| Selector | Before (CPU only) | After (GPU + parallel) |
|---|---|---|
| LSTM (Fold 1, 671k→50k seqs) | ~60 min | ~3–5 min |
| Transformer (671k→50k seqs) | ~70 min | ~3–5 min |
| GNN (8 snapshots, 35 nodes) | ~3 min | ~1 min |
| Cointegration (595 pairs) | ~15 s | ~3–5 s (parallel) |
| XGBoost MLSelector | ~5 s | ~2 s (GPU hist) |
| **Full fold (all 8 selectors)** | **~133 min** | **~8–12 min** |
| **6 folds total** | **~13 hours** | **~50–70 min** |

Mixed precision (fp16) on the RTX 4060 gives ~2–3x training speedup for LSTM/Transformer.
The `max_sequences=50_000` cap limits data volume while preserving signal quality
(50k covers all unique pair × time combinations for the smaller folds).

### VRAM usage estimate

| Model | Batch | VRAM estimate |
|---|---|---|
| LSTM (BiLSTM units=32) | 256 | ~0.5 GB |
| Transformer (embed=32, heads=4, layers=2) | 256 | ~0.8 GB |
| GNN (hidden=32, N=35) | full graph | ~0.1 GB |

Total peak: ~1.5 GB — well within the 8 GB RTX 4060 budget.

---

## Troubleshooting

**TF still shows no GPU after install:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If empty, try: `pip install tensorflow[and-cuda] --force-reinstall`
Or on Windows, fall back to DirectML: `pip install tensorflow==2.10 tensorflow-directml-plugin`

**XGBoost CUDA error:**
The code has a CPU fallback — check the log for `[MLSelector] CUDA XGBoost failed; falling back to CPU.`
This is non-fatal; XGBoost will use CPU automatically.

**OOM (out of memory) on GPU:**
Reduce `max_sequences` in the selector constructors (default 50,000):
```python
LSTMSelector(max_sequences=25_000)
TransformerSelector(max_sequences=25_000)
```
Or reduce `batch_size` (default 256):
```python
LSTMSelector(batch_size=128)
```
These are constructor defaults in `core/selectors_ml.py` lines ~320 and ~460.
