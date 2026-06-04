# SLURM Job Scripts — Hybrid Pairs Trading Ensemble

Job submission scripts for the CMInDS Kalpana cluster (partition: `cn3_anandi`, account: `cminds_anandi`).

---

## Experiment Status (as of 2026-04-06)

All core experiments are **complete**. This README documents how to replicate them from scratch.

| Experiment | Script | Status | Key Result File(s) |
|---|---|---|---|
| E1 Frequency comparison (stat_only) | `e1_frequency.sh` | DONE | `freq_comparison_20260402_025539.json` |
| E2 Hold period sweep | *(local — `scripts.md`)* | DONE | `hold_period_sweep_20260402_031710.json` |
| E3 Ablation (stat_only + stat_ml + full) | `e3_ablation.sh` | DONE | `ablation_20260406_054108.json` |
| E3 Transformer re-run (Lambda/GPU fix) | `e3_transformer_rerun.sh` | DONE | *(in ablation output above)* |
| E4 Walk-forward validation (full + ou_only) | `e4_walk_forward.sh` | DONE | `walk_forward_20260406_011541.json` |
| E5 Benchmark comparison | *(local — `scripts.md`)* | DONE | `benchmark_20260403_001455.json` |
| E6 Statistical significance (all results) | `e6_significance_full.sh` | DONE | `significance_20260406_15502*.json` |
| E7 Weighted ensemble (4 configs) | `e7_weighted_ensemble.sh` | DONE | `walk_forward_20260406_04{4108,4657,5231,51419}.json` |

**Best result:** E7 Config C — LSTM + Correlation only, OU S2 — Full-OOS Net SR **0.451**, Gross SR 0.762, Net CAGR 2.58%, MaxDD 9.54%. File: `walk_forward_20260406_045231.json`.

---

## One-Time Cluster Setup

Run once to create the conda environment:

```bash
cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation
bash setup_env.sh
```

Verify environment:
```bash
conda activate pairs_trading
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import xgboost; print(xgboost.__version__)"
```

---

## Replicating All Experiments

Use `run_full_experiments.sh` to submit all jobs in dependency order (respects cluster 2-job limit via SLURM chains):

```bash
cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation
bash jobs/run_full_experiments.sh
```

Or submit individually in the order below.

### Round 1 — Independent (submit together)

```bash
sbatch jobs/e4_walk_forward.sh    # ~12h — full-mode WFV, headline OOS result
sbatch jobs/e3_ablation.sh        # ~10h — full ablation across all selectors + signals
```

### Round 2 — After Round 1 completes

```bash
sbatch jobs/e7_weighted_ensemble.sh   # ~4h — 4 weighted ensemble configs
sbatch jobs/e3_transformer_rerun.sh   # ~3h — Transformer-only ablation re-run
```

### Round 3 — After Round 2 completes

```bash
sbatch jobs/e1_frequency.sh           # ~2h — daily vs hourly frequency comparison
sbatch jobs/e6_significance_full.sh   # ~1h — bootstrap + Newey-West significance tests
```

> **Note:** E2 (hold period sweep) and E5 (benchmark comparison) run locally on your laptop in under 5 minutes each. See `scripts.md` for the exact commands.

---

## Individual Job Descriptions

### E4: Walk-Forward Validation
**File:** `e4_walk_forward.sh` | **Runtime:** ~12h | **Mode:** full, s2=ou_only

Runs the headline 6-fold OOS walk-forward validation. All 8 pair selectors are re-fit per fold; OU signal model only (empirically optimal from E3).

### E3: Ablation Study
**File:** `e3_ablation.sh` | **Runtime:** ~10h

Runs three ablation passes sequentially: `stat_only`, `stat_ml`, `full`. Each pass isolates individual selectors and signal models to measure their standalone contribution.

### E3: Transformer Re-run
**File:** `e3_transformer_rerun.sh` | **Runtime:** ~3h

Re-runs the full Stage 1 ablation after the `_PositionalEncodingLayer` fix (replaced Lambda layer that crashed on GPU). Validates Transformer_only result (Full-OOS Net SR +0.023).

### E7: Weighted Ensemble
**File:** `e7_weighted_ensemble.sh` | **Runtime:** ~4h

Tests 4 S1 weight configurations, all with OU-only S2:
- Config A: LSTM=3, Corr=2, Dist=1, Coint=1, Trans=1
- Config B: LSTM=1, Corr=1, Dist=1, Coint=1
- **Config C: LSTM=1, Corr=1 only ← Best result (Net SR 0.451)**
- Config D: Full weighted (LSTM=3, Corr=2, Dist=1, Coint=1, Trans=1, GNN=0.25, Comb=0.25)

### E1: Frequency Comparison
**File:** `e1_frequency.sh` | **Runtime:** ~2h | **Mode:** stat_only

Compares daily (1D) vs hourly (1H) strategy performance. Demonstrates that gross Sharpe degrades from 1.14 (daily) to 0.49 (hourly) and the hourly strategy bankrupts net-of-costs. Justifies daily data choice.

### E6: Statistical Significance
**File:** `e6_significance_full.sh` | **Runtime:** ~1h

Runs bootstrap CI (B=10,000, block=30) and Newey-West HAC t-test on:
1. Full-mode WFV result (`walk_forward_20260406_011541.json`)
2. E7 Config C best result (`walk_forward_20260406_045231.json`)

Key finding: Config C gross alpha is significant (p=0.011); net alpha is marginal (p=0.084).

---

## Monitoring

```bash
squeue -u $(whoami)                    # job queue status
tail -f logs/<jobname>_<id>.out        # live output
cat  logs/<jobname>_<id>.err           # errors
scancel <job_id>                       # cancel a job
```

## Results

All JSON output: `experiments/results/<experiment>_<YYYYMMDD_HHMMSS>.json`

---

## Cluster Constraints

- Max **2 running + 2 pending** jobs per user — submit in rounds
- Max **24h** runtime per job
- Max **2 GPUs** per job (all scripts use 1)
- Max **8 CPUs** per GPU (all scripts use 4)
- Account `cminds_anandi` required on every job
- Only `sbatch` supported (no `srun`/`salloc`)
