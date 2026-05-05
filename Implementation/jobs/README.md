# SLURM Job Scripts — Hybrid Pairs Trading Ensemble

Job submission scripts for the CMInDS Kalpana cluster (partition: `cn3_anandi`, account: `cminds_anandi`).

## Current Status (as of 2026-04-06)

| Experiment | Job Script | Status | Result File |
|---|---|---|---|
| E1 (stat_only) | `e1_frequency.sh` | DONE | `freq_comparison_20260402_025539.json` |
| E1 (full mode) | `e1_frequency_full.sh` | **PENDING** | — |
| E2 Hold sweep | — (run locally) | DONE | `hold_period_sweep_20260402_031710.json` |
| E3 ablation (stat_ml + full) | `e3_ablation.sh` | DONE | `ablation_20260406_*.json` |
| E3 Transformer_only re-run | `e3_transformer_rerun.sh` | **PENDING** | — |
| E4 WFV (all modes) | `e4_walk_forward.sh` | DONE | `walk_forward_20260406_011541.json` |
| E5 Benchmarks | — (run locally) | DONE | `benchmark_20260403_001455.json` |
| E6 Significance (headline) | — (run locally) | DONE | `significance_20260403_002057.json` |
| E6 Significance (full-mode) | `e6_significance_full.sh` | **PENDING** (run after E7) | — |
| **E7 Weighted Ensemble** | `e7_weighted_ensemble.sh` | **PENDING ← START HERE** | — |

## Recommended Submission Order

Cluster limit: 2 running + 2 pending jobs per user.

**Round 1 (submit both together):**
```bash
sbatch jobs/e7_weighted_ensemble.sh   # ~24h; highest priority
sbatch jobs/e3_transformer_rerun.sh   # ~3h; validates the Lambda fix
```

**Round 2 (after Round 1 completes):**
```bash
sbatch jobs/e6_significance_full.sh   # ~2h; update --wfv arg with E7 best result first
sbatch jobs/e1_frequency_full.sh      # ~4h; lower priority, paper completeness
```

---

## Job Scripts

### E7: Weighted Ensemble (NEW — highest priority)
**File:** `e7_weighted_ensemble.sh`  
**Runtime:** ~24h (4 weight configs × 6 folds × ~1h/fold)  
**Purpose:** Test whether LSTM-heavy S1 + OU-only S2 beats equal-weight ensemble

Runs 4 S1 weight configurations:
- Config A: LSTM=3, Corr=2, Dist=1, Coint=1, Transformer=1 (no ML/GNN/Combined)
- Config B: LSTM=1, Corr=1, Dist=1, Coint=1 (stat+LSTM, tightest clean set)
- Config C: LSTM=1, Corr=1 only (most selective)
- Config D: Full 8 selectors quality-weighted (LSTM=3, Corr=2, Transformer=1, GNN=0.25)

```bash
sbatch jobs/e7_weighted_ensemble.sh
```

### E3: Transformer_only Re-run
**File:** `e3_transformer_rerun.sh`  
**Runtime:** ~3h  
**Purpose:** Validate the Lambda+GPU fix for TransformerSelector. Re-runs full Stage 1 ablation.

```bash
sbatch jobs/e3_transformer_rerun.sh
```

### E6: Significance on Full-mode Results
**File:** `e6_significance_full.sh`  
**Runtime:** ~2h  
**Purpose:** Bootstrap CI and Newey-West tests on full-mode WFV + E7 best result

Before submitting, edit the script to add the E7 best result filename:
```bash
# In e6_significance_full.sh, uncomment the E7 block and set:
E7_BEST_RESULT=experiments/results/walk_forward_<YYYYMMDD_HHMMSS>.json
```

```bash
sbatch jobs/e6_significance_full.sh
```

### E1: Frequency Comparison (full mode)
**File:** `e1_frequency_full.sh`  
**Runtime:** ~4h  
**Purpose:** Paper completeness — frequency comparison with all 8 selectors

```bash
sbatch jobs/e1_frequency_full.sh
```

### Legacy Scripts (experiments already completed)

| Script | Experiment | Notes |
|---|---|---|
| `e4_walk_forward.sh` | E4 WFV (full mode) | COMPLETE — do not re-submit |
| `e3_ablation.sh` | E3 ablation (stat_ml + full) | COMPLETE — do not re-submit |
| `e1_frequency.sh` | E1 (stat_only) | COMPLETE — do not re-submit |

---

## Monitoring

```bash
squeue -u $(whoami)                              # job status
tail -f logs/e7_weighted_ensemble_<id>.out       # live output
cat  logs/e7_weighted_ensemble_<id>.err          # errors
scancel <job_id>                                 # cancel a job
```

## Results Location

All output: `experiments/results/<experiment>_<YYYYMMDD_HHMMSS>.json`

---

## Cluster Constraints

- Max 2 running + 2 pending jobs per user
- Max 24h runtime per job
- Max 2 GPUs per job (use 1 for these experiments)
- Max 8 CPUs per GPU
- Only `sbatch` supported (no `srun`/`salloc`)
- Account `cminds_anandi` required on every job
