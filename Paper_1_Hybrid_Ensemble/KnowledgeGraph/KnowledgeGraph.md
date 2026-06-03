# KnowledgeGraph — Paper 1: Hybrid Ensemble Pairs Trading
# Status: PLAN ONLY — experiments run but paper not written
# Last updated: 2026-06-04

## Paper Identity
- Title: Hybrid Ensemble Pairs Trading: Do ML Selectors Outperform Statistical Baselines? (working title)
- Status: Experimental data exists; paper not yet written
- Prior work: E1-E8 experiments on NSE Nifty 100 (2020-2025)
- Target venues: See Implementation/research_plan.txt

## Research Question
Do hybrid ML+statistical ensemble pair selectors generate superior risk-adjusted returns
compared to single-strategy statistical pairs trading on NSE Nifty 100?

## What Exists
- Implementation/core/: full engine (selectors_statistical.py, selectors_ml.py, ensemble.py, backtest.py, entry.py)
- Implementation/experiments/: E1-E8 scripts (ablation, walk_forward, benchmark_comparison, significance_tests)
- Implementation/experiments/results/: ~40 JSON result files (expanding WFV, 6-fold NSE Nifty100)
- Implementation/jobs/: SLURM job scripts for HPC runs
- Implementation/logs/: E1-E8 run logs
- Implementation/reports/: 10 equity/trade report folders
- Implementation/Research.md: early experiment narrative (E1-E8, OLD 22.9 bps costs — superseded)
- Implementation/Decisions.md: 11 design rationale entries

## Key Known Results (from E1-E8, expanding WFV, Nifty100)
- Best config (E4, 6-fold expanding): Net SR ~+0.05 to +0.40 range (cost-drag heavy)
- Expanding window fails badly on NSE (overfits early folds)
- LSTM+Correlation 2-selector was early headline — later superseded by stat-only
- GPU non-determinism affected ML results (resolved in Paper 2)

## PLAN STATUS
- See Paper_1_Plan.md for full structured plan
- Paper not written yet — needs literature framing, clean experiments, honest reporting

## Critiques/
- No critiques yet (paper not submitted)
