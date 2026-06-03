# KnowledgeGraph — Paper 2: Universe Quality Dominates Methodology
# Status: SUBMISSION READY
# Last updated: 2026-06-04

## Paper Identity
- Title: Universe Quality Dominates Methodology: A Multi-Market Ensemble Pairs Trading Study
- Status: Submission-ready (all 29 Round 4 critiques resolved)
- Target: JFM ~55%, QF ~70%
- Headline: NSE Nifty50, stat-only ZScore, 4-fold rolling WFV: Net Sharpe +0.752 (p=0.036)

## Key Results
- Nifty50 stat-only: Net SR +0.752, folds [+1.127/+0.218/+0.627/+1.036], 126 trades
- Multi-market honest mean: +0.284 (5.5x vs Nifty100 +0.052)
- ML ensemble: exploratory negative (+0.354-+0.484 CPU; dilutes stat-only)
- Active selectors: 7 (Correlation, Distance, Cointegration, Combined + LSTM, Transformer, GNN; CNN DISABLED)
- Transaction costs: 16.28 bps India, 8.4 bps Brazil, 9.6 bps UK, 5.4 bps US

## Thesis Drafts (Implementation/thesis_drafts/)
- abstract.md: clean
- chapter_1_introduction.md: patched (7 FATAL fixes, 4 minor fixes)
- chapter_2_literature_review.md: clean
- chapter_3_integrated.md: patched (geographic narrative, Gatev 2006)
- chapter_4_updated_with_rolling_baseline.md: patched (ablation Section 4.4.12-13 added)
- chapter_5_conclusions_final.md: patched (fund launch removed, 16x->5.5x)
- section_3.6_rolling_sensitivity.md: patched (merged into ch3)

## Critiques/ — START HERE when resuming
- INDEX.md: master status, all 29 issues, 0 open
- Round_4_Open.md: current ground truth (all resolved)
- RESOLVED_SUMMARY.md: compact fix log
- Round_1/2/3.md: SKIP (resolved)

## Experimental Data (Implementation/experimental-ablation/)
- results/nse_nifty50/: 13 JSON files — headline 4-fold rolling results
- results/nse_nifty50_longrun/: 16-fold (2004-2024) exploratory
- results/india/ uk/ us/ brazil/: multi-market 4-fold results
- DATA_APPENDIX.md: full transparency table, bootstrap CIs, CVaR, CPU/GPU comparison

## Documentation/
- PROJECT_NARRATIVE.md: full E1-E6 phase history
- DATA_APPENDIX.md: 921 lines — all run tables, CIs, CVaR, ML nondeterminism data
- NSE_Trading_Costs_Research_2024.md: cost breakdown source
