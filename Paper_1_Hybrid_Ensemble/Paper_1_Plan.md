# Paper 1 — Work Plan
# Hybrid Ensemble Pairs Trading: Does a ML Selector + Statistical Model Ensemble Outperform Single-Strategy Baselines on NSE Nifty 100?
# Created: 2026-06-29

## Core Argument (Paper 1)

Paper 1 is the foundational study: does the HYBRID ensemble (ML + stat) outperform SINGLE-STRATEGY
stat-only baselines (pure distance, pure cointegration, pure correlation) on NSE Nifty 100?
This is a different question — comparing ensemble vs single-strategy, NOT ML vs stat.

## Phase 1 — Data Audit (1 session)
1. Load all E1-E8 JSON results and extract per-fold net Sharpe for each selector config
2. Identify which configs are deterministic (stat-only) vs non-deterministic (ML)
3. Map cost model used — old 22.9 bps results need recalculation to 16.28 bps
4. Identify any gaps: missing configs, missing benchmarks (buy-and-hold, equal-weight)

## Phase 2 — Experiment Cleanup (1-2 sessions)
1. Rerun key ablation with corrected 16.28 bps costs and CPU-only ML (CUDA_VISIBLE_DEVICES='')
2. Configs to run:
   a. Single-strategy baselines: Correlation-only, Distance-only, Cointegration-only, Combined-only
   b. 2-selector combos: Correlation+Distance, Correlation+Cointegration
   c. Full stat ensemble (4 selectors)
   d. Full hybrid ensemble (4 stat + LSTM + Transformer + GNN)
   e. Benchmarks: Nifty100 buy-and-hold, equal-weight pairs
3. Use 6-fold expanding WFV on NSE Nifty100, 2020-2025, 16.28 bps
4. Three CPU runs per ML config (report mean ± std)

## Phase 3 — Statistical Testing (1 session)
1. Bootstrap 95% CI for all net Sharpe figures (1000 resamples)
2. Bonferroni correction across N comparisons
3. Pairwise t-test: ensemble net SR vs best single-strategy net SR
4. Diebold-Mariano test: ensemble PnL vs single-strategy PnL

## Phase 4 — Paper Writing (3-4 sessions)
Structure:
- Abstract: honest framing — does ensemble add value over best single selector?
- Ch1 Introduction: RQ1 (does ensemble > single-strategy?), RQ2 (which selector adds most?), RQ3 (is ML additive or detrimental?)
- Ch2 Literature Review: pairs trading history, distance/cointegration/correlation methods, ML in pairs trading, ensemble methods
- Ch3 Methodology: NSE Nifty100 dataset, 8 selectors (incl CNN), expanding WFV design, signal generation, backtest setup
- Ch4 Results: single-strategy baselines table, ensemble ablation table, benchmark comparison, ML vs stat breakdown
- Ch5 Conclusions: ensemble value proposition, ML negative finding (forward ref Paper 2), limitations

## Phase 5 — Review and Submission (1 session)
1. Internal critique pass (same checklist as Paper 2 rounds)
2. Check all numbers consistent, no cherry-picking, honest mean reporting
3. Identify target venue (JFM, QF, or conference — see research_plan.txt)

## Known Risks
- Expanding WFV on NSE fails badly (established in Paper 2 salvage) — must disclose this honestly
- ML GPU non-determinism — MUST use CPU-only for all ML runs
- Old E1-E8 results used 22.9 bps — CANNOT use old results directly, must rerun with 16.28 bps
- CNN selector was disabled (performance collapse) — must decide: include as negative result or exclude
- Sample period 2020-2025 = 5 years, COVID + rate shock included — limits generalisability

## Key Files to Reference
- Implementation/core/: engine (do not modify — shared with Paper 2)
- Implementation/experiments/: E1-E8 scripts (reuse with cost fix)
- Implementation/experiments/results/: old results (reference only — wrong costs)
- Implementation/Research.md: E1-E8 narrative (old costs, read for context only)
- Implementation/Decisions.md: 11 design rationale entries (useful for methodology section)
- Paper 2 Implementation/experimental-ablation/configs/: use as config template

## Success Criteria
Paper is submission-ready when:
- All results use 16.28 bps and CPU-only ML (deterministic for reproducibility)
- All net Sharpe figures have bootstrap 95% CIs
- Bonferroni correction applied 
- No uncaveated cherry-picked single runs
- CNNSelector disabled status disclosed
- Clear answer to: does ensemble > best single-strategy? (honest regardless of direction)
