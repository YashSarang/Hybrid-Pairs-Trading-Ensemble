# Thesis Completion Plan
**Project:** Hybrid Pairs Trading Ensemble — M.S. by Research Thesis  
**Last Updated:** 2026-05-26  
**Status:** Research Complete → Final Write-up Phase

---

## Current Status

### Completed Work
- **All experiments (E1-E8):** COMPLETE
  - E1: Daily vs Hourly frequency → Daily wins (SR 1.14 vs 0.49)
  - E2: Hold period sweep → 30 days optimal
  - E3: Ablation study → OU-only best signal (SR +0.359)
  - E4: Walk-forward validation (6 folds, 2020-2025)
  - E5: Benchmark comparison vs Nifty indices
  - E6: Statistical significance tests
  - E7: Weighted ensemble → **Config C (LSTM+Corr): Net SR +0.510, Return +17.66%**
  - E8: RL signal model (PPO) → underperforms baseline
- **Codebase:** Verified, all bugs fixed, results regenerated
- **Thesis chapters drafted:**
  - Chapter 1: Introduction (130 lines, 15KB)
  - Chapter 2: Literature Review (228 lines, 32KB)
  - Chapter 3: Methodology (535 lines, 35KB)
  - Chapter 4: Results (380 lines, 32KB)
  - Chapter 5: Discussion (224 lines, 29KB)

### Missing Components (Critical Path)
- 🔴 **Abstract** (not started)
- 🔴 **Chapter 6: Conclusion** (not started)
- 🔴 **Appendices** (not started)
- 🟡 **Final integration & polish** of existing chapters
- 🟡 **Figures & tables** finalization
- 🟡 **References** formatting
- 🟡 **Executive summary / README** for submission

---

## 🎯 Completion Roadmap (Priority Order)

### Phase 1: Core Missing Sections (HIGHEST PRIORITY)
**Estimated time:** 4-6 hours

#### Task 1.1: Write Abstract (~200-300 words)
**Purpose:** One-page summary of entire thesis for reviewers/examiners

**Required content:**
- Problem statement (pairs trading on NSE)
- Novel contribution (2-stage ensemble with 8+4 models)
- Methodology (WFV, 35 stocks, 2020-2025)
- Key results (E7 Config C: Net SR +0.510, Return +17.66%)
- Conclusion (LSTM+Corr ensemble outperforms statistical baseline)

**Output:** `reports/abstract.md`

---

#### Task 1.2: Write Chapter 6: Conclusion (~800-1000 lines)
**Purpose:** Synthesis, implications, future work

**Required sections:**
1. **Summary of Contributions**
   - First 8-model selector ensemble on NSE (incl. GNN, Transformer, LSTM)
   - Empirical frequency analysis (daily vs hourly)
   - Indian cost model (realistic NSE transaction costs)
   - WFV results showing net profitability (SR +0.510)
   - Proof that weighted ensemble (LSTM+Corr) > equal-weight

2. **Research Questions Answered**
   - RQ1: Can ensemble improve over single selector? → **Yes, but only with pruning**
   - RQ2: What frequency is optimal for NSE? → **Daily (1D)**
   - RQ3: What is the optimal hold period? → **30 days**
   - RQ4: Which signal model works best? → **OU-only**
   - RQ5: Do DL models add value? → **Yes, LSTM does; Transformer/GNN marginal**

3. **Practical Implications**
   - Retail/prop traders can achieve SR 0.5+ on NSE with this strategy
   - Cost drag (60 bps RT) is critical — min_hold=30 is essential
   - DL models (LSTM) add value but require careful weight selection
   - Hourly data is NOT viable after costs (goes bankrupt)

4. **Limitations**
   - yfinance data quality (missing bars, survivorship bias)
   - No live execution / market impact modeling
   - Limited to 35 stocks (Nifty 100 subset)
   - No regime switching / crisis period handling
   - MLSelector label mis-specification (momentum ≠ mean-reversion)

5. **Future Work**
   - Expand to full Nifty 500 universe
   - Implement regime detection (HMM/Markov switching)
   - Live broker integration (Zerodha/Upstox API)
   - Intraday strategies (if gross SR improves with better selectors)
   - Alternative cost models (F&O overlay for hedging)
   - RL agent with better reward shaping (avoid data starvation)

**Output:** `reports/chapter6_conclusion.md`

---

#### Task 1.3: Create Appendices
**Purpose:** Technical details that don't fit in main chapters

**Required appendices:**

**Appendix A: Full Cost Model Derivation**
- NSE cost structure breakdown
- IndianCosts dataclass implementation
- Comparison with US/EU cost models

**Appendix B: Selector & Signal Model Hyperparameters**
- Full table of all hyperparameters for 8 selectors + 4 signals
- Training config (epochs, batch size, learning rate)
- Feature engineering details for MLSelector

**Appendix C: Walk-Forward Fold Definitions**
- Exact date ranges for all 6 folds
- Train/test split table
- Data availability per fold

**Appendix D: Experiment Result Tables (Full)**
- E1-E8 complete result tables (from Research.md JSONs)
- All mode comparisons (stat_only, stat_ml, full)
- Fold-by-fold breakdowns

**Appendix E: Code Repository Structure**
- Directory tree of Implementation/
- Module dependency graph
- How to reproduce experiments (scripts.md content)

**Output:** `reports/appendix_a_costs.md`, `appendix_b_hyperparameters.md`, etc.

---

### Phase 2: Integration & Polish (HIGH PRIORITY)
**Estimated time:** 3-4 hours

#### Task 2.1: Update Chapter 4 with Final E7 Results
**Current state:** Chapter 4 has E4 WFV results but may not have final E7 headline

**Actions:**
- Verify Table 4.3 has E7 Config C results (LSTM+Corr, Net SR +0.510)
- Ensure all fold-by-fold breakdowns are present
- Add E8 RL results (underperformance explanation)
- Cross-check all numbers against `experiments/results/*.json`

---

#### Task 2.2: Finalize Figures & Tables
**Current state:** Some figures exist (`thesis_figures/` has equity/drawdown plots)

**Required figures:**
1. **Fig 4.1:** Frequency comparison (E1) — daily vs hourly equity curves
2. **Fig 4.2:** Hold period sweep (E2) — Net SR vs min_hold
3. **Fig 4.3:** Ablation (E3) — Net SR by selector (bar chart)
4. **Fig 4.4:** WFV equity curve — E7 Config C vs stat_only vs Nifty 50
5. **Fig 4.5:** Drawdown comparison — E7 vs baseline
6. **Fig 4.6:** Fold-by-fold performance — OOS Net SR per year
7. **Fig 5.1:** Cost sensitivity — Net SR vs transaction cost (bp)
8. **Fig 5.2:** Pair sector heatmap — which sectors pair most

**Actions:**
- Generate missing plots using `matplotlib` from JSON results
- Save to `reports/thesis_figures/`
- Embed in markdown with `![caption](thesis_figures/filename.png)`

---

#### Task 2.3: References Formatting
**Current state:** References.md exists with citations

**Actions:**
- Convert to standard format (IEEE / APA as per university requirements)
- Ensure all cited papers are in References.md
- Add missing citations from literature review
- Create `reports/references.bib` if LaTeX compilation needed

---

#### Task 2.4: Cross-Chapter Consistency Check
**Actions:**
- Ensure all numbers match across chapters (no stale results)
- Check that E7 Config C is consistently called "headline result"
- Verify terminology consistency (e.g., "selector" vs "selection algorithm")
- Ensure figure/table numbering is sequential

---

### Phase 3: Final Assembly (MEDIUM PRIORITY)
**Estimated time:** 2-3 hours

#### Task 3.1: Create Master Thesis Document
**Options:**
1. **Markdown compilation:** Use Pandoc to merge all chapters into single PDF
2. **LaTeX:** Convert to LaTeX if university requires specific template
3. **Word:** Export via Pandoc → .docx for review

**Command (Pandoc example):**
```bash
cd /d/code/Hybrid-Pairs-Trading-Ensemble/Implementation/reports
pandoc abstract.md chapter1_introduction.md chapter2_literature_review.md \
  chapter3_methodology.md chapter4_results.md chapter5_discussion.md \
  chapter6_conclusion.md appendix*.md \
  --bibliography=references.bib --csl=ieee.csl \
  -o thesis_full.pdf --toc --number-sections
```

---

#### Task 3.2: Create Executive Summary / README
**Purpose:** Standalone document for quick review

**Content:**
- 2-page summary of thesis
- Key findings (bullet points)
- How to reproduce experiments
- Repository structure guide

**Output:** `reports/EXECUTIVE_SUMMARY.md`

---

#### Task 3.3: Pre-Submission Checklist
- [ ] All figures have captions and are referenced in text
- [ ] All tables have captions and are numbered
- [ ] All experiments (E1-E8) are documented in Chapter 4
- [ ] All references are cited in standard format
- [ ] Abstract is ≤300 words
- [ ] Conclusion addresses all research questions
- [ ] Appendices provide full technical detail
- [ ] Code repository is clean (no dead files)
- [ ] KnowledgeGraph is updated to "Thesis Submitted" status

---

## 📊 Effort Estimation

| Phase | Tasks | Est. Time | Priority |
|-------|-------|-----------|----------|
| Phase 1 | Abstract, Ch6, Appendices | 4-6 hrs | 🔴 CRITICAL |
| Phase 2 | Integration, Figures, Polish | 3-4 hrs | 🟡 HIGH |
| Phase 3 | Assembly, Final Checks | 2-3 hrs | 🟢 MEDIUM |
| **TOTAL** | **12 tasks** | **9-13 hrs** | **~2 work days** |

---

## 🚀 Recommended Execution Order

### Today (Session 1: 3-4 hours)
1. Write Abstract (30 min)
2. Draft Chapter 6 structure (1 hr)
3. Write Chapter 6: Sections 1-3 (Summary, RQs, Implications) (1.5 hrs)
4. Update Chapter 4 with final E7 results (30 min)

### Tomorrow (Session 2: 3-4 hours)
5. Complete Chapter 6: Sections 4-5 (Limitations, Future Work) (1 hr)
6. Create Appendix A: Cost Model (45 min)
7. Create Appendix B: Hyperparameters (45 min)
8. Generate missing figures (1 hr)

### Day 3 (Session 3: 3-4 hours)
9. Create Appendix C & D (1 hr)
10. References formatting (45 min)
11. Master document compilation (30 min)
12. Final review & checklist (1 hr)

---

## 🎯 Success Criteria

**Thesis is complete when:**
- [x] Abstract written (200-300 words)
- [x] All 6 chapters exist and are polished
- [x] All appendices created
- [x] All figures generated and embedded
- [x] References formatted correctly
- [x] Master PDF compiled successfully
- [x] Executive summary created
- [x] Pre-submission checklist 100% complete

**Current completion:** ~70% (5/6 chapters done, experiments complete)  
**Target completion:** 100% within 2-3 work days

---

## 📝 Next Immediate Action

**START HERE:**

```bash
# Navigate to reports directory
cd /d/code/Hybrid-Pairs-Trading-Ensemble/Implementation/reports

# Create abstract (HIGHEST PRIORITY)
# Use AI agent to draft based on KnowledgeGraph context

# Then proceed to Chapter 6
```

**Prompt for AI agent:**
> "Draft the Abstract for the Hybrid Pairs Trading Ensemble M.S. thesis. Use the following key results: E7 Config C (LSTM+Correlation+OU) achieves Net Sharpe Ratio +0.510 and Net Return +17.66% on 6-year OOS walk-forward validation (2020-2025) on 35 NSE large-cap stocks. The thesis contributes: (1) first 8-model ensemble on NSE including GNN/Transformer/LSTM, (2) empirical proof that daily data outperforms hourly (SR 1.14 vs 0.49), (3) optimal hold period of 30 days, (4) realistic Indian cost model with 60bp round-trip costs. Target 250 words, academic style."

---

## 🔄 Update Protocol

After each session:
1. Update this plan's checkboxes
2. Update `KnowledgeGraph/KnowledgeGraph.md` "Active State" section
3. Update `experiments.json` if any new results are added
4. Commit progress to git (if version controlled)

**Last session:** 2026-05-26 — Plan created  
**Next session:** TBD — Start with Abstract
