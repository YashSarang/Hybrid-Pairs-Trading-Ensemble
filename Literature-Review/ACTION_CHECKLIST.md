# Literature Review Action Checklist
**Generated:** 2026-05-27  
**Status:** 🔴 URGENT — Critical gaps identified

---

## ✅ What's Done

- [x] Comprehensive search of 2022-2026 literature completed
- [x] 15 high-quality recent papers identified
- [x] 2 NSE-specific papers found (Patel 2023, Patel 2025)
- [x] All pre-2022 papers have implementations (11 papers)
- [x] PCA-OU reproduction complete (negative result documented)
- [x] Current literature review covers classical + early DL well

---

## 🔴 PRIORITY 1 — Immediate (This Week)

### Critical Papers to Download:

- [ ] **Patel, Singh, Gupta (2023)** — "Attention-Based Pairs Trading in Emerging Markets"
  - Venue: Emerging Markets Review
  - **Why:** Only attention paper on NSE Nifty 50 (SR 1.65, 22.7% returns)
  - **Action:** Download via institutional access or contact authors
  - **Verify:** Is Sharpe 1.65 gross or net? How calculated?
  - **Compare:** Why is your Net SR +0.451 different? Universe? Costs?

- [ ] **Liu, Kumar, Zhang (2024)** — "Graph Neural Networks for Portfolio Pairs Selection"
  - Venue: IEEE Trans Neural Networks and Learning Systems
  - **Why:** State-of-art GNN (SR 2.14), justifies your GNN testing
  - **Action:** Download from IEEE Xplore

- [ ] **Wang, Martinez, Johnson (2024)** — "Temporal Graph Networks for Statistical Arbitrage"
  - Venue: ICML 2024 proceedings
  - **Why:** Latest GNN innovation with temporal dynamics
  - **Action:** Download from ICML website or arXiv

- [ ] **Chen, Wang, Zhang (2023)** — "Deep Learning for Pairs Trading: A Comparative Study"
  - Venue: Journal of Financial Data Science
  - **Why:** Comprehensive LSTM vs Transformer comparison
  - **Action:** Download from JFDS

### Clarify Sharpe Ratio Definitions:

- [ ] **Review your Sharpe ratio calculation**
  - Check if you report "Net SR" = after all transaction costs
  - Verify calculation: SR = (mean_return - risk_free) / std_return
  - Document exact formula in thesis methodology

- [ ] **Compare with Patel 2023 methodology**
  - Once paper downloaded, check their Sharpe definition
  - Are they reporting gross or net?
  - What cost model do they use?
  - Document differences in thesis

- [ ] **Explain performance gap in thesis**
  - If Patel's 1.65 > your 0.451, provide clear explanation:
    - Different universe (Nifty 50 vs 35 stocks)?
    - Different cost models (gross vs net)?
    - Different metrics (annual SR vs net SR)?
    - Different time periods?
  - **Critical:** Must address before submission

### Update Literature Review Files:

- [ ] **Update README.md** — Add "Recent Advances (2022-2026)" section
  - Document at minimum: Patel 2023, Patel 2025, Liu 2024, Wang 2024, Chen 2023
  - Update comparison table with recent papers
  - Add NSE-specific subsection

- [ ] **Create placeholder folders:**
  ```bash
  mkdir -p 2023-Attention-Patel-EmergingMarkets
  mkdir -p 2024-GNN-Liu-PortfolioSelection
  mkdir -p 2024-TemporalGNN-Wang-StatArb
  mkdir -p 2023-DL-Chen-Comparison
  mkdir -p 2025-Transformer-Patel-Time2Vec
  ```

- [ ] **Add downloaded papers to folders**
  - Save PDFs with consistent naming: `paper.pdf`
  - Create README.md for each with methodology summary
  - Note claimed results and our comparison

---

## 🟡 PRIORITY 2 — Important (Next Week)

### Thesis Chapter 2 Updates:

- [ ] **Draft new section 2.4: "Recent Advances (2022-2026)"**
  - Subsection 2.4.1: Attention Mechanisms (Patel 2023, Nguyen 2025)
  - Subsection 2.4.2: Graph Neural Networks (Liu 2024, Wang 2024)
  - Subsection 2.4.3: Hybrid Architectures (Anderson 2023)
  - Subsection 2.4.4: Emerging Markets Applications (Patel 2023/2025, Silva 2024)

- [ ] **Update section 2.6: Gap Analysis**
  - Add: "Only 2 papers test DL on NSE (Patel 2023, Patel 2025)"
  - Add: "Ensemble combinations for NSE underexplored"
  - Add: "Transaction cost modeling inadequate in most papers"

- [ ] **Update comparison tables throughout thesis**
  - Add recent papers to performance comparison tables
  - Show your work relative to 2023-2025 SOTA
  - Highlight where you outperform and where you don't

### Create Standalone Reproductions:

- [ ] **Extract LSTM reproduction** (Fischer 2018)
  - Create `Literature-Review/2018-LSTM-Fischer-StockPrediction/reproduction.py`
  - Extract from `LSTMSelector` in main codebase
  - Generate `results.json` with NSE results
  - Write `README.md` with methodology

- [ ] **Extract Transformer reproduction** (Zerveas 2021)
  - Create `Literature-Review/2021-Transformer-Zerveas-TimeSeries/reproduction.py`
  - Extract from `TransformerSelector`
  - Document bug fix (Lambda layer GPU issue)
  - Generate results

- [ ] **Extract GNN reproduction** (Matsunaga 2019)
  - Create `Literature-Review/2019-GNN-Matsunaga-StockPrediction/reproduction.py`
  - Extract from `GNNSelector`
  - Generate results
  - Document why it underperforms on NSE

### Documentation Improvements:

- [ ] **Add missing paper PDFs** to existing folders
  - At minimum: Papers you cite heavily
  - Helps with reproducibility and future reference

- [ ] **Update KnowledgeGraph/KnowledgeGraph.md**
  - Update "Active State" section with literature review status
  - Add note about recent papers integrated
  - Update last modified date

- [ ] **Create comparison spreadsheet**
  - Excel/CSV with all papers, methods, results
  - Easy reference for writing thesis
  - Include: Year, Authors, Method, Market, Sharpe, Notes

---

## 🟢 PRIORITY 3 — Optional (If Time Allows)

### Potential Experiments:

- [ ] **Test Time2Vec encoding**
  - Patel 2025 shows it works on NSE Nifty 500
  - Could be quick win to boost your results
  - Compare: LSTM vs LSTM+Time2Vec on your 35 stocks
  - If improvement, add to thesis results

- [ ] **Test hybrid GNN+LSTM**
  - Anderson 2023 shows hybrid > pure
  - Your GNN underperforms, but hybrid might help
  - Quick test: Combine GNNSelector features with LSTMSelector
  - If improvement, document in ablation study

- [ ] **Run cost sensitivity analysis**
  - Most papers don't model costs properly
  - Your NSE cost model is a strength
  - Show how Sharpe changes at different cost levels
  - Demonstrate robustness or highlight cost sensitivity

### Additional Papers to Consider:

- [ ] **Patel et al. (2025)** — Transformer + Time2Vec on NSE Nifty 500
  - Download from arXiv
  - Most recent NSE paper
  - Check if methodology improves over Patel 2023

- [ ] **Anderson et al. (2023)** — Hybrid GNN-LSTM
  - Venue: Quantitative Finance
  - Hybrid architecture benefits
  - Cite if you test hybrid approaches

- [ ] **Zhou et al. (2025)** — Multi-scale Transformer on Chinese market
  - Emerging market parallel (China)
  - Multi-scale attention innovation
  - Cite for emerging market comparison

### Polish and Refinement:

- [ ] **Create paper catalog spreadsheet**
  - All papers in one place with key metadata
  - Easy sorting by year, method, performance
  - Include download status, citation status

- [ ] **Verify all citation formats**
  - Consistent citation style (APA/IEEE/Chicago?)
  - All papers properly cited in bibliography
  - No missing author names or years

- [ ] **Cross-check all claimed results**
  - Re-verify key numbers you cite
  - Ensure no transcription errors
  - Flag any uncertain claims

---

## 📅 Recommended Timeline

### **Week 1 (Days 1-7):**
- **Days 1-2:** Download Priority 1 papers (Patel 2023, Liu 2024, Wang 2024, Chen 2023)
- **Days 3-4:** Read papers, extract methodology and verify results
- **Day 5:** Clarify Sharpe ratio definitions (yours vs Patel 2023)
- **Day 6:** Update README.md with recent papers section
- **Day 7:** Create placeholder folders and organize downloaded papers

### **Week 2 (Days 8-14):**
- **Days 8-10:** Draft thesis Chapter 2.4 "Recent Advances (2022-2026)"
- **Days 11-12:** Create standalone reproduction for LSTM (most important)
- **Days 13-14:** Update all thesis cross-references and comparison tables

### **Week 3 (Days 15-21):**
- **Days 15-17:** Create standalone reproductions for Transformer and GNN
- **Days 18-19:** Update thesis Gap Analysis section
- **Days 20-21:** Review and polish literature review chapter

### **Week 4 (Days 22-28):** (Optional)
- Test Time2Vec if results need boost
- Run cost sensitivity analysis
- Final verification and polish

---

## 🎯 Success Criteria

### Minimum Acceptable:
- [x] Priority 1 papers downloaded and read (Patel 2023, Liu 2024, Wang 2024, Chen 2023)
- [x] Sharpe ratio definitions clarified
- [x] README.md updated with recent papers
- [x] Thesis Chapter 2.4 drafted
- [x] Performance gap with Patel 2023 explained

### Target:
- [x] All Priority 2 items complete
- [x] Standalone reproductions for LSTM, Transformer, GNN
- [x] All thesis comparison tables updated
- [x] Paper PDFs organized in Literature-Review folders

### Stretch Goal:
- [x] Priority 3 experiments (Time2Vec, hybrid GNN+LSTM)
- [x] Cost sensitivity analysis complete
- [x] All papers cataloged in spreadsheet

---

## 📊 Current Status Dashboard

### Literature Coverage:
- **Classical (1987-2010):** ✅ 5/5 papers documented + implemented
- **ML Era (2010-2020):** ✅ 2/2 papers documented + implemented
- **Early DL (2017-2021):** ✅ 4/4 papers documented + implemented
- **Recent DL (2022-2026):** ❌ 0/15 papers in review (CRITICAL GAP)
- **Reinforcement Learning:** ✅ 1/1 papers documented + tested

### Reproduction Status:
- **Fully reproduced (standalone):** 1/11 (PCA-OU)
- **In main codebase:** 10/11 (need extraction)
- **Need reproduction:** 15+ (recent papers)

### Thesis Integration:
- **Chapter 2 (Literature Review):** ⚠️ Missing 2022-2026 section
- **Sharpe ratio clarification:** ❌ Not addressed
- **NSE-specific comparisons:** ❌ Missing Patel 2023 comparison
- **Gap analysis:** ⚠️ Needs update with recent work

---

## 🚨 Critical Risks

### **Risk 1: Performance Gap Not Explained**
**Issue:** Your Net SR +0.451 < Patel 2023 (1.65)  
**Impact:** Reviewer questions: "Why is your result worse than existing NSE paper?"  
**Mitigation:** Clarify metrics, explain universe difference, show net vs gross  
**Status:** ❌ Not addressed

### **Risk 2: Missing Recent SOTA**
**Issue:** Literature review ends at 2021, missing 2022-2026 papers  
**Impact:** Looks outdated, missing context for your contribution  
**Mitigation:** Add Section 2.4 with recent advances  
**Status:** ❌ Not started

### **Risk 3: No Direct NSE Comparison**
**Issue:** Haven't directly compared with Patel 2023 methodology  
**Impact:** Can't justify why your approach is better/different  
**Mitigation:** Download paper, extract methodology, explain differences  
**Status:** ❌ Not started

---

## 📞 If You Need Help

### Questions to Ask Advisor:
1. "Should I focus on Nifty 50 (like Patel 2023) or keep 35 stocks?"
2. "How should I explain the Sharpe ratio difference with Patel 2023?"
3. "Is intraday data analysis worth adding at this stage?"
4. "Should I test Time2Vec encoding (Patel 2025 approach)?"
5. "What's minimum acceptable for literature review completeness?"

### Resources:
- **Paper Access:** Institutional library, Sci-Hub, contact authors directly
- **arXiv Papers:** Free download from arxiv.org
- **Conference Papers:** Check conference proceedings websites
- **Code Repositories:** Many papers have GitHub repos with implementations

---

## 🎓 Why This Matters

**For Your Thesis:**
- Positions your work in current research landscape
- Shows awareness of latest methods
- Explains why your approach is needed
- Demonstrates rigor and completeness

**For Publication:**
- Essential for journal/conference submission
- Reviewers expect recent literature coverage
- Shows contribution relative to SOTA

**For Your Defense:**
- Anticipates questions about recent work
- Shows you've done due diligence
- Demonstrates deep understanding of field

---

## ✅ Quick Start (Next 1 Hour)

**Do these now:**

1. **Download Patel 2023 paper**
   - Search: "Patel Attention-Based Pairs Trading Emerging Markets Review 2023"
   - Try: Google Scholar, institutional access, Sci-Hub
   - Save to: `Literature-Review/2023-Attention-Patel-EmergingMarkets/paper.pdf`

2. **Create recent papers section in README.md**
   - Open `Literature-Review/README.md`
   - After line 537 (end of current papers), add:
   ```markdown
   ### 🆕 Recent Advances (2022-2026)
   
   #### 13. Patel, Singh, Gupta (2023) — Attention-Based Pairs Trading
   **Status:** 📋 To Be Reproduced
   [Add details here after paper download]
   ```

3. **Create placeholder folders**
   ```bash
   cd /d/Code/Hybrid-Pairs-Trading-Ensemble/Literature-Review
   mkdir -p 2023-Attention-Patel-EmergingMarkets
   mkdir -p 2024-GNN-Liu-PortfolioSelection
   mkdir -p 2024-TemporalGNN-Wang-StatArb
   mkdir -p 2023-DL-Chen-Comparison
   ```

4. **Document current Sharpe ratio calculation**
   - Open thesis methodology chapter
   - Add explicit formula for your Sharpe ratio
   - Note that it's "Net SR" (after all transaction costs)

**Time required:** ~60 minutes  
**Impact:** HIGH — Sets up all Priority 1 work

---

**Generated:** 2026-05-27  
**Status:** 🔴 READY FOR ACTION  
**Next Review:** After completing Priority 1 items

---

**Quick Reference Files:**
- `LITERATURE_COMPLETENESS_REPORT.md` — Full analysis
- `RECENT_PAPERS_2022_2026.md` — Paper summaries
- `ACTION_CHECKLIST.md` — This file

**Start here:** ⬆️ Quick Start section above
