# Literature Review Action Checklist

---

## To Do

- Comprehensive search of 2022-2026 literature completed
- 15 high-quality recent papers identified
- 2 NSE-specific papers found (Patel 2023, Patel 2025)
- All pre-2022 papers have implementations (11 papers)
- PCA-OU reproduction complete (negative result documented)
- Current literature review covers classical + early DL well

---


### Clarify Sharpe Ratio Definitions:

- [ ] **Review your Sharpe ratio calculation**
  - Check if you report "Net SR" = after all transaction costs
  - Verify calculation: SR = (mean_return - risk_free) / std_return
  - Document exact formula in thesis methodology

- [ ] **Compare with methodologies of diff papers**
  - Once paper downloaded, check their Sharpe definition
  - Are they reporting gross or net?
  - What cost model do they use?
  - Document differences in thesis

- [ ] **Explain performance gap in thesis**
  - If any paper's Sharpe Ratio > your Sharpe Ratio, provide clear explanation:
    - Different universe (Nifty 50 vs 35 stocks)?
    - Different cost models (gross vs net)?
    - Different metrics (annual SR vs net SR)?
    - Different time periods?
  - **Critical:** Must address before submission

### Update Literature Review Files:
  - Update comparison table with recent papers
  - Add NSE-specific subsection

- [ ] **Create placeholder folders:**
  
  Format = yyyy-type-auth-shortpapername
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

- [ ] **Create comparison spreadsheet**
  - Excel/CSV with all papers, methods, results
  - Easy reference for writing thesis
  - Include: Year, Authors, Method, Market, Sharpe, Notes
---

### Potential Experiments:

- [ ] **Run cost sensitivity analysis**
  - Most papers don't model costs properly
  - Your NSE cost model is a strength
  - Show how Sharpe changes at different cost levels
  - Demonstrate robustness or highlight cost sensitivity

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


## 📊 Current Status Dashboard

### Literature Coverage:
- **Classical (1987-2010):** ✅ 5/5 papers documented + implemented
- **ML Era (2010-2020):** ✅ 2/2 papers documented + implemented
- **Early DL (2017-2021):** ✅ 4/4 papers documented + implemented
- **Recent DL (2022-2026):** ❌ 0/15 papers in review (CRITICAL GAP)
- **Reinforcement Learning:** ✅ 1/1 papers documented + tested
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
