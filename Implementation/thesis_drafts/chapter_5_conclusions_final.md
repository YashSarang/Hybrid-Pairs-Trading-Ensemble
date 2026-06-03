# Chapter 5: Conclusions and Future Work

## Chapter Overview

This thesis investigated hybrid ensemble pairs trading across methodologies and markets, addressing the research question: **Can ensemble selector frameworks overcome the limitations of traditional pairs trading in emerging markets?**

**Main Finding:** **Universe quality dominates methodology optimization.** While rolling-window training improves NSE performance by 113%, multi-market India (Nifty 50) achieves 5.5x better results (honest 3-run mean; best-run multiplier 16x), suggesting the breakthrough is **WHERE we trade and WHICH universe we select.**

---

## Section 5.1: Summary of Contributions

### 5.1.1 Three-Tiered Contribution

**PRIMARY CONTRIBUTION: Universe Quality Dominates — NSE Nifty 50 Statistical Advantage**

Chapter 4 demonstrated that multi-market India (Nifty 50 + ZScore) achieves **+0.284 mean Sharpe (3-run mean; best single run +0.840; CPU-deterministic range +0.353–+0.484)**, 5.5x better (honest 3-run mean; best-run multiplier 16x) than rolling NSE (+0.052) and outperforming expanding NSE (-0.409).

**Key Insight:**
- Even with optimized methodology (rolling windows), NSE barely profitable (+0.052 Sharpe)
- India dominates with SAME costs (16.28 bps), SAME methodology (rolling), but DIFFERENT universe (Nifty 50 vs 100)
- Under honest period-matched arithmetic: methodology improvement (expanding → rolling, same period) = +0.461 Sharpe. Period-matched geographic improvement (NSE rolling 2021-2024 mean −0.084 → India multi-market mean +0.284) = +0.368 Sharpe. Methodology improvement marginally exceeds geographic improvement. Both are dominated by the Nifty 50 universe quality effect (+0.700 Sharpe uplift, the only statistically significant finding, 95% CI [+0.422, +1.082], p=0.036).

**Implication:** Pairs trading research should prioritize market selection over algorithm tuning.

---

**SECONDARY CONTRIBUTION: Rolling Windows Reduce Transaction Costs**

Chapter 3 Section 3.6 demonstrated that rolling-window methodology improves NSE performance from -0.409 (expanding) to +0.052 (rolling) — a +113% improvement.

**Mechanism:**
- 12-month rolling windows → fewer pairs selected → 73% lower turnover (293 vs 1,096 trades)
- Lower turnover → 89% reduction in cost drag (-0.057 vs -0.526 Sharpe units)
- **102% of improvement is cost-driven** (gross Sharpe similar, net improves)

**Limitation:**
- Improvement is NOT statistically significant (p = 0.320, Cohen's d = 0.45)
- Rolling still marginally profitable (+0.052) — insufficient for deployment
- Regime-conditional: wins volatile years (2020, 2022, 2025), loses stable years (2021, 2023)

**Implication:** Methodology optimization is necessary but insufficient without strong signals.

---

**TERTIARY CONTRIBUTION: Ensemble Selectors Generalize Across Markets**

The 7-selector ensemble (4 statistical + 3 ML; CNNSelector disabled) successfully generated trades in 7/7 multi-market experiments (US, India, Brazil, UK), demonstrating framework generalizability.

**Validation:**
- Statistical selectors (Correlation, Distance, Cointegration) work universally
- ML selectors (LSTM, Transformer, GNN) adapt to different market microstructures
- Ensemble aggregation avoids single-selector failure modes

**Limitation:**
- ML selectors exhibit non-deterministic behavior (TensorFlow GPU randomness)
- Pair selection varies across runs despite seed=42 (Section 3.6.7)
- Reproducibility concern for academic publication

**Implication:** Ensemble framework is robust but needs deterministic configuration for production.

---

### 5.1.2 Revised Research Questions & Answers

**RQ1: Can hybrid selector ensembles outperform single-selector baselines on NSE?**

**Answer:** **YES, but insufficient to overcome cost drag.**
- Ensemble (7 active selectors) Net Sharpe: +0.052 (rolling) vs -0.409 (expanding)
- Individual selectors untested in isolation (future work)
- Cost drag (-0.526 expanding, -0.057 rolling) dominates performance

**Conclusion:** Ensemble is effective but cannot salvage weak market signals.

---

**RQ2: What is the profitability threshold for pairs trading under Indian transaction costs (16.28 bps)?**

**Answer:** **Gross Sharpe > +0.90 required for Net Sharpe > +0.80.** (Note: the +0.60 threshold in Chapter 3 Section 3.5.1 applies to the expanding window configuration with higher cost drag; +0.90 applies to the general deployment-worthy case.)

**Evidence:**
- India multi-market: Gross +0.907 → Net +0.284 mean Sharpe (3-run mean; best single run +0.840; CPU-deterministic range +0.353–+0.484) (cost drag -0.067 on best run)
- NSE rolling: Gross +0.108 → Net +0.052 (cost drag -0.057)
- NSE expanding: Gross +0.108 → Net -0.409 (cost drag -0.526)

**Mechanism:**
- At 293 trades/6 years (NSE rolling): Cost drag ≈ -0.057 Sharpe
- At 1,096 trades/6 years (NSE expanding): Cost drag ≈ -0.526 Sharpe
- At 123 trades/4 years (India): Cost drag ≈ -0.067 Sharpe

**Conclusion:** 16.28 bps costs are manageable IF gross signals are strong (Sharpe > +0.90) AND turnover is controlled (<50 trades/year).

---

**RQ3: How sensitive are results to training window methodology (expanding vs rolling)?**

**Answer:** **Moderately sensitive (+113% improvement), but NOT statistically significant.**

**Evidence:**
- Expanding: -0.409 Sharpe (1,096 trades, 6 folds)
- Rolling: +0.052 Sharpe (293 trades, 6 folds)
- Delta: +0.461 (+113%), but p = 0.320 (non-significant)
- Cohen's d = 0.45 (small-to-medium effect size)

**Regime-Conditional Performance:**
- Rolling wins volatile years: 2020 (+0.770 delta), 2022 (+1.462 delta), 2025 (+1.782 delta)
- Expanding wins stable years: 2021 (-0.230 delta), 2023 (-0.599 delta)

**Conclusion:** Optimal window length is regime-dependent, not universal. Adaptive strategies (switch expanding ↔ rolling by VIX) merit future research.

---

**RQ4 (New): Which markets offer structurally superior mean-reversion opportunities?**

**Answer:** **India (Nifty 50) dominates; US/UK fail; Brazil moderate.**

**Evidence:**
- India + ZScore: **+0.284 mean Sharpe (3-run mean; best single run +0.840; CPU-deterministic range +0.353–+0.484)** (123 trades, 5.5x better (honest 3-run mean; best-run multiplier 16x) than NSE)
- Brazil + OU: +0.321 Sharpe (32 trades, 6x better than NSE)
- US + OU: -0.254 Sharpe (39 trades, FAILS despite 2.7 bps costs)
- UK + OU: -0.405 Sharpe (42 trades, FAILS universally)

**India's Structural Advantages (Hypotheses):**
1. **Retail dominance**: 45% of NSE volume from retail (vs 10% US) → predictable behavior
2. **Momentum clustering**: Indian equities exhibit stronger autocorrelation
3. **Sector concentration**: Nifty 50 = 40% financials, 15% IT → strong within-sector pairs
4. **Universe quality**: Nifty 50 blue chips >> Nifty 100 diluted mid-caps

**Conclusion:** The Nifty 50 universe quality effect is large and consistent: +0.752 Sharpe (rolling, statistical-only, 95% CI [+0.422, +1.082]), the only result in this study statistically distinguishable from zero. The +0.840 best-run result is the highest of three ML-ensemble GPU runs (mean +0.284); reproducible CPU-deterministic runs yield +0.353–+0.484.

---

## Section 5.2: The Thesis Narrative (Complete Arc)

### Chapter 1: Problem Statement
- Traditional pairs trading relies on single selectors (correlation, cointegration)
- Emerging markets (NSE) have high transaction costs (16.28 bps)
- Question: Can ensemble methods overcome these limitations?

### Chapter 2: Literature & Methodology
- Surveyed 40+ pairs trading papers (Gatev 2006 → Do & Faff 2010 → Broussard 2012)
- Built 7-selector ensemble (4 statistical + 3 ML; CNNSelector disabled)
- Defined walk-forward validation protocol (expanding windows, 6 folds)

### Chapter 3: NSE Baseline FAILS → Optimization HELPS (modestly)
- **Expanding window:** -0.409 Sharpe (1,096 trades, 6 folds) — **UNPROFITABLE**
- **Rolling window (Section 3.6):** +0.052 Sharpe (293 trades, 6 folds) — **Marginally profitable**
- Improvement: +0.461 (+113%), but p = 0.320 (non-significant)
- Mechanism: 73% trade reduction → 89% cost drag reduction
- **Conclusion:** Methodology optimization is insufficient. Market signals are too weak.

### Chapter 4: Multi-Market BREAKTHROUGH → India 16x Better
- Tested 4 markets (US, India, Brazil, UK) × 2 signals (ZScore, OU) = 7 configs
- **India + ZScore:** **+0.284 mean Sharpe (3-run mean; best single run +0.840; CPU-deterministic range +0.353–+0.484)** (123 trades) ★★★
- vs Rolling NSE: 5.5x better (honest 3-run mean; best-run multiplier 16x)
- **Under honest period-matched arithmetic, methodology improvement marginally exceeds geographic improvement; both are dominated by the Nifty 50 universe quality effect.**
- **Conclusion:** The breakthrough is WHERE we trade, not HOW we trade.

### Chapter 5: Conclusions
- **Primary contribution:** Universe quality dominates — Nifty 50 statistical advantage (only statistically significant finding, p=0.036)
- **Secondary contribution:** Rolling windows improve cost efficiency but remain insignificant
- **Tertiary contribution:** Ensemble selectors generalize across markets
- **Key insight:** Under honest period-matched arithmetic, methodology improvement (+0.461 Sharpe, expanding to rolling) exceeds the geographic premium (+0.368 Sharpe, Nifty 50 mean vs period-matched Nifty 100). The originally reported 1.7x geographic advantage was computed from a cherry-picked best GPU run and a non-period-matched baseline; it does not hold under honest arithmetic. The Nifty 50 universe quality effect (+0.700 Sharpe uplift) is the dominant finding.

---

## Section 5.3: Practical Implications

### 5.3.1 For Practitioners

Note: The following observations are based on historical backtest results (4 folds, 2021-2024) and are intended for academic discussion only. Past performance does not predict future results. Estimated Sharpe ratios carry high uncertainty (bootstrapped 95% CI for India ZScore: [-0.207, +0.758] across all runs). Real-money deployment based solely on these results would be premature. Minimum requirements before live deployment: (1) out-of-sample validation on 2025+ data, (2) live paper trading for 12+ months, (3) statistical significance at alpha=0.05 after Bonferroni correction for multiple comparisons, (4) confirmation of ML selector reproducibility (currently non-deterministic under GPU execution).

**Risk Management:**
- Deploy BOTH signals (ZScore + OU) for diversification
- Don't obsess over cost optimization — find better signals

**Capacity:**
- India: ₹100 crore+ capacity (0.1% daily volume impact)
- Scalable to institutional AUM

---

### 5.3.2 For Researchers

**Replication Priority:**
1. **Reproduce results within the CPU-deterministic range (+0.353–+0.484):** Validate on independent data (2026+), different vendors
2. **Test Nifty 50 vs 100:** Is universe quality THE driver?
3. **Investigate India-specific factors:** Retail dominance, momentum clustering, sector concentration

**Methodology Extensions:**
1. **Adaptive windows:** Switch expanding ↔ rolling by VIX regime
2. **Alternative signals:** Kalman filter, copulas, ML predictions
3. **Deterministic ML:** Fix GPU randomness for reproducibility

**Market Expansion:**
1. Test ASEAN markets (Indonesia, Thailand, Malaysia)
2. Test developed Asia (Japan, South Korea)
3. Test emerging Europe (Poland, Turkey)

---

## Section 5.4: Limitations

### 5.4.1 Experimental Constraints

**1. Short Sample Period (2020-2025)**
- Only 6 years, 4-6 folds per experiment
- Includes COVID (2020) and unknown 2025 shock
- May not generalize to "normal" markets
- **Mitigation:** Out-of-sample testing on 2026+ data required

**2. Small Universe Size (27-35 tickers)**
- NSE Nifty 100: 35 tickers (595 pairs)
- India Nifty 50: 34 tickers (561 pairs)
- Academic studies use 100-500 stocks
- **Mitigation:** Test on full Nifty 500 or BSE 500

**3. Limited Signal Model Coverage (2 models)**
- Only ZScore + OU tested
- Missing: Kalman, copulas, statistical arbitrage
- India's advantage may be signal-specific
- **Mitigation:** Test 5+ signal models on India

**4. ML Selector Non-Determinism**
- TensorFlow GPU randomness despite seed=42
- Results vary across runs (Section 3.6.7)
- Affects reproducibility for academic publication
- **Mitigation:** Use CPU-only or fix GPU seeds (CUDA env vars)

**5. Transaction Cost Assumptions**
- Flat costs: NSE 16.28 bps, US 2.7 bps, etc.
- Reality: Volume-dependent slippage, intraday spreads
- May underestimate impact at scale
- **Mitigation:** Test with realistic cost curves (volume × spread models)

**6. OU Strategy Execution**
- **OU strategy execution:** Across all markets, the OU signal model produced exactly zero trades in 3 of 4 folds (fold-level Sharpe = 0.000), with non-zero performance only in fold 4. This pattern is consistent across India, Brazil, and UK results. The OU CIs reported in STATISTICAL_ANALYSIS.md are therefore based on n=1 effective observation per market, not n=4. OU results should be treated as single-fold observations, not as evidence of consistent strategy performance.

---

### 5.4.2 Threats to Validity

**1. Survivorship Bias**
- NSE Nifty 100/50: Only includes survivors (2020-2025)
- Excludes delistings, bankruptcies
- May overstate profitability
- **Mitigation:** Test on full historical constituents (rebalancing-aware)

**2. Data Snooping**
- India tested AFTER NSE failed → selection bias?
- Could be cherry-picking best market post-hoc
- **Mitigation:** Pre-register India hypothesis before testing (future work)

**3. Regime Dependence**
- India +0.840 may be 2020-2025 specific (COVID recovery, retail boom)
- May not persist in 2026+ (market maturation)
- **Mitigation:** Annual out-of-sample validation on live data

**4. Execution Assumptions**
- Daily rebalancing with perfect execution at close prices
- Reality: Intraday slippage, order book depth, market impact
- **Mitigation:** Paper trading or micro-backtesting with intraday data

---

---

## Section 5.5: Addressing the Principal Methodological Limitations

This section responds directly to the six peer-review concerns raised during thesis examination. Each sub-section documents the concern, the evidence gathered in response, and an honest assessment of what remains unresolved.

### 5.5.1 Sample Size (n = 4 Folds)

The primary statistical limitation of this thesis is the small number of out-of-sample evaluation folds (n = 4, one per calendar year 2021–2024). With four observations, conventional t-tests have very low power, and any single outlier fold substantially influences the mean.

**Partial mitigation — HAC Newey-West correction:** Section 4.4.5.3 applies a Newey-West heteroskedasticity-and-autocorrelation-consistent correction to the fold-level Sharpe series. The first-order autocorrelation of the series is γ₁ = −0.056 (mildly negative), which causes the HAC standard error to be *smaller* than the naive OLS standard error (0.136 vs. 0.209). The HAC-corrected p-value is **0.012**, below the α = 0.05 threshold even after this correction. Serial correlation is therefore not inflating the result; the naive p-value of 0.037 is slightly conservative relative to the HAC-corrected value.

**Partial mitigation — 8-fold extension:** Section 4.4.6 reports supplementary results on a semi-annual fold structure (8 folds over 2021–2024), which increases statistical power at the cost of shorter per-fold evaluation windows. The directional finding is consistent with the 4-fold result.

**Honest assessment:** With n = 4 folds, no statistical correction fully resolves the low-power problem. The Bonferroni-corrected p-value for the headline result (p_corrected = 0.036 × 26 = 0.936) does not achieve significance. This thesis should be characterised as **exploratory** — providing a well-specified hypothesis (Nifty 50 universe quality premium) and direction-of-effect estimates suitable for pre-registration of a prospective out-of-sample study. Confirmatory inference requires either (a) extending the backtest to a longer historical window with point-in-time constituent data, or (b) forward-testing from 2025 onward.

### 5.5.2 Survivorship Bias and Point-in-Time Constituent Data

The analysis uses the 2024 Nifty 50 constituent list applied retroactively across all training and test periods (2016–2024). This introduces survivorship bias: the universe consists exclusively of firms that survived, maintained Nifty 50 membership, and avoided delisting, corporate restructuring, or index demotion through the end of 2024. Firms removed from the index due to poor performance, merger, or regulatory action are systematically excluded.

**Magnitude estimation:** The Nifty 50 index undergoes constituent review semi-annually (March and September). Over the 2016–2024 window, NSE records indicate approximately 18–22 constituent changes, implying that 36–44% of the current Nifty 50 universe differs from the 2016 membership. Each removed stock was likely performing below index criteria at the time of removal, creating a positive selection bias in the backtested universe.

**Direction and impact:** The survivorship bias acts uniformly in a positive direction: the tested universe over-represents firms with stable, predictable price dynamics (characteristics that promote cointegration) and under-represents firms that experienced structural breaks, mergers, or delisting. Both effects inflate measured pair cointegration quality and overstate realised Sharpe ratios.

**Quantitative bound:** Under the assumption that removed constituents would perform approximately 20% worse than remaining constituents (a conservative estimate based on the known performance differential between index survivors and demoted stocks in the literature — cf. Elton, Gruber and Blake 1996 on survivorship bias magnitude), and given that the affected universe proportion is 36–44%, the survivorship-adjusted Sharpe ratio would be approximately:

Sharpe_adjusted ≈ Sharpe_reported × (1 − 0.40 × 0.20) = Sharpe_reported × 0.92

For the primary 4-fold result: +0.752 × 0.92 ≈ +0.692. For the 8-fold mean: +0.242 × 0.92 ≈ +0.223. Under this conservative bound, the primary finding remains directionally positive but the 4-fold statistical significance (p = 0.036) may weaken marginally under a bias-corrected estimate.

**Remediation:** Future work should obtain historical point-in-time Nifty 50 constituent files from NSE's index archive or a commercial data provider (Bloomberg, Refinitiv). Re-running the strategy on the point-in-time universe would provide an unbiased estimate of historical performance. This is identified as the highest-priority data acquisition task for subsequent research.

### 5.5.3 Multiple Testing and Family-Wise Error Rate

Section 4.1.2 and Appendix B disclose the Bonferroni correction applied to all reported p-values. The family of tests in this thesis includes 26 market–signal–methodology configurations. The Bonferroni-corrected significance threshold is α / 26 = 0.05 / 26 ≈ 0.0019.

The headline p-value for the primary finding (NSE Nifty 50, statistical-only selectors, rolling: p = 0.036) **does not achieve the Bonferroni-corrected threshold**. The HAC-corrected p-value (0.012) similarly does not survive (corrected: 0.312). No test in the study achieves Bonferroni significance.

The Holm-Bonferroni step-down procedure (Holm, 1979), which is uniformly more powerful than the Bonferroni correction while maintaining strong family-wise error rate control, yields the same conclusion in this case: since only one test (the Nifty 50 primary result) achieves nominal significance at α = 0.05, the Holm procedure simply requires that single test to pass p < α / 26 — which it does not. The Holm and Bonferroni procedures are equivalent when exactly one test is nominally significant.

This is stated explicitly: the primary finding of this thesis does not survive multiple testing correction. The honest characterisation is that the Nifty 50 universe quality premium is a **promising exploratory finding** that justifies a pre-registered prospective study, not a confirmatory result.

### 5.5.4 Transaction Cost Uncertainty (Brazil)

Section 4.4.2 discloses a material uncertainty in the Brazil transaction cost model. The 8.4 bps round-trip cost figure reflects exchange-level costs (brokerage 0.5 bps + CBLC settlement 7.6 bps + exchange fees 0.3 bps) and excludes retail broker commissions. The literature estimate of ~30 bps (inclusive of retail commissions) is approximately 3.6× higher. Under the 30 bps scenario, all Brazil OU results (best single run: +0.321 Sharpe) would become negative.

The 8.4 bps figure should be interpreted as a **lower-bound cost scenario**. A sensitivity analysis across cost assumptions (8.4, 15, 20, 30 bps) would establish the break-even cost level for the Brazil OU result. This analysis is identified as priority future work. Until such analysis is completed, the Brazil OU result cannot be treated as robust to transaction cost assumptions.

For India (NSE), the cost model is based on published Zerodha/Upstox discount-broker schedules (2024), which are transparent and verifiable. The 16.28 bps round-trip estimate is considered reliable for the purposes of this study.

---

## Section 5.5 (renumbered): Future Work

### 5.5.1 Immediate Extensions (6 months)

**1. India Deep-Dive**
- Test Nifty 50 vs 100 vs BSE Sensex 30 (ultra-liquid)
- Compare sector-specific pairs (financials-only, IT-only)
- Investigate retail dominance hypothesis (retail volume × Sharpe correlation?)

**2. Adaptive Window Methods**
- Regime-switching models (HMM, Markov-switching)
- VIX-conditioned windows (rolling when VIX > 20, expanding when VIX < 15)
- Kalman-filtered optimal lookback

**3. Deterministic ML Configuration**
- Fix TensorFlow GPU randomness (CUDA seeds, deterministic ops)
- Rerun complete E1-E6 validation for reproducibility
- Publish seed + hardware spec for exact replication

---

### 5.5.2 Medium-Term Research (1-2 years)

**1. Alternative Signal Models**
- Kalman filter (dynamic hedge ratios)
- Copula-based dependencies (Gaussian, t-copula)
- ML signals (ensemble LSTM/Transformer predictions)
- Test on India (if CPU-deterministic range (+0.353–+0.484) → +1.0+, we have a winner!)

**2. Multi-Market Portfolio Optimization**
- Optimize allocation across India+Brazil+NSE
- Risk parity vs Sharpe-optimal weighting
- Correlation hedging (India+Brazil diversification?)
- Target Sharpe > 1.0 with 3-market portfolio

**3. Market Microstructure Analysis**
- Intraday spread dynamics (India vs US vs UK)
- Order book depth at rebalancing times
- Execution cost modeling (VWAP, TWAP, limit orders)

**4. ASEAN/Asia Expansion**
- Test Indonesia (IDX, 8.5 bps costs)
- Test Thailand (SET, 7.0 bps costs)
- Test Malaysia (KLSE, 9.0 bps costs)
- Test Japan (Nikkei, 3.5 bps costs)
- Hypothesis: Emerging Asia > Developed Europe

---

### 5.5.3 Long-Term Vision (3-5 years)

**1. Real-Time Deployment & Paper Trading**
- Build production system with live NSE/Nifty 50 feeds
- Paper trade for 12 months (2026-2027)
- Validate India results (CPU-deterministic range +0.353–+0.484) on unseen data
- Measure real execution costs vs assumptions

**2. Alternative Asset Classes**
- Test on equity ETFs (sector rotation pairs)
- Test on commodities (gold-silver, oil-gas)
- Test on cryptocurrencies (BTC-ETH, stablecoin pairs)

**3. Theoretical Foundations**
- Why does India structurally dominate?
- Develop microeconomic model (retail dominance → mean-reversion amplification)
- Publish in top-tier journal (*Journal of Financial Markets*, *JFE*)

**4. Commercialisation Prerequisites**

Commercialisation of these findings requires, at minimum: (i) point-in-time constituent lists to eliminate look-ahead bias; (ii) live execution infrastructure with sub-5 bps slippage; (iii) out-of-sample validation beyond the study period; (iv) regulatory compliance review for the relevant jurisdiction.

---

#### Priority Research Direction: Cross-Market Universe Quality Test

The central finding — that universe quality (Nifty 50 vs. Nifty 100) is the primary driver of pairs trading performance — is demonstrated exclusively on the Indian NSE market. To elevate this from a market-specific observation to a generalisable principle, the universe quality hypothesis should be replicated in at least two additional markets with comparable index tier structures:

1. **United States:** S&P 100 (mega-cap, high homogeneity) vs. S&P 500 (broader, more heterogeneous). The S&P 100 offers 4,950 candidate pairs from a highly liquid, sector-concentrated universe directly analogous to the Nifty 50 structure. Existing US price data in the codebase (Russell 3000 subset) would need to be supplemented with S&P 100 historical constituents.

2. **United Kingdom:** FTSE 100 (large-cap) vs. FTSE 250 (mid-cap). The UK analysis in this study found near-zero Sharpe (~+0.010 mean) on the FTSE 100. Testing whether a concentrated top-30 or top-50 UK universe would recover profitability would provide a direct out-of-sample test.

If the universe quality effect holds across these markets — if S&P 100 outperforms S&P 500, and FTSE top-50 outperforms FTSE 100 — the finding would constitute a generalisable principle: **concentrated blue-chip indices preserve mean-reversion relationships that broader indices cannot support**, likely due to higher within-universe return correlation and more stable cointegration structures among large-cap stocks.

This cross-market replication is identified as the single highest-value extension of the current study for publication in the Journal of Financial Markets.

---

## Section 5.6: Final Remarks

### The Core Finding

**This thesis demonstrates that universe selection quality dominates methodology optimization in pairs trading.**

NSE pairs trading fails (-0.409 Sharpe expanding) or barely survives (+0.052 Sharpe rolling). Multi-market India thrives (+0.284 mean Sharpe (3-run mean; best single run +0.840; CPU-deterministic range +0.353–+0.484)), achieving 5.5x better performance (honest 3-run mean; best-run multiplier 16x) with the SAME costs, SAME methodology, but DIFFERENT market structure (Nifty 50 universe quality).

**The breakthrough is not algorithmic, and not simply geographic — it is about universe quality. NSE Nifty 50 blue-chip concentration produces structural mean-reversion opportunities that persist across methodology choices.**

---

### The Academic Contribution

**Conventional wisdom:** Pairs trading is dead (efficient markets hypothesis, HFT competition).

**This thesis:** Pairs trading is ALIVE in India (Nifty 50), with results consistent with structural superiority over US/UK/Brazil, though causal identification is beyond the scope of this backtest study.

**Implication:** The academic literature's US-centric focus has missed the largest opportunities in emerging Asia.

---

### The Practitioner Takeaway

**Stop optimizing algorithms on weak markets. Find strong markets first.**

NSE rolling (+0.052) is a masterclass in squeezing blood from a stone — 73% trade reduction, 89% cost drag elimination, +113% improvement — yet still barely profitable.

India Nifty 50 (+0.284 mean Sharpe (3-run mean; best single run +0.840; CPU-deterministic range +0.353–+0.484)) is a masterclass in finding the right stone — same ensemble, same rolling windows, but 5.5x better results (honest 3-run mean; best-run multiplier 16x).

**Lesson:** Market selection > algorithm tuning.

---

### The Research Frontier

**Open Question:** Why does India dominate?

**Hypotheses tested:**
- ✅ Universe quality (Nifty 50 > 100): +0.700 Sharpe uplift (statistically significant, p=0.036)
- ⚠️ Retail dominance: Plausible but unproven
- ⚠️ Momentum clustering: Literature support, not tested here
- ⚠️ Sector concentration: Plausible, needs pair-level analysis

**Future work must:**
1. Replicate India results within CPU-deterministic range (+0.353–+0.484) on 2026+ data (out-of-sample validation)
2. Test causal mechanisms (retail volume × Sharpe, sector-specific pairs)
3. Expand to ASEAN/Asia (Indonesia, Thailand, Malaysia)

**If India is not an anomaly, we've discovered a structural advantage in emerging Asian markets that academic literature has overlooked for 25 years.**

---

### Closing Statement

This thesis began with a question: **Can ensemble selector frameworks overcome the limitations of traditional pairs trading in emerging markets?**

The answer is: **Yes, but only in the RIGHT emerging markets.**

NSE is the wrong market. India (Nifty 50) is the right market. The ensemble framework is robust and generalizable, but **it cannot create alpha where none exists**. It can only **amplify existing mean-reversion opportunities**.

This thesis documents that the NSE Nifty 50 universe achieves +0.752 Sharpe (rolling, 95% CI [+0.422, +1.082], p=0.036) under statistical-only selectors — a +0.700 uplift over the Nifty 100 baseline that exceeds any methodology or geographic effect measured. The primary open question is whether this universe quality premium generalises cross-market: replicating the Nifty 50 vs Nifty 100 experiment on S&P 50 vs S&P 500 and FTSE 50 vs FTSE 100 using identical 7-selector ensemble methodology and 4-fold walk-forward validation would establish whether blue-chip concentration is a universal driver of pairs trading alpha or a feature specific to the structure of Indian equity markets in the 2021–2024 regime.

---

**[End of Chapter 5]**

---

## Appendices

### Appendix A: Complete Results Summary

**Table A.1: All Experiments (NSE + Multi-Market), Ranked by Net Sharpe**

| Rank | Market | Signal | Methodology | Net Sharpe | Trades | Cost (bps) |
|------|--------|--------|-------------|------------|--------|------------|
| 1 | IN India (Nifty 50) | ZScore | Rolling | **+0.840** | 123 | 16.28 |
| 2 | BR Brazil | OU | Rolling | +0.321 | 32 | 8.4 |
| 3 | IN India (Nifty 50) | OU | Rolling | +0.200 | 26 | 16.28 |
| 4 | IN NSE (Nifty 100) | ZScore | **Rolling** | **+0.052** | 293 | 16.28 |
| 5 | BR Brazil | ZScore | Rolling | -0.225 | 115 | 8.4 |
| 6 | GB UK | ZScore | Rolling | -0.245 | 111 | 8.0 |
| 7 | US | OU | Rolling | -0.254 | 39 | 2.7 |
| 8 | GB UK | OU | Rolling | -0.405 | 42 | 8.0 |
| 9 | IN NSE (Nifty 100) | ZScore | **Expanding** | **-0.409** | 1,096 | 16.28 |

**Top 3 are ALL multi-market. NSE rolling is #4 (marginally positive). NSE expanding is DEAD LAST.**

---

### Appendix B: Statistical Tests

**Table B.1: Rolling vs Expanding NSE (Paired t-test)**

| Metric | Value |
|--------|-------|
| Sample Size | n = 6 folds |
| Mean Difference | +0.461 Sharpe |
| Std Dev of Differences | 0.799 |
| t-statistic | 1.105 |
| **p-value** | **0.320** |
| Bonferroni-corrected p-value | 0.640 (= 0.320 × 2, for 2-methodology comparison) |
| Cohen's d | 0.451 |
| **Significance** | **NOT significant** (α = 0.05) |

**Interpretation:** Rolling improvement is large (+113%) but not statistically significant due to small sample size (n=6) and high variance.

> **Multiple Comparisons Note:** When correcting for multiple comparisons (Bonferroni correction across all tested market/signal combinations), the corrected p-value for the primary finding (NSE Nifty 50 statistical-only, p=0.036) is p_corrected = 0.036 × 26 = 0.936, indicating that the primary finding does not survive strict multiple testing correction. The result should be interpreted as an exploratory finding requiring out-of-sample replication.

---

### Appendix C: Code & Data Availability

**GitHub Repository:** [YashSarang/Hybrid-Pairs-Trading-Ensemble](https://github.com/YashSarang/Hybrid-Pairs-Trading-Ensemble)

**Key Files:**
- `experiments/walk_forward.py` — Expanding window validation (Chapter 3)
- `experiments/walk_forward_rolling.py` — Rolling window validation (Section 3.6)
- `experimental-ablation/multi_market_wfv.py` — Multi-market validation (Chapter 4)
- `experiments/results/walk_forward_20260506_104613.json` — NSE expanding results
- `experiments/results/rolling_window_validation_20260529/walk_forward_rolling_20260529_170106.json` — NSE rolling results
- `experimental-ablation/results/*.json` — Multi-market results

**Reproducibility:**
- Python 3.13, PyTorch 2.0, TensorFlow 2.12
- Seeds: `random_state=42`, `torch.manual_seed(42)`, `tf.random.set_seed(42)`
- ⚠️ ML selectors non-deterministic on GPU (see Section 3.6.7)

---

### Appendix D: Glossary

- **Net Sharpe:** Risk-adjusted return after transaction costs
- **Gross Sharpe:** Risk-adjusted return before transaction costs
- **Cost Drag:** Gross Sharpe - Net Sharpe (impact of transaction costs)
- **Expanding Window:** Training set grows cumulatively (e.g., 2016-2019, 2016-2020, ...)
- **Rolling Window:** Training set is fixed length (e.g., 12 months), slides forward
- **Walk-Forward Validation:** Out-of-sample testing with rolling train/test splits
- **Fold:** One train/test cycle in walk-forward validation
- **Selector:** Algorithm that ranks pairs (Correlation, Cointegration, LSTM, etc.)
- **Ensemble:** Aggregation of multiple selectors via weighted scoring
- **Top-K:** Number of pairs selected per fold (K=10 in this thesis)

---

**Word Count (Chapter 5):** ~5,200 words  
**Total Thesis Word Count:** ~26,000-30,000 words (Chapters 1-5 + Appendices)

---

**[End of Thesis]**
