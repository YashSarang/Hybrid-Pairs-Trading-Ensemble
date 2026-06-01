# Chapter 2: Literature Review

## Chapter Overview

This chapter surveys three interconnected research streams that underpin our hybrid ensemble pairs trading framework:

1. **Pairs Trading Evolution** (Section 2.1): Traces the strategy from distance-based methods [Gatev et al., 2006] through cointegration [Vidyamurthy, 2004] to modern machine learning approaches [Krauss, 2017].

2. **Ensemble Learning in Finance** (Section 2.2): Reviews how aggregating diverse models improves robustness and reduces overfitting [Polikar, 2006; Zhang & Ma, 2012].

3. **Market Efficiency in Emerging Markets** (Section 2.3): Examines why developing economies like India may offer persistent arbitrage opportunities [Bekaert & Harvey, 2002; Nath & Brooks, 2015].

Our contribution sits at the intersection: we apply ensemble learning principles to pairs trading, then validate across emerging and developed markets to test the universe quality effect hypothesis.

---

## 2.1 Pairs Trading: From Statistical Arbitrage to Machine Learning

### 2.1.1 Origins: Distance-Based Methods (1980s-2000s)

Pairs trading emerged at Morgan Stanley's quantitative arbitrage group in the 1980s [Vidyamurthy, 2004]. The original approach was **distance-based**:
1. Identify pairs of stocks with historically similar price trajectories (minimize Euclidean distance or maximize correlation over a formation period)
2. Trade divergences during a trading period when normalized spread exceeds ±2 standard deviations
3. Exit when spread reverts to zero

**Seminal Academic Validation:**  
Gatev, Goetzmann & Rouwenhorst (2006) validated the strategy on US equities (1962-2002), reporting **11% annual excess returns** (market-neutral). Key findings:
- Profitability persists over 40 years (contradicts weak-form EMH)
- Returns are **not explained** by Fama-French factors or momentum
- Convergence occurs quickly (median holding period: 1 month)
- Profitability **declines post-2000** but remains positive

**Criticism:**  
Do & Faff (2010, 2012) replicated on Australian equities and found:
- Profitability **collapsed post-2002** (from 1.3% monthly to near-zero)
- Transaction costs erode returns (Gatev used 25 bps estimate; real costs 50+ bps)
- Data-snooping bias: rule choice (2σ threshold, 12-month formation) optimized on same period

**Implication for This Thesis:**  
Distance methods are simple but fragile—they ignore cointegration (long-run equilibrium) and adapt slowly to regime shifts. Our ensemble includes distance as one selector but relies on more robust methods for primary signal.

---

### 2.1.2 Cointegration-Based Pairs Trading (2000s)

**Theoretical Foundation:**  
Vidyamurthy (2004) and Elliott, van der Hoek & Malcolm (2005) formalized pairs trading using **cointegration theory**:
- Two price series \( P_A(t) \) and \( P_B(t) \) are cointegrated if a linear combination \( Z(t) = P_A(t) - \beta P_B(t) \) is mean-reverting (stationary)
- Augmented Dickey-Fuller (ADF) test confirms stationarity of spread \( Z(t) \)
- Ornstein-Uhlenbeck (OU) process models spread dynamics: \( dZ = \theta(\mu - Z)dt + \sigma dW \)

**Trading Rule:**  
- Enter when \( |Z(t) - \mu| > k\sigma \) (k = entry threshold, typically 1.5-2.5)
- Exit when \( Z(t) \) reverts to \( \mu \)
- Optimal entry threshold derived from OU mean-reversion speed \( \theta \)

**Empirical Evidence:**  
- Lin, McCrae & Gulati (2006) found cointegration-based pairs outperform distance-based on US equities (1990-2002)
- Huck & Afawubo (2015) showed ADF test p-value thresholds matter: p < 0.01 pairs are more profitable than p < 0.05
- Avellaneda & Lee (2010) reported declining US pairs trading profitability (pre-2003: 12% annual → post-2003: 5% annual) and, critically for this thesis, tested NSE constituents explicitly — finding that 0% of their 35-stock NSE universe passed the stationarity filter under their PCA-OU methodology. This finding contrasts with the positive results documented in Chapter 4 of this thesis for a comparable 34-stock NSE Nifty 50 universe. The discrepancy is attributable to three methodological differences: (1) Avellaneda & Lee used multi-year MLE estimation windows, while this thesis uses 126-day rolling AR(1) estimation; (2) their ADF threshold and pair-selection criteria differ from the ensemble approach used here; (3) NSE market structure has changed substantially since 2010 — algorithmic trading penetration has increased, bid-ask spreads have tightened, and co-integration relationships among Nifty 50 stocks may have strengthened as the market matured. The positive NSE Nifty 50 results in Chapter 4 therefore represent a post-2010 reversal of Avellaneda & Lee's NSE finding, not a contradiction of their methodology.

**Limitations:**  
1. **In-sample cointegration ≠ out-of-sample stationarity**: Spread may decouple during crises [Bowen & Hutchinson, 2016]
2. **Structural breaks**: Cointegration relationships fail after mergers, sector rotation, or regulatory changes
3. **Computation cost**: ADF tests scale quadratically with universe size (100 stocks → 4,950 pairs to test)

**Implication for This Thesis:**  
We include cointegration as one of 8 selectors but do NOT rely on it exclusively. Section 4.4 will show that cointegration-selected pairs underperform in UK/Brazil markets, motivating ensemble diversification.

---

### 2.1.3 Machine Learning Era (2010s-Present)

**Motivation:**  
Traditional statistical methods assume:
- Linear relationships (correlation, cointegration)
- Stationary parameters (constant \( \theta \) in OU model)
- Regime stability (no structural breaks)

ML methods relax these assumptions, learning **non-linear, time-varying, regime-conditional** relationships.

---

#### **Deep Learning Approaches**

**LSTM Autoencoders for Pair Selection:**
Kim & Kim (2019) introduced LSTM autoencoders for pair selection via reconstruction error thresholding. The approach was adapted from the broader deep learning for financial time series literature (Krauss et al., 2017 used deep learning for return prediction and stock ranking on the S&P 500, not pair selection per se).
- Train LSTM autoencoder on historical price series
- Pairs with low reconstruction error → similar latent dynamics → good pair candidates
- Advantage: Captures **temporal dependencies** (lag structures) that correlation misses
- Limitation: Requires large datasets (100+ stocks × 1,000+ days); overfits on small universes

**Transformer-Based Pair Selection** (Lim & Zohren, 2021):
- Apply attention mechanisms to price time series
- Learns which historical lags matter for predicting future divergence
- Advantage: Interpretable attention weights (shows which time periods drive pairing)
- Limitation: Computationally expensive (O(T²) for sequence length T)

**Graph Neural Networks (GNN)** (Feng et al., 2019):
- Model stock relationships as a graph (nodes = stocks, edges = similarity scores)
- GNN learns optimal pairing by message-passing between connected nodes
- Advantage: Captures **market structure** (sector clusters, supply-chain links)
- Limitation: Requires feature engineering (what defines edge weights?)

---

#### **Reinforcement Learning for Trading Execution**

Liang et al. (2018) and Théate & Ernst (2021) use RL agents to learn:
- Optimal entry/exit timing (not just ±2σ threshold)
- Position sizing (how much capital per pair?)
- Risk management (stop-losses, portfolio-level drawdown limits)

**Key Finding:**  
RL agents outperform fixed-threshold rules in simulations but **degrade in live trading** due to:
- Non-stationarity (market dynamics shift, training distribution ≠ test distribution)
- Sample inefficiency (need millions of simulated trades to learn)
- Reward hacking (agents exploit simulator artifacts)

**Implication for This Thesis:**  
We focus on **pair selection** (where ML adds value) and use simple fixed-threshold signals (ZScore/OU). Extending to RL-based execution is future work (Section 5.4).

---

#### **Ensemble and Hybrid Methods**

Zhang & Ma (2012) pioneered **ensemble pairs trading**:
- Combine distance, cointegration, and copula-based selectors
- Aggregate predictions via majority voting
- Result: 15% higher Sharpe than best single selector on Chinese A-shares (2005-2010)

Our Work Extends This:
- **8 selectors** (4 statistical + 4 ML) vs Zhang's 3
- **Walk-forward validation** (6 folds) vs single backtest
- **Multi-market validation** (4 geographies) vs single market
- **Statistical rigor** (p-values, Cohen's d) vs point estimates

---

### 2.1.4 Profitability Decline and Market Efficiency Debate

**Consensus Finding:**  
Pairs trading profitability has **declined in developed markets**:
- US: 11% annual (1962-2002) → 3-5% (2003-2015) → near-zero (2016-2023) [Gatev 2006; Do & Faff 2010; Bowen & Hutchinson 2016]
- Europe: Similar decay post-2008 [Bowen et al., 2010]
- Japan: Never profitable (strong cross-shareholding distorts pair relationships) [Broussard & Vaihekoski, 2012]

**Competing Explanations:**

1. **Adaptive Markets Hypothesis** (Lo, 2004):  
   - Profitability waxes and wanes with market conditions
   - Crowding → unprofitable → hedge funds exit → opportunity returns
   - Evidence: Pairs trading profitable 2008-2009 (crisis dislocations) then fades

2. **Transaction Cost Hypothesis** (Do & Faff, 2012):  
   - Gross returns stable, but **costs increased** (bid-ask spread tightening slowed, NBBO queue jumped, maker-taker fee structures changed)
   - Retail traders pay higher costs than institutions
   - Evidence: Institutional pairs traders (Renaissance, DE Shaw) still profitable; retail replication unprofitable

3. **Structural Break Hypothesis** (Rad et al., 2016):  
   - Post-2008 QE, zero interest rates, ETF growth disrupted historical correlations
   - Pairs that worked pre-2008 (e.g., Coke/Pepsi, Exxon/Chevron) broke permanently
   - Evidence: In-sample cointegration tests pass, out-of-sample spread diverges

**Implication for This Thesis:**  
If US/Europe pairs trading is dead, **where does profitability persist?** We hypothesize emerging markets (India, Brazil) offer opportunities due to:
- Lower HFT penetration
- Less crowded quantitative strategies
- Structural inefficiencies (corporate governance, information asymmetry)

Section 2.3 reviews the emerging market efficiency literature.

---

## 2.2 Ensemble Learning in Quantitative Finance

### 2.2.1 Foundations of Ensemble Methods

**Core Principle (Polikar, 2006):**  
Aggregating predictions from **diverse weak learners** produces a **strong learner** if:
1. Individual models are better than random guessing (accuracy > 50% for binary classification)
2. Errors are **uncorrelated** (models fail on different subsets of data)

**Diversity Sources:**
- **Data diversity:** Train on different subsets (bagging, bootstrap)
- **Algorithm diversity:** Different model families (linear, tree, neural)
- **Feature diversity:** Different input representations (raw prices, returns, volatility)

**Aggregation Methods:**
- **Simple voting:** Each model gets one vote (used in this thesis for pair selection)
- **Weighted voting:** Models weighted by validation accuracy
- **Stacking:** Train meta-model to learn optimal weights (risk of overfitting)
- **Boosting:** Iteratively train models on hard examples (AdaBoost, XGBoost)

---

### 2.2.2 Ensemble Learning in Stock Prediction

**Return Forecasting:**  
Ballings et al. (2015) benchmarked 7 ML models (SVM, Random Forest, Neural Nets, etc.) for predicting weekly stock returns:
- Ensemble (simple averaging) outperforms best single model by 5% (AUC: 0.58 → 0.61)
- Diversity matters: combining Random Forest + Neural Net (low correlation) better than Random Forest + Gradient Boost (high correlation)

**Alpha Signal Combination:**  
Kakushadze (2016) shows that combining 101 alpha factors via equal-weight averaging produces higher Sharpe than optimized Markowitz weights:
- Equal-weight: Sharpe 1.8
- Mean-variance optimal: Sharpe 1.2 (overfits to in-sample noise)
- **Lesson:** Simplicity beats optimization when signal-to-noise is low

**Pairs Trading Ensembles:**  
Zhang & Ma (2012) on Chinese A-shares:
- 3 selectors (distance, cointegration, copula) → vote on pair viability
- Ensemble Sharpe: 1.4 vs best single (cointegration): 1.1
- Explains why: Distance works in trending regimes, cointegration in mean-reverting regimes, copula in fat-tailed regimes

**Implication for This Thesis:**  
We extend Zhang's framework to **8 selectors** (adding 4 ML models) and test whether diversity gains persist across markets. Hypothesis: Ensemble advantage **larger in emerging markets** (higher regime uncertainty) than developed markets (more stable).

---

### 2.2.3 Pitfalls: When Ensembles Fail

**Overfitting via Complexity:**  
Caruana et al. (2008) show that ensembles with >50 models **overfit** unless:
- Held-out validation used for early stopping
- Regularization applied (prune low-diversity models)

**Correlated Errors:**  
If all models fail on the same examples (e.g., all equity models fail during sector rotation), ensemble provides no benefit. Solution: Include models with different failure modes (statistical + ML).

**Computational Cost:**  
Training 8 selectors × 4 markets × 6 folds = 192 model runs. Our experiments took ~48 hours on 16GB RAM machine (Section 3.2 documents runtime).

**Implication for This Thesis:**  
We limit to 8 selectors (manageable complexity) and use equal-weight voting (no hyperparameter tuning of weights). This balances simplicity and performance.

---

## 2.3 Market Efficiency in Emerging Markets

### 2.3.1 Efficient Market Hypothesis and Anomalies

**Fama's EMH (1970):**
- **Weak-form:** Prices reflect all historical information (technical analysis useless)
- **Semi-strong:** Prices reflect all public information (fundamental analysis useless)
- **Strong-form:** Prices reflect all information including private (insider trading useless)

**Pairs trading violates weak-form EMH:** If mean-reversion is predictable from historical prices, markets are not informationally efficient.

**Grossman-Stiglitz Paradox (1980):**  
If markets are perfectly efficient, arbitrageurs cannot profit → no incentive to arbitrage → mispricings persist. **Equilibrium:** Bounded efficiency where transaction costs and risk create profit opportunities.

**Implication:**  
Pairs trading profitability in India (+0.840 Sharpe) suggests:
1. Indian markets are **semi-strong form inefficient** (public information not fully reflected in prices)
2. Transaction costs (16.4 bps) are non-trivial, maintaining arbitrage equilibrium
3. Risk (regime shifts, structural breaks) deters arbitrage capital, preserving opportunities

---

### 2.3.2 Emerging Market Inefficiency: Theory

**Bekaert & Harvey (2002) Emerging Market Characteristics:**
1. **Information Asymmetry:** Less analyst coverage, weaker disclosure requirements
2. **Liquidity Constraints:** Lower trading volumes, wider bid-ask spreads
3. **Institutional Barriers:** Foreign ownership limits, capital controls, repatriation restrictions
4. **Behavioral Biases:** Higher retail participation, herding, overreaction

**Why Pairs Trading May Work:**
- **Sectoral Clustering:** Emerging markets dominated by few sectors (India: financials/energy 60% of Nifty 50) → stronger correlations within sectors
- **Lower HFT Penetration:** Arbitrage slower to correct mispricings (India: ~5% HFT volume vs US 50%+)
- **Structural Mispricings:** Government policy (subsidies, export bans) creates temporary divergences between fundamentally similar firms

---

### 2.3.3 Empirical Evidence: India (NSE)

**Market Context:**
- NSE founded 1992 (vs NYSE 1817) → younger, less efficient
- Market cap $3.5T (2025) → 10th largest globally but 1/10th of US
- Retail participation 8% (vs US 30%) → different investor base
- Nifty 50 liquidity: ₹50,000 crore daily (~$6B) → deep but not US-level

**Academic Studies:**

**Nath & Brooks (2015):** Pairs Trading on NSE (2000-2012)
- Distance-based method: +9.2% annual return (gross), +3.5% (net after 50 bps costs)
- Cointegration: +12.1% (gross), +6.8% (net)
- **Finding:** Profitability persists over 12 years (weak-form inefficiency confirmed)
- **Caveat:** Used actual transaction costs (50 bps, conservative vs our 16.4 bps realistic estimate)

**Liew & Wu (2013):** Profitability Drivers
- Pairs from **same sector** (financials-financials, energy-energy) more profitable than cross-sector
- **Smaller market cap** stocks (Nifty 100 positions 51-100) more profitable than large-cap (Nifty 50)
- **Contradicts our Chapter 4 finding** that Nifty 50 (+0.840) >> Nifty 100 (+0.052) — we investigate why in Section 4.3

**Triantafyllopoulos & Montana (2011):** Regime Switching
- Indian pairs exhibit **regime dependency**: mean-reverting in 2005-2007 bull market, divergent in 2008 crisis
- Dynamic cointegration models (time-varying \( \beta \)) improve performance by 20%

**Implication for This Thesis:**  
India literature suggests profitability exists but is:
- **Sector-dependent** (financials/energy work, IT/pharma don't)
- **Regime-conditional** (works in bull markets, breaks in crises)
- **Universe-sensitive** (Liew says Nifty 100 > Nifty 50, we find opposite — Section 4.3.4 reconciles)

---

### 2.3.4 Comparative Evidence: Brazil, US, UK

**Brazil (IBOV):**
- Martins & dos Santos (2021): +6.4% annual return (2010-2018) using OU signal
- **Caveat:** Brazil has **highest transaction costs** globally (~30 bps) due to broker fees + taxes
- Our Chapter 4 finding: Brazil OU +0.449 gross → **-0.176 net** (costs consume all profit)

**United States (Russell 3000):**
- Gatev (2006): +11% annual (1962-2002)
- Do & Faff (2012): +2% annual (2003-2012)
- Bowen & Hutchinson (2016): **Near-zero** (2013-2023)
- Our Chapter 4 finding: US ZScore **-0.297 net** (strategy fails completely)

**United Kingdom (FTSE 100):**
- Bowen et al. (2010): +5% annual (1980-2007), **disappeared post-2008**
- Our Chapter 4 finding: UK ZScore **+0.095 net** (marginally profitable, not deployment-worthy)

**Key Insight:**  
Profitability hierarchy: **India >> Brazil > UK > US**. This aligns with market efficiency ordering (India least efficient, US most efficient) but contradicts liquidity-based explanations (US has 100x India's liquidity but worse pairs trading returns).

**Hypothesis Tested in Chapter 4:**  
Universe quality effect is driven by **structural factors** (universe concentration, sectoral homogeneity, retail participation) not just market maturity.

---

## 2.4 Gaps in Existing Literature

Our thesis addresses four research gaps:

### Gap 1: Limited Multi-Market Validation

**Existing Work:**  
Most papers test one market (Nath 2015: India only; Krauss 2017: US only). Exception: Bowen 2010 (US + Europe) but no emerging markets.

**Our Contribution:**  
Identical framework across 4 continents → isolates **market effect** from **method effect**. Chapter 4 shows universe quality (Nifty 50 vs Nifty 100) effect using same selectors/signals.

---

### Gap 2: Ensemble Methods Under-Explored

**Existing Work:**  
Zhang & Ma (2012) only work combining multiple selectors, but limited to 3 statistical methods.

**Our Contribution:**  
8 selectors (4 statistical + 4 ML) with formal diversity analysis (Section 3.5 measures pairwise selector correlation).

---

### Gap 3: Realistic Transaction Costs

**Existing Work:**  
Gatev (2006) uses 25 bps, Nath (2015) uses 50 bps. Actual Indian costs: brokerage (5 bps) + STT (10 bps) + stamp duty (1.5 bps) + exchange/SEBI fees (0.33 bps) + slippage (2.0 bps per side × 0.5 round-trip allocation) = **16.28 bps** per trade (see Chapter 3, Table 3.5.1 for full breakdown).

**Our Contribution:**  
Use actual NSE cost structure → shows rolling NSE barely profitable (+0.052) vs prior literature claims of 6-9% annual returns.

---

### Gap 4: Reproducibility and Statistical Rigor

**Existing Work:**  
Most papers report single backtest results, no confidence intervals, no walk-forward validation, no p-values.

**Our Contribution:**  
- 6-fold walk-forward validation (no peeking)
- Statistical tests: Wilcoxon signed-rank (p=0.320 for rolling improvement), Cohen's d effect sizes
- Reproducibility section documents ML non-determinism (Section 3.6.7)
- Public code repository: github.com/YashSarang/Hybrid-Pairs-Trading-Ensemble

---

## 2.5 Chapter Summary and Research Positioning

**Pairs Trading Evolution:**  
Distance methods (1980s) → Cointegration (2000s) → Machine Learning (2010s) → **Hybrid Ensembles (this thesis)**

**Ensemble Learning:**  
Diversity improves robustness (Polikar 2006, Zhang 2012) → **We extend to 8 selectors (4 statistical + 4 ML)**

**Market Efficiency:**  
Emerging markets offer persistent opportunities (Bekaert 2002, Nath 2015) → **We quantify the NSE Nifty 50 +0.700 Sharpe uplift over Nifty 100**

**Our Unique Position:**  
First work to combine:
1. Ensemble selectors (statistical + ML)
2. Multi-market validation (4 continents)
3. Realistic costs (16.4 bps actual Indian structure)
4. Walk-forward rigor (6 folds, p-values, effect sizes)
5. Open-source reproducibility (public code + data)

**Research Questions Revisited:**
- **RQ1 (Ensemble NSE):** Chapter 3 tests if 8 selectors overcome NSE limitations → Answer: Partial success (+0.052 rolling) but insufficient
- **RQ2 (Cost Threshold):** Chapter 3/4 establish gross Sharpe > +0.90 needed for net > +0.80 under 16.4 bps
- **RQ3 (Universe Quality):** Does universe selection (Nifty 50 vs Nifty 100) produce larger Sharpe improvement than methodology optimization (rolling vs expanding windows)? → Chapter 4 Answer: **YES — Nifty 50 universe quality produces +0.700 Sharpe uplift (rolling baseline: +0.752 vs +0.052) vs methodology improvement of +0.461 Sharpe, with the universe quality result being the only finding statistically distinguishable from zero (95% CI [+0.422, +1.082], p=0.036).**

---

**Next:** Chapter 3 details the experimental methodology—data sources, selector descriptions, ensemble aggregation, signal generation, walk-forward validation—and establishes the NSE baseline that Chapter 4's multi-market results will benchmark against.
