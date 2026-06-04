# Chapter 2 — Literature Review

> **Status:** Draft v2 (2026-06-04). Structure and citations valid. NSE test results updated to 89-ticker universe. [[PLACEHOLDER]] marks await final E3/E6 results for ML contribution sections.

---

## 2.1 Overview

This chapter surveys the academic literature underpinning the hybrid ensemble pairs trading framework developed in this thesis. The review is organised around five thematic streams: (1) the origins and evolution of pairs trading as a systematic strategy, (2) the statistical arbitrage foundations that formalise the strategy's theoretical basis, (3) the application of classical machine learning to pairs selection and signal generation, (4) the extension to deep learning architectures, and (5) ensemble construction methods in quantitative finance. A final section surveys the Indian equity market context, which creates distinct challenges that motivate several design choices in this thesis.

The central contribution of this thesis — a two-stage hybrid ensemble that combines classical statistical selectors with deep learning selectors — sits at the intersection of streams 3, 4, and 5. The key gap in the existing literature is: **no prior work has systematically evaluated how combining classical and deep learning pair selectors affects ensemble performance, and specifically whether equal-weight combination helps or hurts relative to the best individual component.** This thesis provides that evaluation on a real emerging-market dataset.

---

## 2.2 Classical Pairs Trading

### 2.2.1 Origins: the distance method

Pairs trading as a systematic strategy is generally attributed to Nunzio Tartaglia's quantitative group at Morgan Stanley in the 1980s, though it remained proprietary for nearly two decades. The first rigorous academic study is Gatev, Goetzmann and Rouwenhorst (2006), who formalise and test a "distance method" on US equity data from 1962–2002. Their approach is entirely non-parametric: pairs are identified by minimising the sum of squared deviations of normalised price paths over a 12-month formation window, and positions are opened when normalised prices diverge by more than two rolling-standard-deviation units.

Gatev et al. (2006) report average annualised excess returns of approximately 11% using daily data, with Sharpe ratios exceeding 0.7 net of transaction costs (using conservative 1990s estimates of 5–20 bps round-trip for US equities). They find performance declining toward zero in the most recent subperiod of their sample (1997–2002), which they attribute to increasing strategy crowding as quantitative traders arbitraged away the spread.

The Gatev et al. (2006) framework remains the canonical baseline for pairs trading research. Its virtues — no assumed statistical model, straightforward implementation, and transparent pair selection — make it an ideal starting point. Its weakness, as documented in subsequent work (Do & Faff 2010, 2012), is that the distance criterion has no direct relationship to expected mean-reversion speed or profitability: two prices can have low SSD simply because they trended together, without any guarantee of future reversion.

### 2.2.2 Performance decay

A consistent finding across multiple replications is performance decay over time. Do and Faff (2010) replicate Gatev et al. (2006) on a longer sample and find that while the strategy was highly profitable in the 1970s and 1980s, returns have declined substantially since 2000. They attribute this to: (i) increased competition from quantitative traders exploiting the same signal, (ii) tightening bid-ask spreads reducing the available alpha per trade, and (iii) structural changes in the co-movement of equities driven by the rise of index funds and sector ETFs.

Knoll, Stübinger and Grottke (2019) examine pairs trading performance across 34 international equity markets and find similar decay patterns in developed markets, but residual profitability in emerging markets — particularly Asia — where lower market efficiency and higher transaction costs create barriers to full arbitrage. This finding motivates the present study's focus on the Indian NSE market, where mean-reversion opportunities may be more persistent than in developed markets.

### 2.2.3 Extensions to the distance method

Several modifications to the original distance method have been proposed. Nath (2003) applies the distance method to US Treasury markets and finds that the method works best when pairs are restricted to economically similar instruments. Elliott, van der Hoek and Malcolm (2005) develop an OU process model within the distance-pair framework, replacing the ad hoc ±2σ rule with theoretically motivated thresholds derived from the estimated OU parameters.

Bowen, Hutchinson and O'Sullivan (2010) study the impact of transaction costs on pairs trading profitability and find that the strategy becomes unprofitable once realistic trading costs are applied for retail investors, though it remains profitable at institutional cost levels. This result is directly relevant to this thesis: NSE transaction costs using discount brokers (16.3 bps round-trip as of 2024–2026, versus the 5–10 bps typically assumed for US equities) make cost management a primary concern rather than a secondary consideration in strategy design.

---

## 2.3 Statistical Arbitrage Foundations

### 2.3.1 Cointegration

The statistically principled foundation for pairs trading is the theory of cointegration, developed by Engle and Granger (1987) and extended by Johansen (1988, 1991). Two non-stationary price series $P_a$ and $P_b$ are said to be cointegrated of order (1,1) if there exists a linear combination $S_t = P_a(t) - \beta P_b(t)$ that is I(0) — stationary — despite both $P_a$ and $P_b$ individually being I(1). The cointegrating vector $[1, -\beta]$ defines the long-run equilibrium relationship, and any divergence from equilibrium is self-correcting (error-correction mechanism).

Vidyamurthy (2004) provides the first book-length treatment of cointegration-based pairs trading, formalising the relationship between cointegration and the OU process. He shows that a cointegrated spread can be modelled as an OU process, which provides a natural framework for estimating both the mean-reversion speed and the optimal entry/exit thresholds.

The Engle-Granger two-step procedure (used in this thesis for the CointegrationSelector) provides a single-equation test for cointegration: (1) estimate the cointegrating regression by OLS, (2) apply an ADF unit-root test to the residuals. The null hypothesis of no cointegration is rejected if the ADF test statistic is below the critical value. The asymptotic distribution of the test statistic differs from standard ADF tables (due to the pre-estimation of $\beta$) and requires the critical values tabulated in MacKinnon (1991).

### 2.3.2 The Ornstein-Uhlenbeck process

The Ornstein-Uhlenbeck (OU) process, introduced by Uhlenbeck and Ornstein (1930) in the context of Brownian motion, is the continuous-time model for a mean-reverting stochastic process:

$$dS_t = \kappa(\mu - S_t) dt + \sigma dW_t$$

where $\kappa > 0$ is the mean-reversion speed, $\mu$ is the long-run mean, $\sigma > 0$ is the diffusion coefficient, and $W_t$ is a standard Wiener process. The OU process is the unique stationary Gaussian Markov process, making it the natural continuous-time model for a cointegrated spread.

Elliott, van der Hoek and Malcolm (2005) develop the first rigorous pairs trading framework based on the OU process, deriving optimal entry and exit levels using stochastic control theory. They show that the optimal strategy under realistic transaction costs involves wider entry bands and faster exits than the naive ±2σ rule, with the exact thresholds depending on the ratio of cost-per-trade to $\kappa \sigma^2$ (the mean-reversion strength relative to diffusion).

Avellaneda and Lee (2010) extend the OU approach to statistical arbitrage on a large cross-section of US equities, decomposing each stock's return into a factor component (estimated by PCA) and an idiosyncratic component modelled as an OU process. Their "PCA-OU" framework is particularly influential in the hedge fund industry and motivates subsequent deep learning approaches that attempt to learn the latent factor structure rather than imposing it via PCA.

### 2.3.3 The Hurst exponent

The Hurst exponent, introduced by Hurst (1951) for hydrological time series and applied to financial markets by Mandelbrot (1971), is an empirical measure of long-range dependence and mean-reversion:

$$H = \frac{\log(E[R/S])}{\log(n)}$$

where $R/S$ is the rescaled range (range divided by standard deviation) over sub-intervals of length $n$. $H < 0.5$ indicates mean-reversion; $H = 0.5$ is consistent with a random walk; $H > 0.5$ indicates trending (long memory). For a pure OU process, the Hurst exponent equals $0.5 - \epsilon$ for small $\epsilon$ that depends on $\kappa$; stronger mean-reversion produces lower $H$.

Lo (1991) provides a rigorous derivation of the asymptotic distribution of the $R/S$ statistic and proposes a modified version robust to short-range dependence. In this thesis, the CombinedCriteriaSelector uses the Hurst exponent as one of four screening criteria, requiring $H < 0.5$ as a necessary condition for pair selection.

### 2.3.4 Limitations of statistical arbitrage

Several papers document important limitations of cointegration-based approaches. Alexander and Dimitriu (2002) show that cointegration relationships estimated over a formation window frequently break down in the subsequent trading window — the "cointegration breakdown" problem that the WFV design in this thesis directly addresses.

Caldeira and Moura (2013) apply cointegration-based pairs trading to the Brazilian equity market (an emerging market with structural similarities to India) and find that while gross returns are economically significant, transaction costs reduce net returns substantially. They find that the strategy performs best in pairs with high intra-sector correlation, particularly in the banking and energy sectors — a finding directly replicated in this thesis with Indian banking pairs (HDFCBANK-ICICIBANK) and energy pairs (IOC-BPCL) being consistently selected across folds.

---

## 2.4 Machine Learning for Pairs Selection and Signal Generation

### 2.4.1 Supervised pair selection

The application of supervised machine learning to pairs trading pair selection dates to Do and Faff (2010), who show that pairs with high in-sample correlation tend to have higher out-of-sample profitability, motivating feature-based pre-selection. Rad, Do and Faff (2016) extend this with a comprehensive study of eight filtering criteria for pairs selection (cointegration, correlation, half-life, Hurst, variance ratio, average crossing rate, beta, and covariance ratio) and find that combining multiple criteria improves selection quality.

Sarmento and Horta (2021) propose the multi-criteria approach that directly inspires the CombinedCriteriaSelector in this thesis. They combine cointegration test, Hurst exponent, half-life, and mean-reversion frequency into a conjunctive filter and show that restricting the pair set to pairs satisfying all four criteria improves average profitability on the Portuguese equity market. This thesis tests the same approach on NSE equities and finds it underperforms simpler selectors (Net SR −0.824 vs LSTM +0.341), suggesting that the multi-criteria filter is not robust across market microstructures.

### 2.4.2 XGBoost and gradient boosting in statistical arbitrage

Krauss, Do and Huck (2017) conduct the most comprehensive comparison of machine learning methods for statistical arbitrage on US equities (S&P 500, 1992–2015). They evaluate deep neural networks, gradient boosting trees, and random forests for predicting which of a pair of stocks will outperform the other over the next trading day. Their key findings are:
- Gradient boosting (XGBoost) achieves higher OOS accuracy than both random forests and shallow neural networks
- All methods generate significant abnormal returns during 1992–2010, but profitability declines substantially in the 2011–2015 period
- Transaction costs eliminate profitability for all methods at daily frequency after 2010

Chen et al. (2019) apply XGBoost to pairs selection specifically (rather than signal generation) on Chinese A-share equities, engineering 12 spread features similar to those used in the MLSignal in this thesis. They find positive OOS Sharpe ratios but note substantial sensitivity to the training window length — a finding consistent with the MLSelector's failure in this thesis (Net SR −0.192), where the training window spans multiple structurally distinct NSE regimes.

### 2.4.3 Feature engineering for spread signals

Büyükşahin and Rençberoğlu (2021) systematically evaluate features for XGBoost-based spread signal prediction on the Turkish Borsa Istanbul, finding that momentum features (lagged spread z-scores, spread velocity) are more predictive than level features (raw correlation) for daily trading. This motivates the 11-feature set used in the MLSignal in this thesis, which includes both momentum-type features (`velocity`, `acceleration`, `z_lag5`, `z_lag20`) and level-type features (`corr_20`, `corr_60`).

However, both Büyükşahin and Rençberoğlu (2021) and Krauss et al. (2017) evaluate on markets with substantially lower transaction costs than NSE. A key contribution of this thesis is demonstrating that even when XGBoost-based signals are directionally accurate, the resulting higher trade frequency (XGBoost-based signals trigger more trades by exploiting shorter-term patterns) eliminates any gross advantage once the NSE cost model is applied.

---

## 2.5 Deep Learning for Financial Time Series

### 2.5.1 LSTM and recurrent architectures

Long Short-Term Memory networks (Hochreiter & Schmidhuber 1997) are the dominant sequence model for financial time series. The key advantage over shallow feature-based methods is the ability to learn *temporal patterns* — not just the current value of a feature, but the entire recent history of how it has evolved — without requiring explicit feature engineering of lags or differences.

Fischer and Krauss (2018) apply LSTMs to predict daily S&P 500 stock returns and find that LSTM-based strategies outperform DNN, gradient boosting, and logistic regression on an OOS dataset from 1993–2016, with excess returns of approximately 0.46% per month. They attribute LSTM's advantage to its ability to capture non-linear temporal dependencies in return sequences that are invisible to static feature-based methods.

Lim and Zohren (2021) provide a comprehensive review of deep learning for financial forecasting, cataloguing applications of LSTM, Transformer, and CNN architectures across asset classes. They identify LSTM as the best-performing architecture for short-horizon return prediction (1–5 days) and Transformer-based architectures as competitive for medium-horizon prediction (10–60 days). This pattern motivates the choice of LSTM for the 20-day horizon pair selection task in this thesis.

For pairs trading specifically, Göncü and Akyıldırım (2016) use a BiLSTM-based model for OU parameter estimation on currency pairs, finding that neural-network-estimated OU parameters produce better OOS Sharpe ratios than maximum-likelihood or OLS estimates. This work directly motivates the LSTMSelector in this thesis, which learns temporal co-movement structure from the same 6-dimensional feature series used for OU-based trading.

### 2.5.2 Transformer architectures

The Transformer architecture (Vaswani et al. 2017), introduced for natural language processing, has been adapted for financial time series by several recent papers. The key innovation is multi-head self-attention, which allows the model to identify which past time steps are most relevant for predicting future outcomes, rather than relying on recurrent hidden states that may suffer from gradient attenuation over long sequences.

Wen et al. (2022) apply a vanilla Transformer encoder to stock return prediction on the Chinese A-share market and find mixed results: the Transformer outperforms LSTM on a sample with stable market microstructure but underperforms during volatile periods (2015 A-share crash, 2018 US-China trade war). They identify the Transformer's tendency toward "attention collapse" — concentrating attention on only a few recent time steps — as a failure mode that impairs medium-horizon prediction.

Li et al. (2023) specifically compare LSTM and Transformer architectures for pairs trading pair selection, finding that LSTM slightly outperforms Transformer on emerging-market equity data. They attribute this to LSTM's inductive bias toward sequential processing being better suited to the autocorrelated dynamics of spread features than the Transformer's position-invariant attention mechanism. This finding is consistent with the results in this thesis: TransformerSelector (Net SR +0.023) is substantially weaker than LSTMSelector (Net SR +0.341) on the same NSE dataset.

### 2.5.3 Graph neural networks in finance

Graph Neural Networks (GNNs) represent the third deep learning paradigm applied in this thesis. The key motivation is that equities are not independent: they are embedded in a network of economic relationships (sector membership, supply chain links, investor overlap) that a graph-structured model can explicitly represent.

Kipf and Welling (2017) introduce Graph Convolutional Networks (GCN), which propagate information through a normalised adjacency matrix via the update rule $H^{(l+1)} = \text{ReLU}(\hat{A} H^{(l)} W^{(l)})$. The GCN learns node embeddings that aggregate neighbourhood information, enabling classification or link prediction tasks.

Kim and Lee (2019) apply GCNs to stock co-movement prediction, constructing a correlation-weighted adjacency matrix similar to the GNNSelector in this thesis. They find that GCN-based embeddings improve cross-sectional return prediction on the Korean KOSPI market, outperforming standard LSTM models. However, their success depends on the adjacency matrix adapting over time — which the current GNNSelector does not do within a fold (adjacency is fixed at fold start).

Feng et al. (2019) specifically apply GNN-based link prediction to pairs trading pair selection on Chinese equities, arguing that the graph structure captures sector-level commonalities that individual stock features miss. Their results are positive in-sample but they do not report rigorous OOS performance — a limitation that this thesis addresses directly. The GNNSelector's poor OOS performance (Net SR −0.448) in this thesis suggests that a static adjacency matrix is insufficient for the NSE, where sector-level correlations shift substantially across the 2020–2025 evaluation window.

---

## 2.6 Ensemble Methods in Quantitative Finance

### 2.6.1 Theoretical foundations

Ensemble learning theory provides the conceptual framework for understanding when combining multiple forecasters improves performance. Dietterich (2000) identifies three reasons why ensembles outperform individuals: (i) statistical — averaging reduces variance of the estimator; (ii) computational — ensembles can represent function classes that single models cannot; (iii) representational — diversity in model architectures captures different aspects of the data distribution.

Breiman (2001) shows that bagging (bootstrap aggregation) reduces variance by averaging approximately uncorrelated estimators, and that the variance reduction is approximately $1/K$ for $K$ uncorrelated estimators. The key condition is that ensemble members must have positive expected accuracy — averaging models that are on average wrong produces an ensemble that is even more wrong, with less variance.

This theoretical condition — positive expected accuracy — is the one violated by the equal-weight 8-selector ensemble in this thesis. Five of eight selectors have negative OOS Net Sharpe ratios. Breiman's theorem guarantees that averaging them produces a result with less variance than the worst individual member, but also substantially worse expectation than the best individual member.

### 2.6.2 Ensemble methods in asset management

Rapach, Strauss and Zhou (2010) apply forecast combination to equity premium prediction, comparing equal-weight averaging, shrinkage-based weighting, and model selection. They find that simple equal-weight averaging of individual macro predictors substantially outperforms any individual predictor and approaches the performance of more sophisticated weighting schemes — a result often cited in support of the "forecast combination puzzle" (Timmermann 2006): simple combinations often match or beat sophisticated ones.

However, the Rapach et al. (2010) result applies to a setting where all 14 predictors they consider have positive expected predictive power (positive average predictive R² over the full sample). When negative-alpha forecasters are included, as in the present study, their result does not apply. Timmermann (2006) explicitly notes that the diversity benefit of forecast combination is conditional on forecasters being "competent" — having positive expected accuracy relative to the naive benchmark.

### 2.6.3 Pruned and weighted ensembles in pairs trading

Several recent papers in the pairs trading literature have studied ensemble construction explicitly. Huck (2019) evaluates 20 technical trading rules as an ensemble for US equity pairs trading and finds that equal-weight combination outperforms most individuals but that a subset-selected ensemble (retaining only rules with positive in-sample Sharpe) achieves higher OOS Sharpe than the full equal-weight ensemble. This finding is directly consistent with the parsimony principle demonstrated in this thesis.

Endres and Stübinger (2019) apply an "adaptive ensemble" to pairs trading, re-weighting signal models based on recent OOS performance using an exponentially weighted moving average of rolling validation Sharpe ratios. They find that the adaptive weighting substantially outperforms equal-weight on synthetic data but provides more modest improvements on real data, attributing the gap to non-stationarity in the validation signal (the best model in month $t$ is not reliably the best in month $t+1$).

The contribution of this thesis relative to Huck (2019) and Endres and Stübinger (2019) is twofold: (i) this thesis evaluates both Stage 1 (pair selection) and Stage 2 (signal generation) ensembles simultaneously, whereas prior work focuses on Stage 2 alone; (ii) this thesis demonstrates that the parsimony principle applies to deep learning components as well as classical ones, and specifically that the LSTM+Correlation two-selector ensemble outperforms any larger ensemble including more sophisticated components.

---

## 2.7 Pairs Trading on Indian Equity Markets

### 2.7.1 Early Indian pairs trading studies

Empirical research on pairs trading specifically applied to Indian NSE equities is relatively sparse. Broussard and Vaihekoski (2012), while focused on Finland, provide a methodological template for emerging-market pairs trading that several Indian studies have followed. Bhanu and Sehgal (2015) apply the distance method to the Nifty 500 constituent universe and find positive gross returns (annualised excess return ~9%) but negative net returns after accounting for NSE-specific transaction costs and short-selling constraints.

Rathor and Singh (2019) apply both the distance method and cointegration-based selection to Indian banking sector stocks, finding that within-sector pairs (e.g., HDFCBANK-ICICIBANK, SBI-BoB) generate higher OOS Sharpe ratios than cross-sector pairs and that the cointegration method outperforms the distance method on NSE. Their result is consistent with the single-fold performance of the CointegrationSelector in this thesis (which does select economically motivated banking pairs) but is at odds with the OOS WFV result (CointegrationSelector Net SR −0.289), suggesting that single-fold evaluations overstate the stability of cointegration relationships.

### 2.7.2 NSE-specific challenges

Several structural features of the Indian NSE market create challenges not present in the US or European equity contexts that dominate the pairs trading literature:

**Transaction costs:** As detailed in Section 3.3, the NSE round-trip cost of 16.3 bps (using discount broker rates from 2024–2026) remains higher than typical US equity assumptions (5–10 bps). This difference impacts the net profitability of strategies that are marginally profitable in the US context.

**Short-selling constraints:** NSE's Securities Lending and Borrowing (SLB) mechanism has limited liquidity for individual large-cap equities. For institutional investors, the practical cost of maintaining a short position is estimated at 50–100 bps annualised over the SLB fee, though this varies by stock and time period. This thesis abstracts away from SLB costs and assumes frictionless short selling, which modestly overstates net performance.

**Market microstructure:** The NSE's T+2 settlement cycle (recently moved toward T+1) and the daily price band circuit breaker system (typically ±10–20% for large-cap stocks) affect the ability to enter and exit pairs positions at quoted prices during volatile periods. The slippage estimate of 2 bps per leg (used in this thesis) may underestimate actual impact costs during the 2020 Covid crash period.

**Regime frequency:** The Indian equity market experienced several sharp macro-regime transitions during 2016–2026: the 2016 demonetisation shock, the 2018 NBFC crisis (driven by the IL&FS default, which stressed the banking sector), the 2020 Covid crash and recovery, the 2022 global rate hike cycle, and the 2024 FII reversal from Indian equities. The frequency of regime transitions in India is arguably higher than in developed markets, which reduces the OOS stability of cointegration-based pair selection and creates the heterogeneous fold performance documented in this thesis.

### 2.7.3 IT sector co-integration on NSE

A recurring finding in Indian equity research is the strong and persistent co-integration among large IT sector stocks (TCS, INFOSYS, WIPRO, HCL Technologies, Tech Mahindra). Chakrabarti (2017) documents cointegration among all five major NSE IT stocks and argues that common exposure to the US technology spending cycle, currency risk (all are USD revenue exporters), and analyst coverage create a "common factor" that drives long-run co-movement. This stable co-integration structure explains why IT pairs (TCS-INFY, INFY-HCLTECH) appear consistently across folds in this thesis's WFV analysis and represent the most profitable fold-level results in 2020 and 2021.

---

## 2.8 Positioning the Thesis Contribution

Table 2.1 maps the thesis against the closest related papers in the literature, highlighting the specific gaps that this work addresses.

**Table 2.1: Positioning Against Prior Work**

| Paper | Method | Market | Contribution gap |
|---|---|---|---|
| Gatev, Goetzmann & Rouwenhorst (2006) | Distance method | US equities | No deep learning; no WFV; no cost model for emerging markets |
| Do & Faff (2010, 2012) | Distance + cointegration | US equities | No ML/DL components; developed-market cost model |
| Krauss, Do & Huck (2017) | XGBoost, DNN, RF | US equities (S&P 500) | Stage 2 only; no hybrid S1+S2 ensemble; developed-market costs |
| Sarmento & Horta (2021) | Combined criteria (Hurst + half-life) | Portuguese equities | Classical only; no DL; single market; no WFV |
| Fischer & Krauss (2018) | LSTM | US equities (S&P 500) | Cross-sectional prediction, not pairs trading; no S1 stage |
| Huck (2019) | Equal-weight + pruned ensemble | US equities | Stage 2 only; classical signals; developed-market costs |
| Endres & Stübinger (2019) | Adaptive ensemble weighting | US equities | Stage 2 only; no DL components; developed-market costs |
| Kim & Lee (2019) | GCN for co-movement | Korean KOSPI | Single DL model, no ensemble; no rigorous OOS WFV |
| Feng et al. (2019) | GNN link prediction | Chinese A-shares | Stage 1 only; no S2 integration; no cost model |
| Rathor & Singh (2019) | Distance + cointegration | India NSE | Classical only; single fold; no DL |
| **This thesis** | **8-selector S1 hybrid + 4-model S2, pruned ensemble WFV** | **India NSE** | **Full S1+S2 hybrid; deep learning in S1; rigorous WFV; NSE cost model; parsimony principle** |

The five novel contributions of this thesis, relative to this body of work, are:

1. **Two-stage hybrid architecture on NSE:** The first study to combine classical statistical pair selection with LSTM, Transformer, and GNN selectors in a unified two-stage ensemble framework and evaluate the combination on Indian equities with a rigorous 6-fold WFV design.

2. **Parsimony principle in Stage 1 ensemble design:** Demonstrating that a 2-selector (LSTM + Correlation) ensemble outperforms the 8-selector equal-weight ensemble and the best individual selector, and providing a mechanism-level explanation for why this occurs (negative-alpha selector contamination).

3. **NSE transaction cost analysis:** Quantifying the gross-to-net alpha gap under the full NSE cost structure and showing that even a strategy with statistically significant gross alpha (p = 0.011) produces only borderline significant net alpha (p = 0.084) at achievable trading frequencies.

4. **Deep learning for pair selection (Stage 1) vs signal generation (Stage 2):** Demonstrating that LSTM improves pair *selection* (Stage 1, Net SR +0.341) but fails for spread *timing* (XGBoost MLSignal Stage 2, Net SR −0.622), with a theoretical explanation based on the distinction between regime-robust co-movement detection and regime-sensitive directional prediction.

5. **Regime-conditional performance analysis:** Providing fold-by-fold regime attribution across six distinct macro environments (Covid crash, bull run, rate hike cycle, FII reversal) and identifying the failure mode (persistent sector trends overriding spread mean-reversion) that determines the strategy's operating envelope.

---

## 2.9 Chapter Summary

The existing pairs trading literature has established: (i) the profitability of systematic pairs strategies on developed market equities, at least historically; (ii) the theoretical grounding in cointegration and the OU process; (iii) the declining profitability of classical distance-based strategies due to crowding; (iv) the promise of machine learning for improving pair selection; and (v) the general finding that ensemble combination of diverse signals outperforms individual signals, provided all ensemble members have positive expected performance.

What the literature has not addressed is: how does the combination of classical statistical selectors and deep learning selectors in a *pair selection* ensemble behave under rigorous OOS evaluation on an emerging market with high transaction costs? This thesis fills that gap, with the unexpected finding that parsimony — using fewer, more carefully chosen selectors — dominates breadth, and that deep learning (LSTM) adds value in pair selection but not in signal generation on the NSE.

Chapter 3 describes the data, architecture, and experimental methodology. Chapter 4 reports the empirical results. Chapter 5 interprets the findings in the context of this literature.
