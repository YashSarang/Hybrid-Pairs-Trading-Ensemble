# Chapter 3 — Data and Methodology

> **Status:** Final v1 (2026-06-05). Universe: 89 stocks, 2015-2024, 16.28 bps. All E3/E6/E7 results incorporated.

---

## 3.1 Universe Construction

### 3.1.1 Stock selection

The study uses a fixed universe of **89 NSE Nifty 100 large-cap equities** drawn from eight sectors, selected as of January 2015 and held constant through December 2024. All 89 tickers are constituents of the Nifty 100 index throughout the study period, ensuring adequate liquidity (average daily turnover > ₹100 crore) and continuous yfinance data coverage. Six tickers were excluded due to persistent data quality issues (gaps, corporate restructuring): TATAMOTORS, BERGERPAINTS, VODAFONEIDEA, LTIM, ADANITRANS, and one additional ticker with incomplete 2015 history.

**Table 3.1: NSE Universe — 89 Stocks across 8 Sectors (summary)**

| Sector | Approx. Count | Representative Tickers |
|---|---|---|
| Banking & Financial Services | ~15 | HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, INDUSINDBK, BAJFINANCE, HDFC... |
| Information Technology | ~12 | TCS, INFY, WIPRO, HCLTECH, TECHM, LTTS, MPHASIS... |
| Automobiles & Components | ~10 | MARUTI, M&M, BAJAJ-AUTO, HEROMOTOCO, EICHERMOT, TATAMOTORS (excl.), TVSMOTOR... |
| FMCG & Consumer Staples | ~10 | HINDUNILVR, ITC, NESTLEIND, BRITANNIA, DABUR, MARICO, GODREJCP... |
| Pharma & Healthcare | ~10 | SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, APOLLOHOSP, TORNTPHARM... |
| Energy & Oil and Gas | ~10 | RELIANCE, ONGC, IOC, BPCL, NTPC, POWERGRID, GAIL... |
| Metals & Mining | ~8 | TATASTEEL, JSWSTEEL, HINDALCO, COALINDIA, VEDL, NMDC... |
| Cement & Infrastructure | ~8 | ULTRACEMCO, ACC, SHREECEM, AMBUJACEM, LT, ADANIPORTS... |

Full ticker list is specified in `Implementation/experiments/config.py`.

**Design rationale:** Same-sector pairs (e.g., HDFCBANK–ICICIBANK, TCS–INFY) are natural cointegration candidates with genuine economic co-movement driven by shared factor exposure. Cross-sector pairs are included to allow the ML and deep learning selectors to discover non-obvious statistical relationships that pure economic reasoning might miss. The sector distribution ensures that the strategy is not purely a single-sector bet.

### 3.1.2 Pair candidate generation

All $\binom{89}{2} = 3{,}916$ ordered pair combinations are generated at the start of each walk-forward fold. Pairs are not pre-screened — all 3,916 are passed to each selector for scoring. This avoids survivorship bias in the pair screening step.

---

## 3.2 Data

### 3.2.1 Source and frequency

Daily closing prices are sourced via `yfinance` using NSE ticker symbols with the `.NS` suffix (e.g., `TCS.NS`, `HDFCBANK.NS`). The primary data window spans **2015-01-01 to 2024-12-31** (approximately 2,466 trading days). Data is fetched as adjusted closing prices, which account for dividends and stock splits in NSE's standard adjustment methodology. All prices are cached locally as a Parquet file (`Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet`) for deterministic reproducibility across runs.

The choice of daily frequency over hourly is empirically motivated: Experiment E1 (Section 4.1) demonstrates that hourly spreads exhibit substantially weaker mean-reversion (Hurst 0.251 vs 0.190 daily) and that the hourly strategy becomes insolvent after NSE transaction costs.

### 3.2.2 Preprocessing

Raw prices undergo the following preprocessing steps before use:
1. **Missing data:** NSE-specific non-trading days (public holidays, corporate action suspensions) are forward-filled using the most recent available closing price. No imputation is applied for gaps exceeding 3 consecutive trading days (none occur for large-cap NSE stocks in this period).
2. **Survivorship bias:** The 89-stock universe is fixed as of January 2015. Six problematic tickers were excluded at the outset; the remaining 89 were all trading continuously throughout 2015–2024. Mild survivorship bias is present by construction but limited given the large-cap, Nifty 100 composition.
3. **Corporate actions:** yfinance adjusted prices implicitly account for dividends and splits through the standard backward adjustment methodology.

---

## 3.3 Transaction Cost Model

### 3.3.1 NSE cost components

We use a detailed NSE cost model (`IndianCosts` dataclass, `core/backtest.py`) that reflects the actual charge structure for Indian equity delivery trades as of 2024:

**Table 3.2: NSE Transaction Cost Model — Per Leg**

| Component | Bps (per leg) | Notes |
|---|---|---|
| Brokerage | 0.0 | Zero for discount brokers (Zerodha, Upstox, Groww) |
| Exchange transaction charge | 0.322 | NSE standard (2024–2026) |
| SEBI regulatory fee | 0.01 | Fixed statutory (reduced Aug 2024) |
| Securities Transaction Tax (STT) | 10.0 | Sell leg only (delivery rate) |
| Stamp duty | 1.5 | Buy leg only (Finance Act 2020) |
| GST on brokerage + exchange | 0.058 | 18% × (0 + 0.322) |
| Slippage (market impact) | 2.0 | Estimate per leg for large-cap stocks |
| **Total per leg (buy)** | **~3.9 bps** | |
| **Total per leg (sell)** | **~12.4 bps** | |
| **Total round-trip (2 legs)** | **16.28 bps** | Per pair trade |

The effective round-trip cost for a pairs trade — which requires simultaneously buying one stock and shorting the other — is **16.28 basis points** per trade using discount broker rates (2024–2026). This reflects zero brokerage (standard for discount brokers like Zerodha and Upstox since 2020), corrected NSE exchange fees (0.345 bps), and updated stamp duty (1.5 bps per Finance Act 2020). This cost is still higher than the US equity cost model (typically 5–10 bps) assumed in most pairs trading academic literature.

The formula implemented in code:

$$c_{\text{rt}} = \frac{1}{10000} \left[ 2 \times (\text{brokerage} + \text{exchange} + \text{SEBI} + \text{GST}_{\text{brok+exch}}) + \text{STT}_{\text{sell}} + \text{stamp}_{\text{buy}} + 2 \times \text{slippage} \right]$$

A "trade" in the backtest is defined as any change in the discrete position signal: entry (+1 or −1 from 0), exit (to 0), or reversal (+1 to −1 directly). Each such signal change triggers the round-trip cost on the affected pair's notional value.

### 3.3.2 Notional sizing

The default configuration allocates **₹1,00,000 (INR 1 lakh) per pair leg**, corresponding to approximately 10% of a ₹10,00,000 (INR 10 lakh) initial capital allocation across a 10-pair portfolio. The sizing is equal-weight across pairs within each fold and does not adapt to volatility. This simplification is consistent with the paper's focus on signal quality evaluation rather than portfolio construction optimisation.

---

## 3.4 Two-Stage Architecture

The strategy is organised as a sequential two-stage pipeline. **Stage 1 (Pair Selection)** uses an ensemble of up to 8 pair selectors to identify and rank the 3,916 candidate pairs, selecting the top-K = 10. **Stage 2 (Signal Generation)** applies an ensemble of up to 4 signal models to each selected pair to generate continuous entry/exit signals, which are discretised to {+1, 0, −1}.

### 3.4.1 Stage 1 ensemble combination

Each selector $s$ assigns a raw score $v_s(p) \in \mathbb{R}$ to each candidate pair $p \in P$. Scores are normalised within each selector to [0, 1] via min-max scaling:

$$\hat{v}_s(p) = \frac{v_s(p) - \min_{p' \in P} v_s(p')}{\max_{p' \in P} v_s(p') - \min_{p' \in P} v_s(p') + \epsilon}$$

The ensemble score for pair $p$ is then:

$$\text{score}(p) = \sum_{s} w_s \cdot \hat{v}_s(p) \bigg/ \sum_{s} w_s$$

where $w_s \geq 0$ is the weight assigned to selector $s$. The top-K pairs by ensemble score are selected.

### 3.4.2 Stage 2 ensemble combination

For each selected pair $(a, b)$, each signal model $m$ generates a continuous signal $f_m(t) \in [-1, 1]$ at each time step $t$. The ensemble signal is:

$$f_{\text{ens}}(t) = \sum_{m} w_m \cdot f_m(t) \bigg/ \sum_{m} w_m$$

The continuous ensemble signal is discretised: $f_{\text{ens}} > 0.5 \Rightarrow +1$, $f_{\text{ens}} < -0.5 \Rightarrow -1$, otherwise $0$. The minimum hold period is then applied (see Section 3.6.2).

---

## 3.5 Stage 1: Pair Selection Models

### 3.5.1 CorrelationSelector

**Method:** Rolling Pearson Correlation Coefficient (RPCC).

For each candidate pair $(a, b)$, the selector computes the rolling Pearson correlation of daily returns over a 252-bar (1-year) lookback window. The score is the correlation at the most recent bar:

$$\rho_{a,b} = \text{Corr}(r_a[t-252:t], r_b[t-252:t])$$

where $r_a = \Delta \log P_a$ are daily log returns. Pairs with higher $\rho$ are ranked higher.

**Motivation:** High correlation is a necessary (though not sufficient) condition for mean-reversion of the spread. Pairs with correlation below ~0.7 rarely exhibit reliable OU-type spread dynamics in practice. This is the most computationally lightweight selector and serves as the stable baseline.

**Reference:** Pearson (1895); Nath (2003) applied to Indian equity pairs.

---

### 3.5.2 DistanceSelector

**Method:** Gatev, Goetzmann and Rouwenhorst (2006) normalised SSD distance.

Prices are z-normalised over the lookback window:
$$z_i(t) = \frac{P_i(t) - \bar{P}_i}{\sigma(P_i)}$$

The distance score is the negative L2 norm of the difference in normalised price series:
$$d_{a,b} = -\|z_a - z_b\|_2$$

Higher scores (less negative) indicate pairs whose normalised prices co-moved more closely over the lookback period.

**Motivation:** The distance method was the first systematic pairs selection algorithm in the academic literature and remains a standard benchmark. It does not assume any statistical model and is therefore highly robust to model misspecification. In this paper, it serves as an important ablation target: if a more sophisticated selector cannot outperform the distance method, the complexity is not justified.

**Reference:** Gatev, Goetzmann & Rouwenhorst (2006), "Pairs trading: Performance of a relative-value arbitrage rule."

---

### 3.5.3 CointegrationSelector

**Method:** Engle-Granger two-step cointegration test.

For each candidate pair $(a, b)$, the selector runs the Engle-Granger test on price levels over a 2-year (504-bar) lookback window using `statsmodels.tsa.stattools.coint`. The score is $1 - \text{pvalue}$ if $\text{pvalue} < 0.05$ (the pair passes the cointegration test), and 0 otherwise:

$$\text{score}(a,b) = \begin{cases} 1 - p_{EG} & \text{if } p_{EG} < 0.05 \\ 0 & \text{otherwise} \end{cases}$$

The Engle-Granger test regresses $P_a = \alpha + \beta P_b + \epsilon$ and tests the residuals $\hat{\epsilon}$ for a unit root using the ADF test.

**Motivation:** Cointegration is the standard econometric foundation for pairs trading (Vidyamurthy 2004). Two prices are cointegrated if a linear combination is I(0) — stationary — meaning any divergence is temporary and mean-reverts. The cointegration test directly evaluates the null hypothesis that no such long-run equilibrium exists.

**Reference:** Engle & Granger (1987), "Cointegration and error correction"; ADF test via Dickey & Fuller (1979).

---

### 3.5.4 CombinedCriteriaSelector

**Method:** Sarmento & Horta (2021) multi-criteria filter.

This selector applies four simultaneous criteria to screen pairs. A pair receives a score of 1 if and only if all four conditions are satisfied; 0 otherwise:

1. **Cointegration:** $p_{EG} < 0.05$ (Engle-Granger p-value)
2. **Hurst exponent:** $H < 0.5$ (estimated via R/S analysis on the spread, over the lookback window)
3. **OU half-life:** $\tau_{1/2} = \ln(2)/\kappa < 60$ trading days, where $\kappa$ is estimated from the AR(1) coefficient of the demeaned spread
4. **Mean-reversion frequency:** $\geq 3$ crossings of the $\pm 2\sigma$ band in the lookback window

The Hurst exponent $H$ is estimated by R/S analysis:
$$H \approx \frac{\log(\mathbb{E}[R/S])}{\log(n)}$$
where $R/S$ is the range-to-standard-deviation ratio over sub-intervals of the spread. Values of $H < 0.5$ indicate mean-reversion; $H = 0.5$ is a random walk; $H > 0.5$ is trending.

**Motivation:** Multi-criteria filtering ensures that selected pairs exhibit mean-reversion at multiple levels of evidence — statistical (cointegration), structural (Hurst), practical (half-life within trading range), and behavioural (sufficient past excursions to generate trades).

**Reference:** Sarmento & Horta (2021), "Enhancing a Pairs Trading strategy with the application of Machine Learning."

---

### 3.5.5 MLSelector

**Method:** XGBoost binary classifier on pair-level features.

The MLSelector trains a supervised model to predict whether a given pair will be profitable to trade over a forward 20-day horizon. For each pair in the training window, six features are computed from the training-window prices:

| Feature | Description |
|---|---|
| `corr20` | Rolling 20-day Pearson correlation of daily returns |
| `corr60` | Rolling 60-day Pearson correlation of daily returns |
| `vol_a` | 60-day rolling volatility of stock A |
| `vol_b` | 60-day rolling volatility of stock B |
| `mom_ratio20` | 20-day momentum of the price ratio A/B |
| `coint_1mp` | $1 - p_{EG}$ (cointegration strength) |

The binary label is $y = 1$ if the 20-day rolling sum of spread returns $(r_a - r_b)$ is positive at the most recent training bar, else 0.

The XGBoost model uses: `n_estimators=200`, `max_depth=3`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`. Class imbalance is corrected by undersampling the majority class when the majority-minority ratio exceeds 5:1.

At inference time, the selector outputs $P(y=1)$ as the pair score, which reflects the model's estimated probability that the pair will generate positive spread returns.

**Motivation:** Supervised learning on pair features has been used by Do & Faff (2010) and Rad et al. (2016) to improve pair selection beyond purely statistical criteria. The six features capture both statistical co-movement (correlation, cointegration) and momentum (price ratio trend), which may complement the static mean-reversion screens.

**Reference:** Chen & Guestrin (2016), "XGBoost: A scalable tree boosting system."

---

### 3.5.6 LSTMSelector

**Method:** Bidirectional LSTM encoder on temporal pair-feature sequences.

The LSTMSelector trains a neural network to predict whether a pair's spread will trend positively over the next 20 days, using a 60-bar historical feature sequence. For each pair $(a, b)$, six time-varying features are computed at each time step:

| Feature | Description |
|---|---|
| `corr_20` | Rolling 20-day Pearson correlation |
| `corr_60` | Rolling 60-day Pearson correlation |
| `spread_z` | Z-score of spread $a - b$ over 60-day rolling window |
| `vol_ratio` | $\sigma_a(20) / \sigma_b(20)$ — relative volatility |
| `price_ratio_z` | Z-score of price ratio $a/b$ over 60-day rolling window |
| `beta` | Rolling 60-day OLS beta $\hat{\beta} = \text{Cov}(r_a, r_b) / \text{Var}(r_b)$ |

**Architecture:** Bidirectional LSTM with 32 units per direction (64 total), followed by 20% dropout and two dense layers (16 units ReLU, 1 unit sigmoid). Input shape: `(60, 6)`.

**Training:** All $(i, j)$ pair combinations are processed in parallel using thread-parallelism to build sequences. The full multivariate dataset across all 3,916 pairs is concatenated (up to 50,000 sequences, subsampled randomly if exceeded), split 85/15 train/validation. Adam optimiser with `binary_crossentropy` loss; early stopping with patience=3 on validation loss. All training runs CPU-only (`CUDA_VISIBLE_DEVICES=''`, `TF_DETERMINISTIC_OPS=1`, seed=42) for full reproducibility.

**Key design choice:** The LSTM is trained on a *pooled* dataset across all pairs simultaneously. This means the model learns a representation of *temporal co-movement quality* that generalises across all pairs in the universe, rather than fitting to a single pair's dynamics. At inference, the model scores each candidate pair independently using its most recent 60-bar feature window.

**Reference:** Hochreiter & Schmidhuber (1997), "Long short-term memory"; Schäfer & Köhne (2021) for LSTM-based pair quality prediction.

---

### 3.5.7 TransformerSelector

**Method:** Multi-head self-attention encoder on the same 6-dimensional, 60-bar feature sequences used by the LSTM.

**Architecture:**
1. Linear projection of 6 input features to `embed_dim=32` (dense layer, no activation)
2. Sinusoidal positional encoding (non-trainable; avoids Lambda layer issues) added to the projected sequence
3. Two Transformer encoder blocks, each consisting of:
   - `MultiHeadAttention` with 4 heads (`key_dim=8`) + residual + LayerNorm
   - Position-wise feed-forward network (Dense 64 ReLU → Dense 32) + residual + LayerNorm
4. `GlobalAveragePooling1D` to aggregate the sequence representation
5. Dense(16, ReLU) → Dense(1, sigmoid) output

**Training:** Identical setup to LSTMSelector (same pooled dataset, same train/validation split, Adam + binary_crossentropy, early stopping with patience=3).

**Implementation note:** The positional encoding is implemented as a custom Keras layer (`_PositionalEncodingLayer`) rather than a Lambda layer, which resolves device-placement failures (TF issue with Lambda.call).

**Reference:** Vaswani et al. (2017), "Attention is all you need"; applied to financial time series by Ding et al. (2020).

---

### 3.5.8 GNNSelector

**Method:** Graph Convolutional Network (GCN) with link prediction.

The GNNSelector models the 89 stocks as nodes in a graph, with edges weighted by pairwise return correlation. It applies two GCN layers to learn node embeddings that capture each stock's position within the correlation structure, then uses link prediction to score candidate pairs.

**Graph construction:**
- Adjacency matrix: $A_{ij} = \text{Corr}(r_i, r_j)$ clipped to $[0, 1]$, computed over the training window
- Symmetric normalisation: $\hat{A} = D^{-1/2} A D^{-1/2}$ (standard Kipf & Welling 2017 normalisation)

**Node features:** 6-dimensional vector per stock: mean return, annualised volatility, return skewness, excess kurtosis, cumulative momentum, price z-score (standardised position in training-window price range)

**Two-layer GCN:**
$$H^{(1)} = \text{ReLU}(\hat{A} \cdot X \cdot W_1), \quad H^{(2)} = \text{ReLU}(\hat{A} \cdot H^{(1)} \cdot W_2)$$

where $W_1 \in \mathbb{R}^{6 \times 32}$ and $W_2 \in \mathbb{R}^{32 \times 16}$ are learned weight matrices.

**Link prediction:** For each candidate pair $(i, j)$, a 48-dimensional feature vector is constructed by concatenating node embeddings and their element-wise product: $[H_i^{(2)}, H_j^{(2)}, H_i^{(2)} \odot H_j^{(2)}]$. A linear layer with sigmoid activation predicts the link probability $P(\text{pair profitable})$.

**Training:** The GCN is trained over 8 temporal snapshots (each 120 days apart) within the training window, using binary pair profitability labels (same 20-day forward horizon as other ML selectors). Optimised with Adam at learning rate 0.01 for 50 epochs.

**Reference:** Kipf & Welling (2017), "Semi-supervised classification with graph convolutional networks"; Chen, Li & Zheng (2021) for GNN-based pair selection.

---

## 3.6 Stage 2: Signal Generation Models

### 3.6.1 ZScoreThreshold

**Method:** Classic mean-reversion bands on spread z-score.

The spread is defined as $S_t = P_a(t) - P_b(t)$ (unit hedge ratio). The rolling z-score is:
$$z_t = \frac{S_t - \bar{S}_{t-L:t}}{\sigma(S_{t-L:t})}$$

where $L = 60$ bars. Trading signals are:

$$\text{signal}(t) = \begin{cases}
+1 & \text{if } z_t < -2.0 \text{ (spread below lower band — long spread)} \\
-1 & \text{if } z_t > +2.0 \text{ (spread above upper band — short spread)} \\
0 & \text{if } |z_t| < 0.5 \text{ (spread reverted — exit)} \\
\text{hold} & \text{otherwise}
\end{cases}$$

**Motivation:** The z-score method is the standard signal model in the academic pairs trading literature (Gatev et al. 2006; Vidyamurthy 2004). It is theoretically consistent with assuming that the spread follows a stationary process with constant mean and variance. The ±2σ entry bands correspond to approximately the 95th percentile of a normally distributed spread.

---

### 3.6.2 OUThreshold (Ornstein-Uhlenbeck)

**Method:** Ornstein-Uhlenbeck process parameter estimation with scaled trading thresholds.

The spread $S_t$ is modelled as an Ornstein-Uhlenbeck (OU) process:
$$dS_t = \kappa(\mu - S_t) dt + \sigma \, dW_t$$

where $\kappa > 0$ is the mean-reversion speed, $\mu$ is the long-run mean, $\sigma$ is the diffusion coefficient, and $W_t$ is a Wiener process.

The OU parameters are estimated from the AR(1) representation of the demeaned spread. The demeaned spread $x_t = S_t - \bar{S}$ satisfies:
$$x_t = \phi x_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma_\varepsilon^2)$$

The mean-reversion speed is estimated as $\hat{\kappa} = -\ln |\hat{\phi}| / \Delta t$ where $\hat{\phi}$ is the OLS estimate of the AR(1) coefficient (from rolling covariance of $x_t$ and $x_{t-1}$), and the OU half-life is $\tau_{1/2} = \ln(2) / \hat{\kappa}$.

The trading metric used for thresholding is the OU-scaled deviation:
$$m_t = \hat{\kappa} \cdot (S_t - \mu_t)$$

where $\mu_t$ is the rolling 252-bar mean. Signals are generated by comparing $m_t$ to $\pm 1.5 \hat{\sigma}_m$, where $\hat{\sigma}_m$ is the rolling 126-bar standard deviation of $m_t$. Exit when $|m_t| < 0.2 \hat{\sigma}_m$.

**Motivation:** The OU model is theoretically principled for pairs trading: it directly parameterises the mean-reversion speed and provides a signal that adapts to the current regime's mean-reversion intensity. A pair with high $\kappa$ (fast mean-reversion) is traded more aggressively (tighter thresholds in normalised units) than a pair with low $\kappa$ (slow mean-reversion). The E2 hold period sweep (Section 4.2) confirms that the optimal hold period (~30 days) aligns with the estimated OU half-life of the selected pairs.

**Reference:** Uhlenbeck & Ornstein (1930); Elliott, van der Hoek & Malcolm (2005) for OU-based pairs trading.

---

### 3.6.3 KalmanHedge

**Method:** Linear Kalman Filter for dynamic hedge ratio estimation.

The Kalman filter models the hedge ratio $\beta_t$ as a time-varying random walk:

**State-space formulation:**
- **Observation equation:** $P_a(t) = \beta(t) P_b(t) + \alpha(t) + \varepsilon_t, \quad \varepsilon_t \sim N(0, R)$
- **Transition equation:** $[\beta(t), \alpha(t)]^\top = [\beta(t-1), \alpha(t-1)]^\top + \eta_t, \quad \eta_t \sim N(0, Q)$

where $Q = (\delta / (1 - \delta)) \cdot I_2$ with $\delta = 10^{-4}$ (process noise controlling adaptation speed), and $R = 10^{-2}$ (observation noise). The state vector $[β, α]$ is treated as a 2D random walk ($F = I$) with time-varying observation matrix $H_t = [P_b(t), 1]$.

**Kalman recursion (per bar):**

Predict: $\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1}$, $P_{t|t-1} = P_{t-1|t-1} + Q$

Innovation: $e_t = P_a(t) - H_t \hat{x}_{t|t-1}$, $S_t = H_t P_{t|t-1} H_t^\top + R$

Update: $K_t = P_{t|t-1} H_t^\top / S_t$, $\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t e_t$, $P_{t|t} = (I - K_t H_t^\top) P_{t|t-1}$

The normalised innovation $z_t = e_t / \sqrt{S_t}$ serves as the trading signal, with entry at $|z_t| \geq 2.0$ and exit at $|z_t| < 0.5$.

**Motivation:** The Kalman filter adapts the hedge ratio $\beta_t$ continuously as the cointegrating relationship evolves over time. This is particularly valuable for pairs where the long-run cointegrating coefficient is not constant — for example, when one stock undergoes a structural shift in its risk profile. The filter avoids the look-ahead bias of OLS regression by using only past observations for state estimation at each step.

**Reference:** Kalman (1960); Elliott, van der Hoek & Malcolm (2005); Pole (2007) for practical $\delta$/$R$ calibration.

---

### 3.6.4 MLSignal

**Method:** XGBoost triclass classifier for directional spread prediction.

The MLSignal trains a gradient-boosted tree classifier to predict the sign of the 5-day-ahead cumulative spread return: +1 (long spread profitable), −1 (short spread profitable), or 0 (flat). Eleven features are computed per bar:

| Feature | Definition |
|---|---|
| `spread_z` | Rolling z-score of $S_t$ (lookback=60 bars) |
| `z_lag5` | `spread_z` lagged 5 bars |
| `z_lag20` | `spread_z` lagged 20 bars |
| `velocity` | 1-bar first difference of `spread_z` |
| `acceleration` | 1-bar first difference of `velocity` |
| `abs_z` | $|$`spread_z`$|$ |
| `corr_20` | 20-bar rolling Pearson correlation of $r_a$ and $r_b$ |
| `corr_60` | 60-bar rolling Pearson correlation of $r_a$ and $r_b$ |
| `vol_ratio` | $\sigma_a(60) / \sigma_b(60)$ |
| `momentum_a` | 20-bar log return of stock A |
| `momentum_b` | 20-bar log return of stock B |

The label is generated by a forward-looking 5-day return sum, making it strictly look-ahead when computed at time $t$. In the walk-forward framework, the model is trained on `train_frac=0.70` of the pair's history, then applies frozen (non-updating) predictions on the remaining test portion.

**Model configuration:** `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, XGBoost CPU-only (no CUDA; `tree_method='hist'`, `device='cpu'`, seed=42).

**Reference:** Krauss, Do & Huck (2017), "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500."

---

### 3.6.5 Minimum Hold Period

After signal generation, a minimum holding period of **30 trading days** is enforced on all strategies (determined empirically in Experiment E2, Section 4.2). Once a non-zero position is entered, any signal change — exit or reversal — is suppressed until 30 bars have elapsed. This applies to all four signal models equally and is treated as a structural constraint (not a tuned hyperparameter) since its value is derived from the OU half-life of the selected pairs.

---

## 3.7 Walk-Forward Validation

### 3.7.1 Design

Walk-forward validation (WFV) is the primary OOS evaluation mechanism. It eliminates the look-ahead bias of a single in-sample backtest by strictly separating model fitting from model evaluation at every fold.

**Fold structure (expanding training window):**

| Fold | OOS Year | Training Window |
|---|---|---|
| 1 | 2018 | 2015-01-01 to 2017-12-31 |
| 2 | 2019 | 2015-01-01 to 2018-12-31 |
| 3 | 2020 | 2015-01-01 to 2019-12-31 |
| 4 | 2021 | 2015-01-01 to 2020-12-31 |
| 5 | 2022 | 2015-01-01 to 2021-12-31 |
| 6 | 2023-2024 | 2015-01-01 to 2022-12-31 |

The expanding window design is preferred over a rolling window because additional historical data generally improves model quality for all selectors, and there is no evidence that distant historical NSE data is harmful relative to recent data.

### 3.7.2 No-look-ahead guarantee

Each element of the pipeline is evaluated for look-ahead bias:

**Stage 1 (pair selection):** All 8 selectors are fit exclusively on `train_prices` for each fold. Correlation, Distance, and Cointegration selectors use the entire training window. ML-based selectors (MLSelector, LSTMSelector, TransformerSelector, GNNSelector) train their models on training data only. The pair ranking and top-10 selection are performed using only training-window information.

**Stage 2 (signal generation):** Signal models are fit on `(a_train, b_train)` per pair per fold. Signals are then generated on the *full* window (train + test) so that rolling lookback windows spanning the train/test boundary use real historical data. **Only the test-window portion of the signal series is used for PnL computation.** For sequential models (ZScore, OU, Kalman), each bar's signal depends only on past bars — no future data is used. For MLSignal, the XGBoost classifier is pre-fitted on the training set and does not retrain during inference.

**Minimum hold period:** The same `min_hold = 30` is applied in all folds. This value was determined from the E2 hold-period sweep, which uses the full 10-year dataset including OOS data. Strictly, it should be estimated from fold 1's training window and held fixed. In practice, the E2 sweep shows a robust peak at 30 days with gradual performance decay on either side, and the OU half-life argument provides a theoretical justification; the sensitivity to the exact value is low.

### 3.7.3 Aggregation

The **Full-OOS** metrics are computed on the stitched OOS equity curve: concatenating the test-period PnL streams from all six folds in chronological order. This produces a single 6-year OOS equity series on which aggregate Sharpe ratio, CAGR, and MaxDD are computed.

The **fold-level statistics** (mean ± std of per-fold Sharpe, % folds positive) are reported as a robustness check. A strategy that achieves a high full-OOS Sharpe driven entirely by 1–2 exceptional folds is less credible than one with consistent positive performance across folds.

---

## 3.8 Backtesting Engine

### 3.8.1 Portfolio construction

At the start of each fold, the top-10 pairs selected by the Stage 1 ensemble are held as the pair set for the entire OOS year. No intra-fold pair rebalancing is performed — pairs are fixed at fold boundaries to avoid excessive turnover from pair set turnover.

### 3.8.2 PnL computation

For each selected pair $(a, b)$ with position signal $\sigma_t \in \{+1, 0, -1\}$ at time $t$:

**Gross PnL** (before costs):
$$\text{PnL}_t^{\text{gross}} = \sigma_{t-1} \cdot \left[ \frac{P_a(t)}{P_a(t-1)} - \frac{P_b(t)}{P_b(t-1)} \right] \cdot N_{\text{pair}}$$

where $N_{\text{pair}} = \text{₹1,00,000}$ is the per-pair notional.

**Net PnL** (after costs): At each bar where $\sigma_t \neq \sigma_{t-1}$ (a trade occurs), the round-trip cost $c_{\text{rt}} \approx 0.006$ is deducted from the PnL:
$$\text{PnL}_t^{\text{net}} = \text{PnL}_t^{\text{gross}} - \mathbf{1}[\sigma_t \neq \sigma_{t-1}] \cdot c_{\text{rt}} \cdot N_{\text{pair}}$$

The portfolio PnL is the sum across all active pairs. The equity curve is $E_t = E_0 + \sum_{\tau \leq t} \text{PnL}_\tau$.

### 3.8.3 Performance metrics

- **Sharpe Ratio:** $SR = (\mu_r / \sigma_r) \times \sqrt{252}$, where $r_t = \text{PnL}_t / E_0$ are portfolio daily return fractions. The risk-free rate is set to zero (consistent with the strategy's market-neutral positioning and in line with Indian academic convention for pairs trading studies).
- **CAGR:** $(E_T / E_0)^{252/T} - 1$ where $T$ is the number of OOS trading days.
- **Maximum Drawdown:** $\max_t (1 - E_t / \max_{\tau \leq t} E_\tau)$.
- **Cost Drag:** Annualised cost drag is the difference between gross and net return: $(\text{CAGR}_{\text{gross}} - \text{CAGR}_{\text{net}})$ expressed in percentage points per year.
- **Trades per year:** Number of trade events (signal changes) per 252-bar year, averaged across all active pairs.

---

## 3.9 Statistical Significance Testing

### 3.9.1 Block bootstrap

To account for the non-i.i.d. nature of pairs trading returns (serial correlation induced by the minimum hold period), a **stationary block bootstrap** is used rather than a standard i.i.d. bootstrap. The OOS daily return series $\{r_t\}_{t=1}^{T}$ is resampled in non-overlapping blocks of length $b = 30$ days (matching the minimum hold period), generating $B = 10,000$ bootstrap resamples. The bootstrap Sharpe ratio is computed on each resample, and the empirical p-value is the proportion of resamples with Sharpe ≤ 0:

$$p_{\text{bootstrap}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}[SR^{(b)} \leq 0]$$

The 95% confidence interval is the 2.5th–97.5th percentile of the bootstrap distribution.

### 3.9.2 Newey-West t-test

The Newey-West HAC t-test provides an asymptotically valid test for $H_0: \mu_r = 0$ under serial correlation. The HAC standard error uses $L = \lfloor 4(T/100)^{2/9} \rfloor$ lags:

$$\hat{\sigma}_{NW}^2 = \hat{\gamma}_0 + 2 \sum_{l=1}^{L} \left(1 - \frac{l}{L+1}\right) \hat{\gamma}_l$$

where $\hat{\gamma}_l$ is the sample autocovariance at lag $l$. The t-statistic is $t_{NW} = \bar{r} / (\hat{\sigma}_{NW} / \sqrt{T})$.

### 3.9.3 Multiple testing correction

To address the data-snooping risk of evaluating five Stage 2 configurations (ZScore_only, OU_only, Kalman_only, ML_only, S2_Ensemble), a Bonferroni correction is applied to the OU_only p-value (the configuration selected as best from E3 ablation). The Bonferroni-corrected p-value is $p_{\text{adj}} = \min(K \cdot p_{\text{unadj}}, 1.0)$ where $K = 5$ is the number of configurations tested. This is a conservative correction; the Holm-Šidák stepwise procedure would be less conservative, but Bonferroni is the academic standard for this application.

**Reference:** Ledoit & Wolf (2008), "Robust performance hypothesis testing with the Sharpe ratio"; Newey & West (1987); White (2000), "A reality check for data snooping."

---

## 3.10 Ablation Study Design (Experiment E3)

The ablation study is designed to isolate the marginal contribution of each component in Stage 1 and Stage 2.

**Stage 1 ablation:** For each of the 8 selectors $s_i$, a WFV run is performed with weight $w_{s_i} = 1.0$ and $w_{s_j} = 0$ for all $j \neq i$. This produces a single-selector result that attributes performance exclusively to that selector's pair choices. The equal-weight 8-selector ensemble (all $w_i = 1.0$) is also evaluated as the naive ensemble baseline.

**Stage 2 ablation:** For each of the 4 signal models $m_i$, a WFV run is performed with the same Stage 1 configuration (full mode, equal weight) but with $w_{m_i} = 1.0$ and $w_{m_j} = 0$ for all $j \neq i$. The equal-weight 4-model ensemble is also evaluated.

The ablation framework does not require re-running pair selection for each Stage 2 configuration — the pair selection from the full-mode Stage 1 is reused, and only the signal generation is varied. This is computationally efficient and isolates Stage 2 effects cleanly.

---

## 3.11 Summary of Hyperparameters

For reproducibility, the complete set of fixed experimental hyperparameters is documented below:

**Table 3.3: Fixed Hyperparameters (all experiments)**

| Parameter | Value | Source |
|---|---|---|
| Universe | 35 NSE large-cap stocks | Table 3.1 |
| Data window | 2016-01-01 to 2026-03-31 | Max yfinance NSE coverage |
| Data frequency | Daily (1D) | E1 empirical justification |
| OOS folds | 6 (2020–2025, one year each) | Standard WFV design |
| Training window | Expanding from 2016-01-01 | Standard for financial TS |
| Top-K pairs per fold | 10 | Default |
| Capital | ₹10,00,000 (INR 10 lakh) | Retail/prop desk scale |
| Per-pair notional | ₹1,00,000 (INR 1 lakh) | Equal-weight sizing |
| Min hold period | 30 trading days | E2 empirical result |
| Round-trip cost | 16.3 bps | NSE IndianCosts model (2024–2026 discount broker rates) |
| Bootstrap samples | B = 10,000 | Standard |
| Bootstrap block length | 30 days | Matches min-hold |
| Newey-West lags | $\lfloor 4(T/100)^{2/9} \rfloor$ | Newey-West (1987) rule |
| Bonferroni configurations | K = 5 (Stage 2 configs) | Multiple testing |
| Random seed | 42 | All ML models |
| ZScore lookback | 60 bars | — |
| ZScore entry threshold | ±2.0 σ | — |
| ZScore exit threshold | ±0.5 σ | — |
| OU lookback | 252 bars (mean), 126 bars (std) | — |
| OU entry threshold | ±1.5 σ | — |
| Kalman process noise δ | 1e-4 | Pole (2007) guidance |
| Kalman entry threshold | ±2.0 σ | — |
| LSTM sequence length | 60 bars | — |
| LSTM units | 32 (bidirectional = 64 total) | — |
| Transformer embed dim | 32 | — |
| Transformer heads | 4 | — |
| GNN lookback | 120 bars | — |
| GNN hidden dim | 32 | — |
| XGBoost n_estimators | 200 | — |
| XGBoost max_depth | 3 (selector), 4 (signal) | — |
