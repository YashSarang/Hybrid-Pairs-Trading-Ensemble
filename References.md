# Sources and References

This document provides comprehensive references for the methodologies, algorithms, and techniques implemented in the Pairs Trading Ensemble system.

## Academic Papers and Research

### Pairs Trading Foundations

**Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006)**

- _Title_: "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
- _Journal_: The Review of Financial Studies, 19(3), 797-827
- _Implementation_: Distance-based pair selection using normalized price series
- _Key Contribution_: Established the distance method for identifying mean-reverting pairs
- _Used in_: `DistanceSelector` class in `core/selectors.py`

**Engle, R. F., & Granger, C. W. J. (1987)**

- _Title_: "Co-integration and Error Correction: Representation, Estimation, and Testing"
- _Journal_: Econometrica, 55(2), 251-276
- _Implementation_: Two-step Engle-Granger cointegration test
- _Key Contribution_: Statistical framework for testing long-run equilibrium relationships
- _Used in_: `CointegrationSelector` class in `core/selectors.py`

**Sarmento, S. M., & Horta, N. (2021)**

- _Title_: "A New Approach to Pairs Trading: Multi-Criteria Decision Making"
- _Journal_: Expert Systems with Applications, 173, 114677
- _Implementation_: Combined criteria approach using multiple statistical tests
- _Key Contribution_: Multi-criteria framework combining cointegration, Hurst exponent, and half-life
- _Used in_: `CombinedCriteriaSelector` class in `core/selectors.py`

### Statistical Methods

**Uhlenbeck, G. E., & Ornstein, L. S. (1930)**

- _Title_: "On the Theory of the Brownian Motion"
- _Journal_: Physical Review, 36(5), 823-841
- _Implementation_: Ornstein-Uhlenbeck mean reversion model
- _Key Contribution_: Mathematical framework for mean-reverting processes
- _Used in_: `OUThreshold` class in `core/entry.py`

**Kalman, R. E. (1960)**

- _Title_: "A New Approach to Linear Filtering and Prediction Problems"
- _Journal_: Journal of Basic Engineering, 82(1), 35–45
- _Implementation_: State-space recursion for dynamic hedge ratio [β, α] estimation
- _Key Contribution_: Optimal sequential estimation for time-varying linear systems
- _Used in_: `KalmanHedge` class in `core/entry.py`

**Elliott, R. J., van der Hoek, J., & Malcolm, W. P. (2005)**

- _Title_: "Pairs Trading"
- _Journal_: Quantitative Finance, 5(3), 271–276
- _Implementation_: Application of Kalman filter to dynamic hedge ratio in pairs trading; Kalman innovation as the tradable spread
- _Key Contribution_: First rigorous derivation of the Kalman Filter state-space model for pairs trading
- _Used in_: `KalmanHedge` class in `core/entry.py`

**Pole, A. (2007)**

- _Title_: Statistical Arbitrage: Algorithmic Trading Insights and Techniques
- _Publisher_: Wiley Finance
- _Implementation_: Practical guidance on δ (process noise) and R (observation noise) calibration for daily equity pairs; Chapter 4
- _Key Contribution_: Translates Kalman filter theory into calibration recipes for equity pairs trading
- _Used in_: `KalmanHedge` parameter defaults (`delta=1e-4`, `R_noise=1e-2`) in `core/entry.py`

### Market Microstructure and Regime Analysis

**Longin, F., & Solnik, B. (2001)**

- _Title_: "Extreme Correlation of International Equity Markets"
- _Journal_: The Journal of Finance, 56(2), 649-676
- _Implementation_: Correlation regime detection
- _Key Contribution_: Framework for identifying high/low correlation market regimes
- _Used in_: Market regime analysis in `core/predictions.py`

**Engle, R. F. (1982)**

- _Title_: "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation"
- _Journal_: Econometrica, 50(4), 987-1007
- _Implementation_: Volatility clustering analysis
- _Key Contribution_: Time-varying volatility modeling
- _Used in_: Market regime analysis in `core/predictions.py`

### Deep Learning Models

**Hochreiter, S. & Schmidhuber, J. (1997)**

- _Title_: "Long Short-Term Memory"
- _Journal_: Neural Computation, 9(8), 1735–1780
- _Implementation_: LSTM architecture for sequence-to-label pair selection
- _Key Contribution_: Solves the vanishing gradient problem enabling learning of long-range temporal dependencies
- _Used in_: `LSTMSelector` class in `core/selectors.py`

**Schuster, M. & Paliwal, K.K. (1997)**

- _Title_: "Bidirectional recurrent neural networks"
- _Journal_: IEEE Transactions on Signal Processing, 45(11), 2673–2681
- _Implementation_: Bidirectional LSTM wrapper enabling both forward and backward temporal context
- _Key Contribution_: BiRNN processes sequences in both directions, improving pattern recognition at sequence boundaries
- _Used in_: `LSTMSelector` class in `core/selectors.py` (when `bidirectional=True`)

**Kipf, T.N. & Welling, M. (2017)**

- _Title_: "Semi-Supervised Classification with Graph Convolutional Networks"
- _Conference_: International Conference on Learning Representations (ICLR 2017)
- _Implementation_: Two-layer GCN architecture and symmetrically normalised adjacency Â = D^{-½}(A+I)D^{-½}; self-loop convention
- _Key Contribution_: Established the spectral-to-spatial GCN formulation enabling scalable node classification and, by extension, link prediction on graphs
- _Used in_: `GNNSelector._gcn_forward()` and `GNNSelector._adjacency()` in `core/selectors.py`

**Zhang, M. & Chen, Y. (2018)**

- _Title_: "Link Prediction Based on Graph Neural Networks"
- _Conference_: Advances in Neural Information Processing Systems (NeurIPS 2018)
- _Implementation_: Link-prediction feature construction [hᵢ ‖ hⱼ ‖ hᵢ⊙hⱼ] capturing directionality, magnitude, and pairwise interaction
- _Key Contribution_: Demonstrates that element-wise product of node embeddings is critical for link-prediction tasks beyond simple inner product or concatenation
- _Used in_: `GNNSelector._link_logits()` in `core/selectors.py`

**Matsunaga, A., Suzumura, T., & Takahashi, T. (2019)**

- _Title_: "Exploring Graph Neural Networks for Stock Market Predictions with Rolling Window Analysis"
- _Conference_: NeurIPS 2019 Workshop on Robust AI in Financial Services
- _Implementation_: Rolling graph-snapshot training for stock-market GNN; informs the multi-snapshot training protocol and the use of correlation-weighted adjacency for financial graphs
- _Key Contribution_: Validates GNN on stock-return prediction with a rolling-window scheme directly analogous to the approach used in `GNNSelector.fit()`
- _Used in_: `GNNSelector.fit()` snapshot construction in `core/selectors.py`

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017)**

- _Title_: "Attention Is All You Need"
- _Conference_: Advances in Neural Information Processing Systems (NeurIPS), 30
- _Implementation_: Transformer encoder architecture (multi-head self-attention + feed-forward sublayers + residual connections + layer norm); sinusoidal positional encoding scheme
- _Key Contribution_: Replaces recurrence entirely with self-attention, enabling parallel computation and direct long-range dependency modelling
- _Used in_: `TransformerSelector` class in `core/selectors.py`

**Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C. (2021)**

- _Title_: "A Transformer-based Framework for Multivariate Time Series Representation Learning"
- _Conference_: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD 2021), 2114–2124
- _Implementation_: Transformer encoder + GlobalAveragePooling1D for time-series classification; directly informs the aggregation strategy and model structure used here
- _Key Contribution_: Establishes the Transformer encoder (without decoder) as state-of-the-art for multivariate financial/sensor time-series classification
- _Used in_: `TransformerSelector` architecture in `core/selectors.py`

**Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L. (2023)**

- _Title_: "Transformers in Time Series: A Survey"
- _Conference_: Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI 2023)
- _Implementation_: Design rationale for positional encoding choice, encoder-only architecture, and global pooling head in financial time-series settings
- _Key Contribution_: Comprehensive survey situating design choices (fixed vs learned positional encoding, pooling strategies) for time-series Transformers
- _Used in_: `TransformerSelector` design decisions in `core/selectors.py`

**Fischer, T. & Krauss, C. (2018)**

- _Title_: "Deep learning with long short-term memory networks for financial market predictions"
- _Journal_: European Journal of Operational Research, 270(2), 654–669
- _Implementation_: Sliding-window feature construction, binary profitability label, and train/test temporal split approach adapted for pair selection
- _Key Contribution_: Establishes empirical methodology for applying LSTM to equity return prediction
- _Used in_: `LSTMSelector.fit()` and `LSTMSelector._make_sequences()` in `core/selectors.py`

### ML Signal Models

**Friedman, J. H. (2001)**

- _Title_: "Greedy function approximation: a gradient boosting machine"
- _Journal_: Annals of Statistics, 29(5), 1189–1232
- _Implementation_: Gradient Boosted Machine classifier as primary fallback for ML spread-signal prediction
- _Key Contribution_: Established the gradient boosting framework for function approximation using additive tree ensembles
- _Used in_: `MLSignal` class in `core/entry.py`

**Chen, T. & Guestrin, C. (2016)**

- _Title_: "XGBoost: A scalable tree boosting system"
- _Conference_: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016), 785–794
- _Implementation_: XGBoost classifier (primary model) with {-1,0,+1}→{0,1,2} label remapping for triclass spread-signal prediction
- _Key Contribution_: Regularised gradient boosting with column subsampling and approximate split-finding; state-of-the-art on tabular data
- _Used in_: `MLSignal` class in `core/entry.py`

**Krauss, C., Do, X. A., & Huck, N. (2017)**

- _Title_: "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500"
- _Journal_: European Journal of Operational Research, 259(2), 689–702
- _Implementation_: Supervised classification framework for spread-signal prediction; 11-feature construction (spread z-score, lags, velocity, momentum) and triclass labelling approach
- _Key Contribution_: Demonstrates that gradient-boosted trees and DNN outperform linear benchmarks on statistical arbitrage classification tasks
- _Used in_: `MLSignal.fit()` and `MLSignal._build_features()` in `core/entry.py`

## Technical Implementation Sources

### Data Sources and APIs

**Yahoo Finance API (yfinance)**

- _Source_: https://github.com/ranaroussi/yfinance
- _Implementation_: Real-time and historical market data fetching
- _Used in_: `YFinanceNSESource` class in `core/data.py`
- _Note_: Unofficial API wrapper for Yahoo Finance data

**NSE (National Stock Exchange of India)**

- _Source_: Official NSE data through Yahoo Finance
- _Implementation_: Indian equity market data with .NS suffix
- _Used in_: Stock symbol formatting and data retrieval

### Statistical Libraries

**Statsmodels**

- _Source_: https://www.statsmodels.org/
- _Implementation_: Statistical tests (ADF, OLS regression)
- _Used in_: Cointegration testing, regression analysis
- _Key Functions_: `adfuller()`, `OLS()`

**Pandas**

- _Source_: https://pandas.pydata.org/
- _Implementation_: Time series analysis and data manipulation
- _Used in_: Rolling statistics, correlation calculations, data alignment

**NumPy**

- _Source_: https://numpy.org/
- _Implementation_: Numerical computations and array operations
- _Used in_: Mathematical operations, statistical calculations

**Scikit-learn**

- _Source_: https://scikit-learn.org/
- _Implementation_: Machine learning models for pair selection
- _Used in_: `MLSelector` class with Random Forest

### User Interface

**Streamlit**

- _Source_: https://streamlit.io/
- _Implementation_: Interactive web application framework
- _Used in_: Complete user interface in `app.py`

## Cost Model Sources

### Indian Market Transaction Costs

**Securities Transaction Tax (STT)**

- _Source_: Securities and Exchange Board of India (SEBI)
- _Implementation_: 0.1% on equity delivery, 0.025% on intraday
- _Used in_: `IndianCosts` class in `core/backtest.py`

**Exchange Transaction Charges**

- _Source_: NSE/BSE official fee structures
- _Implementation_: ~0.00345% of turnover
- _Used in_: Exchange fee calculations

**SEBI Charges**

- _Source_: SEBI official regulations
- _Implementation_: 0.0001% of turnover (₹10 per crore)
- _Used in_: Regulatory fee calculations

**Goods and Services Tax (GST)**

- _Source_: Indian tax regulations
- _Implementation_: 18% on brokerage and other charges
- _Used in_: Tax calculations on all applicable fees

## Ensemble Methodology Sources

### Weight Normalization

- _Source_: Standard ensemble learning practices
- _Implementation_: L1 normalization ensuring weights sum to 1.0
- _Used in_: `normalize_weights()` function in `core/ensemble.py`

### Score Aggregation

- _Source_: Weighted voting ensemble methodology
- _Implementation_: Linear combination of normalized model scores
- _Used in_: `ensemble_pair_scores()` function in `core/ensemble.py`

## Risk Management Sources

### Position Sizing

- _Source_: Kelly Criterion and fixed fractional position sizing
- _Implementation_: Maximum capital per pair and concurrent pair limits
- _Used in_: `BacktestConfig` class in `core/backtest.py`

### Stop-Loss Mechanisms

- _Source_: Adaptive stop-loss literature
- _Implementation_: Soft stop-loss with position scaling and persistence checks
- _Used in_: Backtest engine risk management

## Performance Metrics Sources

### Sharpe Ratio

- _Source_: Sharpe, W. F. (1966). "Mutual Fund Performance"
- _Implementation_: Risk-adjusted return calculation
- _Used in_: Performance evaluation across all modules

### Maximum Drawdown

- _Source_: Standard risk management literature
- _Implementation_: Peak-to-trough decline measurement
- _Used in_: Risk assessment in backtesting and predictions

### Information Ratio

- _Source_: Grinold, R. C., & Kahn, R. N. (1999). "Active Portfolio Management"
- _Implementation_: Excess return per unit of tracking error
- _Used in_: Benchmark comparison in `core/reports.py`

## Data Quality and Validation

### Missing Data Handling

- _Source_: Little, R. J. A., & Rubin, D. B. (2019). "Statistical Analysis with Missing Data"
- _Implementation_: Forward fill and dropna strategies
- _Used in_: Data preprocessing across all modules

### Outlier Detection

- _Source_: Tukey, J. W. (1977). "Exploratory Data Analysis"
- _Implementation_: Interquartile range and z-score methods
- _Used in_: Data validation and cleaning

## Optimization Techniques

### Vectorized Operations

- _Source_: NumPy and Pandas best practices
- _Implementation_: Avoiding Python loops for numerical computations
- _Used in_: All statistical calculations for performance

### Lazy Evaluation

- _Source_: Functional programming principles
- _Implementation_: Computing results only when needed
- _Used in_: Report loading and prediction generation

## Disclaimer

This implementation is for educational and research purposes. The methodologies are based on academic research and may not reflect the exact implementations used by professional trading firms. Users should conduct their own research and testing before using any trading strategies in live markets.

## Contributing

When adding new methodologies or algorithms, please:

1. Add appropriate academic references to this document
2. Include source citations in code comments
3. Document any modifications made to published algorithms
4. Provide links to original papers or implementations where possible

## Last Updated

March 2026 — Added Elliott et al. (2005) and Pole (2007) for `KalmanHedge`. Added Deep Learning section: Hochreiter & Schmidhuber (1997), Schuster & Paliwal (1997), Fischer & Krauss (2018) for `LSTMSelector`. Added Vaswani et al. (2017), Zerveas et al. (2021), Wen et al. (2023) for `TransformerSelector`. Added Kipf & Welling (2017), Zhang & Chen (2018), Matsunaga et al. (2019) for `GNNSelector`. Added Friedman (2001), Chen & Guestrin (2016), Krauss et al. (2017) for `MLSignal`.
