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
- _Journal_: Journal of Basic Engineering, 82(1), 35-45
- _Implementation_: Dynamic hedge ratio estimation (placeholder)
- _Key Contribution_: Optimal estimation for time-varying parameters
- _Used in_: `KalmanHedge` class in `core/entry.py`

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

January 2026 - Initial documentation of all sources and references used in the system.
