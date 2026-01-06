# Ensemble Methodology & Exception Handling

## Overview

This document explains the ensemble methodology used in the pairs trading system and documents all exceptions and error handling mechanisms implemented in the codebase.

## Ensemble Architecture

### Two-Stage Ensemble Design

The system implements a **two-stage ensemble approach** that separates pair selection from trading signal generation:

```
Stage 1: Pair Selection Ensemble
├── Correlation-based Selection
├── Distance-based Selection (Gatev 2006)
├── Cointegration-based Selection
├── Combined Criteria Selection
└── ML-based Selection

Stage 2: Entry/Exit Signal Ensemble
├── Mean Reversion (Z-Score)
├── Ornstein-Uhlenbeck Model
└── Kalman Filter Hedge Ratio
```

### Stage 1: Pair Selection Ensemble

#### 1.1 Correlation Selector (`CorrelationSelector`)

**Method:** Pearson correlation coefficient
**Logic:**

- Calculates rolling correlation between all stock pairs
- Higher correlation indicates stronger relationship
- Selects pairs with correlation above threshold

**Implementation:**

```python
correlation = price_a.rolling(lookback).corr(price_b)
score = correlation.mean()  # Average correlation over period
```

#### 1.2 Distance Selector (`DistanceSelector`)

**Method:** Gatev et al. (2006) distance method
**Logic:**

- Normalizes prices using z-score transformation
- Calculates Euclidean distance between normalized price series
- Lower distance indicates better pair relationship

**Implementation:**

```python
norm_a = (price_a - price_a.rolling(lookback).mean()) / price_a.rolling(lookback).std()
norm_b = (price_b - price_b.rolling(lookback).mean()) / price_b.rolling(lookback).std()
distance = np.sqrt(((norm_a - norm_b) ** 2).mean())
score = 1 / (1 + distance)  # Invert so higher is better
```

#### 1.3 Cointegration Selector (`CointegrationSelector`)

**Method:** Engle-Granger cointegration test
**Logic:**

- Tests for long-term equilibrium relationship
- Uses Augmented Dickey-Fuller test on residuals
- Lower p-value indicates stronger cointegration

**Implementation:**

```python
# Run regression: price_a = alpha + beta * price_b + residuals
model = sm.OLS(price_a, sm.add_constant(price_b)).fit()
residuals = model.resid
adf_stat, p_value = adfuller(residuals)
score = 1 - p_value  # Lower p-value = higher score
```

#### 1.4 Combined Criteria Selector (`CombinedCriteriaSelector`)

**Method:** Multi-criteria approach (Sarmento & Horta, 2021)
**Logic:** Combines multiple statistical tests:

- Cointegration p-value < threshold
- Hurst exponent < 0.5 (mean-reverting)
- Half-life within reasonable bounds
- Minimum number of zero-crossings

#### 1.5 ML Selector (`MLSelector`)

**Method:** Supervised machine learning
**Logic:**

- Uses historical features to predict pair profitability
- Features include correlation, volatility, spread statistics
- Trained on past performance data

### Stage 2: Entry/Exit Signal Ensemble

#### 2.1 Z-Score Threshold (`ZScoreThreshold`)

**Method:** Mean reversion based on z-score
**Logic:**

- Calculates z-score of price spread
- Generates signals when spread deviates significantly from mean
- Entry: |z-score| > 2, Exit: |z-score| < 0.5

**Implementation:**

```python
spread = price_a - hedge_ratio * price_b
z_score = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
signal = np.where(z_score > 2, -1, np.where(z_score < -2, 1, 0))
```

#### 2.2 OU Threshold (`OUThreshold`)

**Method:** Ornstein-Uhlenbeck mean reversion model
**Logic:**

- Models spread as mean-reverting process
- Estimates mean reversion speed and equilibrium level
- Generates signals based on deviation from equilibrium

**Implementation:**

```python
# Estimate OU parameters
theta = -np.log(spread.autocorr()) / dt  # Mean reversion speed
mu = spread.mean()  # Long-term mean
sigma = spread.std() * np.sqrt(2 * theta)  # Volatility

# Generate signals based on deviation
deviation = (spread - mu) / sigma
signal = np.where(deviation > threshold, -1, np.where(deviation < -threshold, 1, 0))
```

#### 2.3 Kalman Hedge (`KalmanHedge`)

**Method:** Kalman filter for dynamic hedge ratio
**Logic:**

- Estimates time-varying hedge ratio using Kalman filter
- Adapts to changing relationship between stocks
- Currently placeholder implementation

### Ensemble Combination

#### Weight Normalization

All ensemble weights are normalized to sum to 1.0:

```python
def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        return {k: 1.0/len(weights) for k in weights}
    return {k: v/total for k, v in weights.items()}
```

#### Pair Score Aggregation

Stage 1 scores are combined using weighted average:

```python
def ensemble_pair_scores(scores_by_model, weights, top_k):
    combined_scores = {}
    for pair in all_pairs:
        weighted_score = sum(
            weights[model] * scores_by_model[model][pair]
            for model in scores_by_model
        )
        combined_scores[pair] = weighted_score

    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

#### Signal Aggregation

Stage 2 signals are combined using weighted average:

```python
def ensemble_signals(signals_by_model, weights):
    combined_signal = pd.Series(0.0, index=signals_by_model[list(signals_by_model.keys())[0]].index)

    for model_name, weight in weights.items():
        if model_name in signals_by_model:
            combined_signal += weight * signals_by_model[model_name]

    return combined_signal.round().astype(int)  # Discretize to {-1, 0, 1}
```

## Exception Handling & Error Management

### 1. Data Loading Exceptions

#### `YFinanceNSESource.get_prices()`

**Exceptions Handled:**

- `RuntimeError`: When yfinance is not installed
- `ValueError`: When unsupported frequency is requested
- `Exception`: General data fetch failures

```python
try:
    data = yf.download(tickers=tickers, start=start, end=end, interval=interval)
except Exception as e:
    raise RuntimeError(f"Failed to fetch data from Yahoo Finance: {str(e)}")
```

#### `CSVUploadSource.get_prices()`

**Exceptions Handled:**

- `Exception`: CSV/Parquet reading failures
- `ValueError`: Invalid file format or missing columns

```python
try:
    df = pd.read_csv(f)
except Exception:
    f.seek(0)
    df = pd.read_parquet(f)
```

### 2. Pair Selection Exceptions

#### `CointegrationSelector.score_pairs()`

**Exceptions Handled:**

- `Exception`: Statistical test failures (insufficient data, numerical issues)

```python
try:
    adf_stat, p_value = adfuller(residuals, maxlag=1)
    return 1.0 - p_value
except Exception:
    return 0.0  # Return neutral score on failure
```

#### `MLSelector.score_pairs()`

**Exceptions Handled:**

- `Exception`: Model training/prediction failures

```python
try:
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)
except Exception:
    return np.zeros(len(pairs))  # Return neutral scores
```

### 3. Backtest Engine Exceptions

#### `backtest_pairs()`

**Exceptions Handled:**

- `ValueError`: Empty or invalid price data
- `Exception`: General computation failures

```python
if prices is None or prices.empty:
    raise ValueError("prices is empty")

try:
    # Backtest computation
    pass
except Exception as e:
    # Log error and return empty result
    return BacktestResult(
        equity_gross=pd.Series([capital]),
        equity_net=pd.Series([capital]),
        # ... other empty fields
    )
```

### 4. Report Management Exceptions

#### `ReportManager.save_report()`

**Exceptions Handled:**

- `Exception`: File I/O failures during report saving

```python
try:
    with open(metadata_file, "r") as f:
        data = json.load(f)
        reports.append(ReportMetadata(**data))
except Exception:
    continue  # Skip corrupted report files
```

#### `BenchmarkComparison.fetch_index_returns()`

**Exceptions Handled:**

- `ValueError`: Unknown index name or empty data
- `RuntimeError`: Network/API failures
- `TypeError`: Data conversion failures

```python
try:
    strategy_total_return = float(strategy_returns.iloc[-1])
except (ValueError, TypeError):
    strategy_total_return = 0.0
```

### 5. UI Exception Handling

#### Streamlit Error Display

**Pattern Used Throughout:**

```python
try:
    # Risky operation
    result = perform_operation()
    st.success("Operation completed successfully")
except Exception as e:
    st.error(f"Operation failed: {str(e)}")
    st.info("Troubleshooting tips...")
```

### 6. Numerical Stability Exceptions

#### Division by Zero Protection

```python
# Standard deviation calculations
std = series.std(ddof=0)
if std == 0 or np.isnan(std):
    return pd.Series(0.0, index=series.index)
return (series - series.mean()) / std
```

#### Infinite Value Handling

```python
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
```

#### Missing Data Handling

```python
# Forward fill missing values
prices = prices.ffill()

# Drop pairs with insufficient data
if len(common_index) < min_periods:
    continue
```

## Risk Management Features

### 1. Soft Stop-Loss Mechanism

**Purpose:** Prevent excessive losses from diverging pairs
**Implementation:**

- Monitor z-score of spread
- Scale position when breach threshold
- Exit if breach persists

```python
breach = z_score.abs() > soft_stop_z
breach_persist = breach.rolling(soft_stop_persist_bars).sum() >= soft_stop_persist_bars
scale = pd.Series(1.0, index=z_score.index)
scale.loc[breach] = soft_stop_decay
signal_scaled = (signal * scale).round().astype(int)
signal_scaled.loc[breach_persist] = 0
```

### 2. Position Sizing Constraints

**Purpose:** Limit exposure per pair and total portfolio
**Implementation:**

- Maximum capital per pair
- Maximum concurrent pairs
- Notional limits

```python
notional_each = min(capital / max_concurrent_pairs, per_trade_cap)
```

### 3. Data Validation

**Purpose:** Ensure data quality before processing
**Checks:**

- Minimum data length
- No all-NaN columns
- Valid date ranges
- Sufficient overlap between pairs

## Performance Optimization

### 1. Vectorized Operations

All calculations use pandas vectorized operations instead of loops:

```python
# Vectorized correlation calculation
correlations = prices.rolling(window).corr()

# Vectorized z-score calculation
z_scores = (spreads - spreads.rolling(window).mean()) / spreads.rolling(window).std()
```

### 2. Lazy Evaluation

- Reports loaded only when selected
- Benchmark data fetched only when requested
- Models trained only when needed

### 3. Memory Management

- Use of generators for large datasets
- Cleanup of intermediate variables
- Efficient data structures

## Configuration Management

### 1. Default Parameters

All models have sensible defaults that work out-of-the-box:

```python
@dataclass
class BacktestConfig:
    capital: float = 100_000.0
    max_concurrent_pairs: int = 5
    per_trade_cap: float = 20_000.0
    # ... other defaults
```

### 2. Parameter Validation

Input parameters are validated before use:

```python
if lookback < 10:
    raise ValueError("Lookback period too short")

if threshold <= 0:
    raise ValueError("Threshold must be positive")
```

### 3. Graceful Degradation

System continues to function even when some components fail:

```python
# If ML model fails, use equal weights
try:
    ml_scores = ml_selector.score_pairs(prices, pairs)
except Exception:
    ml_scores = [0.5] * len(pairs)  # Neutral scores
```

## Testing & Validation

### 1. Unit Test Structure

Each component should have unit tests covering:

- Normal operation
- Edge cases
- Error conditions
- Performance benchmarks

### 2. Integration Testing

End-to-end tests covering:

- Full simulation pipeline
- Report generation
- Benchmark comparison
- UI interactions

### 3. Data Validation Tests

- Synthetic data generation
- Known outcome verification
- Statistical property validation

## Conclusion

The ensemble methodology provides robust pair selection and signal generation through:

1. **Diversified Approaches:** Multiple complementary methods reduce single-point-of-failure risk
2. **Flexible Weighting:** User-configurable weights allow strategy customization
3. **Robust Error Handling:** Comprehensive exception handling ensures system stability
4. **Performance Optimization:** Vectorized operations and lazy evaluation for efficiency
5. **Risk Management:** Built-in safeguards prevent excessive losses

The system is designed to be both powerful for experienced users and accessible for beginners, with sensible defaults and clear error messages throughout.
