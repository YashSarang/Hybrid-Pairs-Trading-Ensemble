# Avellaneda & Lee (2010) — PCA-OU Statistical Arbitrage

**Status:** 🔄 Implementation Complete, Testing in Progress

---

## 📄 Paper Details

**Title:** "Statistical Arbitrage in the U.S. Equities Market"  
**Authors:** Marco Avellaneda, Jeong-Hyun Lee  
**Published:** Quantitative Finance, 10(7), 761-782 (2010)  
**Data:** S&P 500 US equities, 1997-2007  

---

## 🎯 Methodology

### Core Idea:
Decompose stock returns into **common factors** (systematic risk) and **idiosyncratic residuals** (stock-specific noise), then trade only on mean-reversion of the residuals.

### Algorithm:

#### 1. **Factor Decomposition (PCA)**
```
r_{i,t} = α_i + Σ_k β_{i,k} f_{k,t} + ε_{i,t}
```
- `r_{i,t}` = return of stock i at time t
- `f_{k,t}` = k-th common factor (extracted via PCA)
- `β_{i,k}` = factor loading (exposure of stock i to factor k)
- `ε_{i,t}` = idiosyncratic residual (stock-specific, mean-reverting)

#### 2. **OU Process Modeling**
Model each residual `ε_{i,t}` as Ornstein-Uhlenbeck process:
```
dε_t = κ(μ - ε_t)dt + σdW_t
```
- `κ` = mean-reversion speed
- `μ` = long-run mean
- `σ` = volatility

#### 3. **S-Score Calculation**
```
s_t = (ε_t - μ) / σ_eq
where σ_eq = σ / √(2κ)
```

#### 4. **Trading Signals**
- **SHORT** when `s_t > +1.5` (residual too high, expect reversion down)
- **LONG** when `s_t < -1.5` (residual too low, expect reversion up)
- **EXIT** when `|s_t| < 0.5` (close to equilibrium)

---

## 🔑 Key Innovation

**Why PCA-OU is powerful:**
1. **Market-neutral by construction** — Trades only idiosyncratic risk, zero factor exposure
2. **Data-driven factor discovery** — PCA learns factors from data (vs. Fama-French assumed factors)
3. **Mean-reversion focus** — Only trades stocks with statistically significant mean-reversion
4. **Industry standard** — Widely used in hedge funds for stat-arb

---

## 📊 Claimed Results (Paper)

**Dataset:** S&P 500, 2003-2007  
**Results:**
- **Sharpe Ratio:** 1.5 - 2.0 (gross)
- **Market Beta:** ≈ 0 (market-neutral)
- **Best Performance:** High-volatility periods (2003, 2007)
- **Number of Factors:** 15 (explains ~40-50% of variance)

---

## 🧪 Our Reproduction on NSE

### Configuration:
- **Universe:** 35 NSE large-cap stocks (vs. 500 in paper)
- **Factors:** 10 PCA components (fewer stocks → fewer factors)
- **Formation Window:** 252 days (1 year)
- **OU Estimation:** 60 days rolling
- **Entry Threshold:** ±1.5 σ_eq
- **Exit Threshold:** ±0.5 σ_eq
- **Half-life Constraint:** 5-60 days

### Test Periods:
- 2020-01-01 to 2020-12-31
- 2021-01-01 to 2021-12-31
- 2022-01-01 to 2022-12-31
- 2023-01-01 to 2023-12-31
- 2024-01-01 to 2024-12-31

---

## 🚀 Running the Reproduction

### Prerequisites:
```bash
pip install numpy pandas scikit-learn statsmodels yfinance
```

### Execute:
```bash
cd /d/code/Hybrid-Pairs-Trading-Ensemble/Literature-Review/2010-PCA-OU-Avellaneda-StatArb
python reproduction.py
```

### Output:
- Console: Detailed metrics for each period
- `results.json`: All results in structured format

---

## 📈 Our Results (NSE 35 stocks, 2020-2024)

> **Status:** 🔄 Testing in Progress — Results will appear here after first run

### Preliminary Expectations:

**Likely Outcome:** ⚠️ **Partial Match**
- ✅ Method should work (PCA-OU is theoretically sound)
- ⚠️ Lower Sharpe than US claims (NSE has higher costs, different regime)
- ✅ Market-neutral property should hold
- ⚠️ Fewer tradeable stocks (35 vs 500)

**Why NSE Might Differ:**
1. **Universe size:** 35 stocks vs 500 → fewer diversification opportunities
2. **Transaction costs:** NSE 60 bps vs US 10-20 bps
3. **Market structure:** Emerging market with different co-movement patterns
4. **Sample period:** 2020-2024 includes COVID crash (high regime uncertainty)

---

## 🔬 Implementation Details

### Classes:
1. **`OUProcess`** — Fits OU parameters (κ, μ, σ) via AR(1) regression
2. **`PCAOUStrategy`** — Main strategy class
   - `fit_pca()` — Extract common factors
   - `compute_residuals()` — Calculate idiosyncratic components
   - `fit_ou_models()` — Fit OU to each residual
   - `generate_signals()` — Create trading signals
   - `backtest()` — Full backtest with performance metrics

### Key Formulas:

**OU Parameter Estimation (AR(1)):**
```
ΔS_t = a + b*S_{t-1} + ε_t
→ κ = -b / Δt
→ μ = -a / b
→ σ = std(ε) / √Δt
→ half_life = ln(2) / κ
```

**Stationarity Test:**
- Augmented Dickey-Fuller (ADF) test
- Reject H0 (unit root) at p < 0.05

---

## 📚 References

**Original Paper:**
- Avellaneda, M., & Lee, J. H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761-782.

**Related Work:**
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Elliott, R. J., van der Hoek, J., & Malcolm, W. P. (2005). Pairs trading. *Quantitative Finance*, 5(3), 271-276.

---

## ✅ Reproduction Checklist

- [x] PCA factor extraction implemented
- [x] OU process estimation (AR(1) method)
- [x] Idiosyncratic residual calculation
- [x] S-score signal generation
- [x] Walk-forward backtesting framework
- [x] Stationarity testing (ADF)
- [x] Half-life constraints
- [ ] **Run on NSE data** ← NEXT STEP
- [ ] Compare results to paper claims
- [ ] Document reproduction status (✅ / ⚠️ / ❌)
- [ ] Add to Literature Review summary table

---

## 🎯 Success Criteria

### ✅ **Reproduced & Verified** if:
- Sharpe ratio > 0.5 on NSE
- Market-neutral (beta ≈ 0)
- Method works as described in paper

### ⚠️ **Partial Match** if:
- Sharpe ratio 0.0 - 0.5 (positive but below paper claims)
- Method works but weaker on NSE

### ❌ **Failed to Reproduce** if:
- Negative Sharpe ratio
- Fundamental implementation issue

---

**Next Steps:**
1. Run `reproduction.py` on NSE data
2. Analyze results vs claimed performance
3. Update status in `Literature-Review/README.md`
4. Add findings to thesis Chapter 2

---

**Maintained by:** Yash Sarang  
**Last Updated:** 2026-05-26  
**Implementation Status:** Complete, ready for testing
