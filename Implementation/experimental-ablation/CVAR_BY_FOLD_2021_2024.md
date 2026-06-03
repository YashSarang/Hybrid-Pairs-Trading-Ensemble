# CVaR Analysis by Fold — NSE Nifty 50 WFV (2021–2024)

**Strategy:** Statistical Selectors (Correlation, Distance, Cointegration, Combined) + ZScore Signal
**Capital Base:** ₹1,00,00,000 (1 Crore INR)
**Folds:** 4 (Rolling 12-month train → 12-month test)

---

## Fold 1 — Test Period: 2021-01-01 to 2021-12-31

- **Total days:** 248
- **Active days (non-zero P&L):** 116
- **Annualised Volatility:** 2.15%
- **VaR @ 95%:** -0.2294%
- **VaR @ 99%:** -0.3885%
- **CVaR / ES @ 95%:** -0.3369%
- **CVaR / ES @ 99%:** -0.5206%
- **Skewness:** -0.3464
- **Kurtosis (excess):** 5.8355

## Fold 2 — Test Period: 2022-01-01 to 2022-12-31

- **Total days:** 248
- **Active days (non-zero P&L):** 123
- **Annualised Volatility:** 3.806%
- **VaR @ 95%:** -0.4589%
- **VaR @ 99%:** -0.7416%
- **CVaR / ES @ 95%:** -0.625%
- **CVaR / ES @ 99%:** -0.8347%
- **Skewness:** 0.1172
- **Kurtosis (excess):** 4.1358

## Fold 3 — Test Period: 2023-01-01 to 2023-12-31

- **Total days:** 245
- **Active days (non-zero P&L):** 118
- **Annualised Volatility:** 5.378%
- **VaR @ 95%:** -0.1689%
- **VaR @ 99%:** -0.4894%
- **CVaR / ES @ 95%:** -0.3716%
- **CVaR / ES @ 99%:** -0.6725%
- **Skewness:** 7.9065
- **Kurtosis (excess):** 84.2786

## Fold 4 — Test Period: 2024-01-01 to 2024-12-31

- **Total days:** 246
- **Active days (non-zero P&L):** 108
- **Annualised Volatility:** 5.981%
- **VaR @ 95%:** -0.2298%
- **VaR @ 99%:** -0.6226%
- **CVaR / ES @ 95%:** -0.7282%
- **CVaR / ES @ 99%:** -1.9444%
- **Skewness:** -6.0172
- **Kurtosis (excess):** 88.52

---

## Pooled CVaR — All 4 Folds Combined

- **Total observations:** 987
- **Annualised Volatility:** 4.577%
- **VaR @ 95%:** -0.2638%
- **VaR @ 99%:** -0.6384%
- **CVaR / ES @ 95%:** -0.549%
- **CVaR / ES @ 99%:** -1.1234%
- **Skewness:** -0.1259
- **Kurtosis (excess):** 107.2689

---

*Generated automatically from actual 2021–2024 daily P&L series captured in walk-forward JSON.*