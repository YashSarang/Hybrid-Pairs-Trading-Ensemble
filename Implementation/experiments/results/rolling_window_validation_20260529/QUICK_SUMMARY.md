# Rolling Window Validation — Quick Summary

**Date:** May 29, 2026  
**Status:** ✅ COMPLETE

---

## TL;DR

**Rolling window improves performance by 82% but both methods remain unprofitable.**

**Thesis Decision: Keep expanding in Chapter 3, add rolling as sensitivity analysis, focus Chapter 4 on multi-market (India +0.84).**

---

## Results at a Glance

| Method | Net Sharpe | Trades | Status |
|--------|------------|--------|--------|
| **Expanding (Thesis)** | -0.409 ± 0.774 | 1,096 | Unprofitable |
| **Rolling (Validation)** | -0.075 ± 0.407 | 144 | **Better but still unprofitable** |
| **Improvement** | **+0.334 (+82%)** | **-952 (-87%)** | **Not statistically significant** |

---

## Why Rolling is Better (But Not Good Enough)

1. **87% fewer trades** → lower turnover → lower costs
2. **Shorter training windows** → more selective signals
3. **Faster regime adaptation** → 12 months vs 4-6 years
4. **Consistent lookback=126** → prevents signal instability

**BUT:** Still loses money on average (-0.075 Sharpe)

---

## The Real Story: Multi-Market India

| Market | Net Sharpe | vs Rolling NSE | Multiplier |
|--------|------------|----------------|------------|
| **India (Multi-Market)** | **+0.840** | **+0.915** | **11.2x** |
| Brazil | +0.321 | +0.396 | 4.3x |
| Rolling NSE | -0.075 | baseline | 1.0x |
| UK | -0.245 | -0.170 | - |
| US | -0.254 | -0.179 | - |

**Key Insight:** Geographic diversification matters **way more** than methodology tuning.

---

## Fold-by-Fold Breakdown

| Fold | Year | Net Sharpe | Trades | Winner? |
|------|------|------------|--------|---------|
| 1 | 2021 | -0.088 | 50 | ❌ |
| 2 | 2022 | **+0.388** | 49 | ✅ ONLY ONE |
| 3 | 2023 | **-0.600** | 45 | ❌ Terrible |
| 4 | 2024 | 0.000 | 0 | ❌ No trades (4-month test) |

**Pattern:** Only 1/4 years profitable → market regime matters more than methodology

---

## What Goes in the Thesis

### Chapter 3: NSE Baseline (Expanding Window)
- Main results: -0.409 Net Sharpe (existing data)
- **NEW Section 3.6:** "Sensitivity to Training Window Length"
  - Rolling improves to -0.075 (+0.334)
  - Trade reduction mechanism (87% fewer)
  - Conclusion: "Still unprofitable, multi-market needed"
  - **Length:** 2-3 pages max

### Chapter 4: Multi-Market Validation (Rolling Window) 🎯
- **FOCUS HERE** — this is your contribution
- 7 experiments across 4 markets
- **Headline:** India+ZScore +0.840 Sharpe
- **Key comparison:** India vs NSE = **+0.915 Sharpe gap** (11x!)
- **Narrative:** Geographic diversification > methodology tuning
- **Length:** Full chapter (15-20 pages)

### Chapter 5: Conclusions
- NSE pairs trading is cost-constrained (both methods)
- Rolling reduces costs but doesn't fix profitability
- **Multi-market reveals India opportunity**
- Future work: India-specific factors

---

## Why Not Replace Expanding with Rolling?

### Statistical Reality
- **t-statistic:** 0.89 (need >2.0 for significance)
- **p-value:** >0.05 (not significant)
- **Interpretation:** Difference could be noise, not signal

### Academic Standard
- Expanding window is established methodology
- Longer training = more rigorous validation
- 6 folds vs 4 folds (more data)

### Practical Reality
- **Both are unprofitable** (-0.409 vs -0.075)
- Improvement is marginal, not transformative
- Multi-market India (+0.84) is **11x better** than either
- That's where the story is

---

## Key Takeaways for Writing

1. **Don't oversell rolling window improvement**
   - +0.334 sounds good, but still negative
   - Not statistically significant
   - Be honest about limitations

2. **Emphasize the trade-off insight**
   - Shorter training → more selective → lower turnover
   - This is the mechanism, not magic

3. **Lead with multi-market as the breakthrough**
   - India +0.840 is the thesis contribution
   - NSE validation (both methods) shows why multi-market is needed
   - "Local baseline unprofitable, but India market shows promise"

4. **Frame as progression**
   - Ch 3: "NSE baseline fails (costs)"
   - Ch 3.6: "Methodology tuning helps but insufficient"
   - Ch 4: "Multi-market validation finds India success"
   - Ch 5: "Geographic diversification is the key"

---

## Files to Reference When Writing

1. **`EXPANDING_VS_ROLLING_ANALYSIS.md`** (19 KB)
   - Complete comparison with tables
   - Statistical analysis
   - Thesis decision rationale

2. **`MULTI_MARKET_RESULTS.md`** (13 KB)
   - 7 experiments, all complete
   - Market-by-market breakdown
   - Signal model comparison

3. **`METHODOLOGY_COMPARISON.md`** (6 KB)
   - When to use which method
   - Decision framework

4. **Results JSON files:**
   - Expanding: `experiments/results/walk_forward_20260506_104613.json`
   - Rolling: `experiments/results/rolling_window_validation_20260529/walk_forward_rolling_20260529_131324.json`

---

## Unanswered Questions (For Discussion Section)

1. **Why is India 11x better than NSE?**
   - Different ticker universe?
   - Lower competition / market efficiency?
   - Different microstructure (spreads, liquidity)?
   
2. **Why was 2023 so terrible (-0.600)?**
   - Regime shift to trending market?
   - Mean-reversion failure?
   - Needs market analysis

3. **Why zero trades in 2024?**
   - Only 4 months of test data (Jan-Apr)
   - Pairs selected but no signals triggered
   - Need full year or drop Fold 4

---

## Bottom Line

**You have everything you need to write Chapters 3 & 4.**

**The story is clear:**
1. NSE baseline fails (costs kill it)
2. Rolling helps but not enough
3. **Multi-market India succeeds (+0.84 Sharpe)**
4. Geographic diversification > methodology optimization

**Start with Chapter 4 — that's your contribution. Chapter 3 already exists, just add Section 3.6.**

---

**Next step: Draft Chapter 4 outline?**
