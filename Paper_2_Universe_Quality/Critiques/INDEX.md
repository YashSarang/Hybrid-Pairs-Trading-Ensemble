---
purpose: MASTER INDEX — read this file first. Use it to navigate Critiques/ without reading old rounds.
last_updated: 2026-06-04
current_open_issues: 0
---

# Critiques — Master Index

## Workflow
1. Read **this file** to see overall status.
2. Read **Round_4_Open.md** to see all current issue statuses (all resolved).
3. Read **RESOLVED_SUMMARY.md** for a compact log of every fix applied.
4. Read Round_1/2/3.md ONLY if you need the original reviewer wording for a specific item.

---

## File Guide

| File | When to Read | Status |
|------|-------------|--------|
| INDEX.md (this file) | Always — start here | Active |
| Round_4_Open.md | When checking if any issues remain | All resolved |
| RESOLVED_SUMMARY.md | When you need to know what was fixed and when | Reference |
| Round_3.md | Only if you need original R3 reviewer wording | RESOLVED — skip |
| Round_2.md | Only if you need original R2 reviewer wording | RESOLVED — skip |
| Round_1.md | Only if you need original R1 reviewer wording | RESOLVED — skip |

---

## Current Status (as of 2026-06-04)

| Round | Verdict | Items | Open |
|-------|---------|-------|------|
| 1 | Reject | 16 | 0 |
| 2 | Major Revision | 11 | 0 |
| 3 | Reject / Resubmit (26 items) | 26 | 0 |
| 4 (self-audit) | — | 29 | **0** |

**All critique items resolved. Thesis drafts are submission-ready.**

---

## Open Issues (next round if resubmitted)

None currently. If a new critique round arrives, create Round_5_Open.md and update this index.

---

## Acceptance Probability
- Journal of Financial Markets (JFM): ~55%
- Quantitative Finance (QF): ~70%

---

## Key Thesis Facts (quick reference)

- Headline result: NSE Nifty 50, stat-only ZScore, 4-fold rolling WFV: Net Sharpe +0.752 (95% CI [+0.422, +1.082], p=0.036)
- Folds: +1.127 / +0.218 / +0.627 / +1.036, 126 trades
- Honest multi-market mean: +0.284 (5.5x vs Nifty100 rolling +0.052)
- ML ensemble: exploratory negative finding (+0.354-+0.484 CPU; degrades stat-only +0.752)
- Transaction costs: 16.28 bps round-trip (India)
- Active selectors: 7 (Correlation, Distance, Cointegration, Combined + LSTM, Transformer, GNN; CNNSelector DISABLED)
- Primary framing: universe quality (Nifty50 > Nifty100) drives alpha, not geography
