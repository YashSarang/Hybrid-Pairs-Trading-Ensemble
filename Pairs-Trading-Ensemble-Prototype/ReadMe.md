# 🟢 Pairs‑Trading Ensemble – Mini Research Stack

A lightweight yet extensible pipeline that **discovers, scores and back‑tests stock pairs** for statistical arbitrage by blending classical cointegration with a machine‑learning classifier (Random‑Forest).

This work is a prototype of a proposed framework for "Hybrid Models for Pairs Trading : An Ensemble of Statistical and Deep Learning" by Mr. Yash Sarang & Prof. Sudeep Bapat [IIT Bombay, India].

---

## 🌳 Repository Layout

The Layout has been created to resemble the aimed framework as close as possible:

### **Proposed Framework**

![Alt text](/Hybrid_Model_Framework_for_Pairs_Trading.png "Equity Curve")

### Code Layout

| File / Dir                | Role                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **`src/fetch.py`**        | Downloads Yahoo Finance prices, caches to `data/`, updates incrementally. **[Domain Specific - Stock Market data]**       |
| **`src/features.py`**     | Builds rolling pairwise features – correlation and Z‑score of price spread. **[Spread Estimation, Continuos Validation]** |
| **`src/ml_module.py`**    | Generates labels (did the spread revert?) and trains a Random‑Forest; persists to `models/rf.pkl`. **[Pair Selection]**   |
| **`src/stats_module.py`** | One‑shot Engle–Granger cointegration test (`engle_granger_test`). **[Pair Selection]**                                    |
| **`src/ensemble.py`**     | Fuses RF probability & statistical strength into one ensemble score; ranks pairs.                                         |
| **`src/backtest.py`**     | Toy mean‑reversion strategy (z>1.5 entry, z<0.3 exit). **[Execution Engine with Statistical Rule Validation]**            |
| **`src/main.py`**         | One‑command driver chaining all steps; CLI flags for training, back‑test, plotting.                                       |
| **`data/`**               | Cached CSV price                                                                                                          |
| **`models/`**             | Saved machine‑learning models.                                                                                            |

> **Tip:** All package‑internal imports are relative (e.g. `from .fetch import …`) so you can run `python -m src.main …` from the repo root.

---

## 🚀 Quick‑start

```bash
# 1  Create & activate venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2  Install dependencies
pip install -r requirements.txt

# 3  Run the full pipeline (download → features → RF train → rank → back‑test)
python -m src.main \
       --tickers AAPL,MSFT,GOOGL,META,NVDA \
       --start 2017-01-01 \
       --backtest --plot
```

CLI flags:

- `--skip-train`  Reuse existing model in `models/rf.pkl`.
- `--top-k 10`    Change how many pairs are scored/back‑tested.
- `--plot`        Requires `matplotlib`; shows equity curve.

---

## 📈 Sample Output Explained

```

Validation ROC‑AUC: 0.656 | Positive rate val: 0.586

Top pairs by ensemble score:
       pair  ml_conf  stat_score  ensemble
 META-MSFT 0.699050    0.773858  0.728973
 META-NVDA 0.693006    0.642177  0.672675
   ...

Back‑test summary:
 final PnL       32 124.57
 max drawdown    12 437.87
 trade count        386
 win rate          51.6 %
```

![Alt text](/Sample_Output.png "Equity Curve")

| Metric           | Meaning                                                                               |
| ---------------- | ------------------------------------------------------------------------------------- |
| **ROC‑AUC**      | How well the RF separates reverting vs non‑reverting spreads on the validation split. |
| **`ml_conf`**    | RF probability that the pair will mean‑revert within 60 days.                         |
| **`stat_score`** | `1 – p_value` from Engle–Granger; higher ⇒ stronger cointegration.                    |
| **`ensemble`**   | Weighted blend (default 0.6 × ML + 0.4 × stat).                                       |
| **final PnL**    | Dollar profit of the toy strategy (fixed \$10 k per leg, no costs).                   |
| **max drawdown** | Worst peak‑to‑trough drop in equity.                                                  |
| **win rate**     | Fraction of closed trades with positive PnL.                                          |

---

## 🔮 Extending the Framework

| Upgrade                          | How to do it                                                                                                                                                                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Multiple ML / DL models**      | • Train additional classifiers (e.g. LightGBM, XGBoost, Siamese LSTM) in their own modules – each exposes `predict_proba`.<br>• In `ensemble.py`, collect _all_ model confidences and feed a meta‑learner (e.g. logistic regression) or stack via weighted average. |
| **Richer statistical tests**     | Add Johansen cointegration, half‑life estimator, Hurst exponent in `stats_module.py`; normalise each to 0‑1 and plug into the ensemble.                                                                                                                             |
| **Hyper‑parameter tuning**       | Wrap `train_rf()` in `sklearn.model_selection.RandomizedSearchCV`; persist the best estimator.                                                                                                                                                                      |
| **Transaction costs & slippage** | Extend `run_backtest()` with cost per share or percentage spread; adjust PnL accordingly.                                                                                                                                                                           |
| **Intraday / live trading**      | 1) stream data via WebSocket or broker API, 2) reuse `score_one_pair()` for real‑time ranking, 3) execute orders via IB / Alpaca.                                                                                                                                   |
| **Automated retraining**         | Schedule weekly cron (or Airflow) that runs `python -m src.main --skip-train` for inference days, full training monthly.                                                                                                                                            |
| **Packaging**                    | Convert `src` into a proper pip package; add entry‑point console script; wrap notebooks for EDA.                                                                                                                                                                    |

The modular design (each concern in its own file, minimal shared state) lets you swap or add components with almost no changes elsewhere.

---

## 📜 License

MIT – do whatever you like; attribution appreciated.

---

Happy pair‑hunting! 🎣
