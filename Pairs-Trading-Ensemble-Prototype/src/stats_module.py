"""
stats_module.py – Classical statistics utilities for pair selection
===================================================================
Focus: **(i) Engle–Granger two‑step cointegration test** wrapped in a tiny,
scikit‑learn‑style interface, plus an optional helper for *rolling* p‑values.

Why ENGLE–GRANGER?
------------------
* Widely cited and very fast (closed‑form OLS under the hood).
* Returns an easily interpretable **p‑value** – ideal to rescale into a
  [0, 1] confidence weight for our ensemble.

Public API
----------
>>> from stats_module import engle_granger_test, rolling_pvalue

- `engle_granger_test(y, x, *, maxlag=None, trend="c", autolag="BIC") -> tuple[pval, stat]`
- `rolling_pvalue(series1, series2, window:int=252, **eg_kwargs) -> pd.Series`

No internal state; safe for multiprocessing.
"""
from __future__ import annotations

from typing import Tuple, Sequence, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

__all__: Sequence[str] = (
    "engle_granger_test",
    "rolling_pvalue",
)

# ----------------------------------------------------------------------------
# Single‑shot Engle–Granger test
# ----------------------------------------------------------------------------


def engle_granger_test(
    series1: pd.Series | np.ndarray,
    series2: pd.Series | np.ndarray,
    *,
    maxlag: int | None = None,
    trend: str = "c",
    autolag: str | None = "BIC",
) -> Tuple[float, float]:
    """Return (p‑value, test_statistic) of the Engle–Granger cointegration test.

    Parameters
    ----------
    series1, series2 : array‑like
        Price series (level data), **not** log‑returns.  Must be equal length.
    maxlag : int | None
        If None, the test chooses automatically.
    trend : {"c", "ct", "ctt", "n"}
        Same semantics as statsmodels: include constant, trend, both or none.
    autolag : {"AIC","BIC","tstat", None}
        Criterion for automatic lag length selection.

    Returns
    -------
    pvalue : float  (0 ⇢ strong cointegration, 1 ⇢ weak)
    test_statistic : float  (ADF statistic)
    """
    if len(series1) != len(series2):
        raise ValueError("Series must be of equal length.")

    # Convert to numpy, drop NaNs pair‑wise
    y = pd.DataFrame({"y": series1, "x": series2}).dropna().values.T
    if y.shape[1] < 20:
        # Test is unreliable for very small samples → return neutral values
        return 1.0, np.nan

    stat, pval, _ = coint(y[0], y[1], trend=trend,
                          maxlag=maxlag, autolag=autolag)
    return pval, stat

# ----------------------------------------------------------------------------
# Rolling window wrapper
# ----------------------------------------------------------------------------


def rolling_pvalue(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 252,
    **eg_kwargs: Any,
) -> pd.Series:
    """Compute rolling window p‑values (approx one trading year by default).

    Useful for feature engineering when you want *time‑varying* cointegration
    strength rather than a single snapshot.
    """
    pv = (
        pd.concat([series1, series2], axis=1)
        .rolling(window)
        .apply(lambda x: engle_granger_test(x[:, 0], x[:, 1], **eg_kwargs)[0], raw=False)
        .iloc[:, 0]
    )
    pv.name = "coint_pval"
    return pv
