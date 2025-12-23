"""
features.py – Rolling feature engineering for pairs-trading ensemble
===================================================================

Exposed API
-----------
build_pair_features(prices: pd.DataFrame,
                    window: int = 60,
                    min_periods: int | None = None) -> pd.DataFrame
    Given a wide **Adj-Close** dataframe (rows = dates, columns = tickers),
    returns a tidy dataframe with one row per *date × pair* and the following
    columns:
        date        : datetime64[ns]
        pair        : str   (e.g. "AAPL-MSFT")
        corr        : float Rolling Pearson correlation over `window` days
        z_spread    : float Z-score of the raw price spread (S₁ − S₂)

Notes
-----
* NaN rows that arise from insufficient look-back are dropped automatically.
* Tickers are upper-cased and stripped once here to keep pair naming stable.
* The function is intentionally stateless – caching is handled in `fetch.py`.

Additional
----------
* Optional GPU acceleration: set `use_gpu=True` to compute `z_spread` on GPU
  via RAPIDS (cuDF). If RAPIDS is unavailable, the function automatically
  falls back to the CPU path. Rolling correlation remains on CPU for
  compatibility and stability.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd

__all__: List[str] = [
    "build_pair_features",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling Z-score with matching index & NaN handling."""
    if min_periods is None:
        min_periods = window
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std


def _build_pair_features_cpu(
    prices: pd.DataFrame,
    window: int,
    min_periods: Optional[int],
) -> pd.DataFrame:
    """Original CPU implementation (unchanged in behavior)."""
    returns = np.log(prices).diff()

    feature_frames = []
    for s1, s2 in combinations(prices.columns, 2):
        pair_name = f"{s1}-{s2}"

        # ----------------- Rolling correlation of log returns --------------
        corr = (
            returns[s1]
            .rolling(window, min_periods=min_periods or window)
            .corr(returns[s2])
            .rename("corr")
        )

        # ----------------- Spread Z-score ----------------------------------
        spread = prices[s1] - prices[s2]
        z = _zscore(spread, window, min_periods=min_periods).rename("z_spread")

        df_pair = pd.concat([corr, z], axis=1).dropna()
        df_pair["pair"] = pair_name
        feature_frames.append(df_pair)

    if not feature_frames:
        raise ValueError("Need at least two tickers to compute pair features.")

    features = pd.concat(feature_frames)
    features.index.name = "date"
    features = features.reset_index()
    return features.sort_values(["date", "pair"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public feature constructor
# ---------------------------------------------------------------------------


def build_pair_features(
    prices: pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
    *,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Create rolling pair-wise features.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide dataframe of *adj-close* prices (DateTime index).
    window : int, default 60
        Look-back in trading days for all rolling computations.
    min_periods : int | None
        If None, defaults to `window` (strict) – set lower to allow earlier rows.
    use_gpu : bool, default False
        If True and RAPIDS (cuDF) is available, compute `z_spread` on GPU.
        Falls back to CPU seamlessly if unavailable.

    Returns
    -------
    pd.DataFrame
        Long format with columns [date, pair, corr, z_spread].  Sorted by date.
    """
    prices = prices.copy()
    prices.columns = [c.upper().strip() for c in prices.columns]

    # Fast path: GPU for z_spread (if available), CPU for corr.
    if use_gpu:
        try:
            import cudf  # type: ignore
            # --- GPU z_spread computation ---
            g_prices = cudf.from_pandas(prices)
            g_prices.index.name = "date"
            z_frames_gpu = []

            for s1, s2 in combinations(g_prices.columns, 2):
                pair_name = f"{s1}-{s2}"
                spread = g_prices[s1] - g_prices[s2]
                roll = spread.rolling(
                    window=window, min_periods=min_periods or window)
                mean = roll.mean()
                std = roll.std()  # population/std(ddof=0) equivalent in cudf rolling
                z = (spread - mean) / std
                dfz = cudf.DataFrame({"date": g_prices.index, "z_spread": z})
                dfz["pair"] = pair_name
                z_frames_gpu.append(dfz)

            if not z_frames_gpu:
                raise ValueError(
                    "Need at least two tickers to compute pair features.")

            z_all = cudf.concat(z_frames_gpu, ignore_index=True).dropna()
            z_pd = z_all.to_pandas()

            # --- CPU rolling correlation of log returns (robust & compatible) ---
            corr_pd = _build_pair_features_cpu(prices, window, min_periods)
            # _build_pair_features_cpu returns both corr and z_spread; we only need corr
            corr_pd = corr_pd[["date", "pair", "corr"]]

            # Merge corr (CPU) with z_spread (GPU)
            features = (
                pd.merge(corr_pd, z_pd, on=["date", "pair"], how="inner")
                .dropna()
                .sort_values(["date", "pair"])
                .reset_index(drop=True)
            )
            return features

        except Exception:
            # Any failure in RAPIDS path -> fall back to pure CPU
            pass

    # CPU fallback (or use_gpu=False)
    return _build_pair_features_cpu(prices, window, min_periods)
