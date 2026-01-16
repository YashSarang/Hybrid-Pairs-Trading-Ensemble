"""Stage 2 (Entry/Exit) models for the Pairs Trading app.

This module implements trading signal generation models that determine when to enter
and exit pairs trading positions. Each model applies different mathematical frameworks
to identify mean reversion opportunities in price spreads.

Academic Foundations:
- Mean Reversion Theory: Statistical tendency of prices to return to historical means
- Ornstein-Uhlenbeck Process: Mathematical model for mean-reverting stochastic processes
- Kalman Filtering: Optimal estimation theory for time-varying parameters
- Z-Score Methodology: Standard statistical approach for outlier detection

Signal Convention:
All models return discrete signals in {+1, 0, -1} format:
- +1: Long stock A, Short stock B (spread expected to narrow)
- -1: Short stock A, Long stock B (spread expected to widen)  
-  0: No position (neutral/exit signal)

Includes:
- Base class: EntryExitModel with standardized interface
- Implementations:
  * ZScoreThreshold: Classic mean-reversion bands on spread z-scores
  * OUThreshold: Ornstein-Uhlenbeck inspired thresholding with estimated parameters
  * KalmanHedge: Dynamic hedge ratio estimation (placeholder implementation)

Key Features:
- Pure signal generation without portfolio management concerns
- Configurable parameters with literature-backed defaults
- Robust error handling and fallback mechanisms
- Support for both unit and dynamic hedge ratios

Sources:
- Uhlenbeck & Ornstein (1930): "On the theory of Brownian motion"
- Kalman (1960): "A new approach to linear filtering and prediction problems"
- Standard statistical literature for z-score methodology

Notes:
- Pure signal generation here; no portfolio sizing, costs, or stop rules
- Spread definition defaults to A - B (unit hedge) for simplicity and robustness
- For dynamic hedging, use KalmanHedge or extend with OLS beta estimation
- All models include generate_signals() method for consistency with prediction engine
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm  # used for OLS in KalmanHedge placeholder
except Exception:  # pragma: no cover
    sm = None

# ---------------------------------------------
# Base
# ---------------------------------------------


class EntryExitModel:
    """Abstract entry/exit model producing discrete trading signals."""
    name: str = "base"

    def fit(self, a: pd.Series, b: pd.Series) -> "EntryExitModel":
        """Optional pre-fit step (e.g., estimate parameters)."""
        return self

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Compute a signal time-series in {+1, 0, -1} indexed like inputs."""
        raise NotImplementedError

    def generate_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Generate signals (alias for trade_signals for consistency)."""
        return self.trade_signals(a, b)


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def _zscore(x: pd.Series, lookback: int) -> pd.Series:
    m = x.rolling(lookback).mean()
    s = x.rolling(lookback).std(ddof=0)
    return (x - m) / (s + 1e-9)


def _rolling_beta(a: pd.Series, b: pd.Series, lookback: int = 100) -> pd.Series:
    """Rolling OLS beta of A on B (no intercept). Fallback to 1.0 when insufficient data."""
    cov = a.rolling(lookback).cov(b)
    var = b.rolling(lookback).var()
    beta = cov / (var + 1e-12)
    return beta.replace([np.inf, -np.inf], np.nan).fillna(1.0)


# ---------------------------------------------
# Implementations
# ---------------------------------------------

@dataclass
class ZScoreThreshold(EntryExitModel):
    """Mean-reversion bands on a simple spread.

    Parameters
    ----------
    lookback : int
        Rolling window for mean/std of spread (default 60 bars).
    entry_z : float
        Enter when |z| >= entry_z.
    exit_z : float
        Flatten when |z| < exit_z.
    hedge : str
        'unit' => spread = A - B
        'ols'  => spread = A - beta*B (rolling beta)
    beta_lb : int
        Lookback for rolling beta if hedge='ols'.
    """
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    hedge: str = "unit"  # or 'ols'
    beta_lb: int = 100

    name: str = "Mean Reversion (±2σ)"

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if self.hedge == "ols":
            beta = _rolling_beta(a, b, self.beta_lb)
            spread = a - beta * b
        else:
            spread = a - b
        z = _zscore(spread, self.lookback)
        sig = pd.Series(0, index=spread.index)
        sig[z > self.entry_z] = -1  # spread high => short A, long B
        sig[z < -self.entry_z] = +1
        sig[(z.abs() < self.exit_z)] = 0
        return sig.ffill().fillna(0).astype(int)


@dataclass
class OUThreshold(EntryExitModel):
    """OU/AR(1)-inspired thresholding.

    We estimate a reversion speed via AR(1) on demeaned spread and
    scale deviations by that speed. Thresholds are set in std units
    of this metric.
    """
    lookback: int = 252
    entry_k: float = 1.5
    exit_k: float = 0.2
    hedge: str = "unit"  # or 'ols'
    beta_lb: int = 100

    name: str = "OU Model"

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        if self.hedge == "ols":
            beta = _rolling_beta(a, b, self.beta_lb)
            spread = a - beta * b
        else:
            spread = a - b
        mu = spread.rolling(self.lookback).mean()
        x = (spread - mu).dropna()
        if len(x) < 10:
            return pd.Series(0, index=spread.index)
        x_lag = x.shift(1).dropna()
        y = x.loc[x_lag.index]
        var = x_lag.var()
        phi = (x_lag.cov(y) / (var + 1e-12)) if var != 0 else 0.0
        halflife = -1 / math.log(abs(phi)) if 1e-6 < abs(phi) < 1 else np.inf
        k = 1 / halflife if halflife != np.inf else 0.0
        metric = (k * (spread - mu)).fillna(0)
        s = metric.rolling(
            self.lookback // 2).std(ddof=0).bfill().fillna(0)
        sig = pd.Series(0, index=spread.index)
        sig[metric > self.entry_k * s] = -1
        sig[metric < -self.entry_k * s] = +1
        sig[(metric.abs() < self.exit_k * s)] = 0
        return sig.ffill().fillna(0).astype(int)


@dataclass
class KalmanHedge(EntryExitModel):
    """Placeholder for a true Kalman Filter hedge.

    We approximate with a rolling-OLS beta hedge and z-score bands.
    Replace with a proper state-space model when ready.
    """
    z_lb: int = 60
    beta_lb: int = 100

    name: str = "Kalman Hedge (placeholder)"

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        # Use rolling OLS beta as a proxy for a time-varying hedge ratio
        beta = _rolling_beta(a, b, self.beta_lb)
        spread = a - beta * b
        z = _zscore(spread, self.z_lb)
        sig = pd.Series(0, index=a.index)
        sig[z > 2.0] = -1
        sig[z < -2.0] = +1
        sig[z.abs() < 0.5] = 0
        return sig.ffill().fillna(0).astype(int)
