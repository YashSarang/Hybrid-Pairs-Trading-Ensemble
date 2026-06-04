"""Stage 2 (Entry/Exit) models using Reinforcement Learning.

This module implements a Stage 2 signal model based on Proximal Policy Optimization (PPO).
It includes a Gymnasium environment for a price spread and an RLSignal wrapper
that fits into the existing Stage 2 architecture.

Dependencies:
- stable-baselines3
- gymnasium
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None
import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

from .backtest import IndianCosts
from .entry import EntryExitModel, ZScoreThreshold, _zscore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class PairsTradingEnv(gym.Env):
    """Gymnasium environment for a single stock pair spread."""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: pd.DataFrame,
        spread_returns: pd.Series,
        cost_frac: float = 0.002,  # 20 bps round-trip approx
        initial_capital: float = 100000.0,
    ):
        super().__init__()
        self.features = features.values.astype(np.float32)
        self.returns = spread_returns.values.astype(np.float32)
        self.cost_frac = float(cost_frac)
        self.initial_capital = float(initial_capital)

        # Observation space: 11 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32
        )

        # Action space: 0: Neutral, 1: Long Spread (A-B), 2: Short Spread (B-A)
        # Remapped to signals: {0 -> 0, 1 -> 1, 2 -> -1}
        self.action_space = spaces.Discrete(3)

        self._current_tick = 0
        self._max_ticks = len(features) - 1
        self._position = 0  # current signal: {-1, 0, 1}
        self._capital = initial_capital
        self._equity_curve = []

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._current_tick = 0
        self._position = 0
        self._capital = self.initial_capital
        self._equity_curve = [self.initial_capital]
        return self.features[self._current_tick], {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Map action {0, 1, 2} -> signal {0, 1, -1}
        new_position = 0
        if action == 1: new_position = 1
        elif action == 2: new_position = -1

        # 1. Compute return from previous position
        ret = self.returns[self._current_tick]
        pnl = self._position * ret * self._capital

        # 2. Compute transaction costs if position changed
        cost = abs(new_position - self._position) * self.cost_frac * self._capital

        # 3. Update state
        # Scale reward to percentage points to prevent exploding gradients in PPO
        # e.g., a 1% raw return (pnl/capital = 0.01) becomes a reward of 1.0
        pct_return = (pnl - cost) / self._capital
        reward = float(np.clip(pct_return * 100.0, -10.0, 10.0))
        
        self._capital = max(1.0, self._capital + (pnl - cost))
        self._position = new_position
        self._current_tick += 1
        self._equity_curve.append(self._capital)

        done = self._current_tick >= self._max_ticks
        truncated = False

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self.features[self._current_tick]

        info = {
            "capital": self._capital,
            "position": self._position,
            "pnl": pnl,
            "cost": cost
        }

        return obs, float(reward), done, truncated, info

# ---------------------------------------------------------------------------
# RL Signal Model
# ---------------------------------------------------------------------------

@dataclass
class RLSignal(EntryExitModel):
    """Reinforcement Learning (PPO) signal model.

    Trains a PPO agent on the training window of a pair and uses it
    to predict signals on the test window.
    """
    lookback: int = 60
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    ent_coef: float = 0.01
    cost_frac: float = 0.0015  # 15 bps (default IndianCosts round-trip)

    name: str = "RL Signal (PPO)"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._model = None
        self._costs = IndianCosts()

    def _build_features(self, a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """Standard 11 features (same as MLSignal)."""
        idx = a.index.intersection(b.index)
        a = a.reindex(idx).ffill()
        b = b.reindex(idx).ffill()

        spread = a - b
        z = _zscore(spread, self.lookback)
        vel = z.diff(1)
        acc = vel.diff(1)

        r_a = np.log(a / a.shift(20).replace(0, np.nan)).fillna(0)
        r_b = np.log(b / b.shift(20).replace(0, np.nan)).fillna(0)

        std_a = a.rolling(self.lookback).std(ddof=0)
        std_b = b.rolling(self.lookback).std(ddof=0)

        df = pd.DataFrame(
            {
                "spread_z": z,
                "z_lag5": z.shift(5),
                "z_lag20": z.shift(20),
                "velocity": vel,
                "acceleration": acc,
                "abs_z": z.abs(),
                "corr_20": a.rolling(20).corr(b),
                "corr_60": a.rolling(60).corr(b),
                "vol_ratio": std_a / (std_b + 1e-9),
                "momentum_a": r_a,
                "momentum_b": r_b,
            },
        index=idx,
        )
        # Sanitize: replace infs, nans, and clip to 10 standard deviations
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df.clip(-10, 10)

    def fit(self, a: pd.Series, b: pd.Series) -> "RLSignal":
        """Train the PPO agent on the training window."""
        if not _HAS_SB3:
            log.warning("RLSignal.fit: stable-baselines3 not installed. Falling back.")
            return self

        features = self._build_features(a, b)
        # Returns: (spread[t+1] / spread[t]) - 1 approx by diff/shift
        # Using percentage change of the assets relative to each other
        # Returns: sanitize to avoid inf from 0-prices
        r_a = a.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        r_b = b.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        spread_returns = (r_a - r_b).clip(-0.2, 0.2)  # clip 20% daily move outliers

        # Environment setup
        env = PairsTradingEnv(
            features=features,
            spread_returns=spread_returns,
            cost_frac=self.cost_frac / 2.0  # single-leg cost for step
        )

        try:
            self._model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                ent_coef=self.ent_coef,
                verbose=0,
                seed=42,
                device="cpu"
            )
            self._model.learn(total_timesteps=self.total_timesteps)
        except Exception as exc:
            log.warning(f"RLSignal.fit: training failed ({exc}). Falling back.")
            self._model = None

        return self

    def trade_signals(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Infer signals using the trained PPO policy."""
        if self._model is None:
            # Fallback to OU if RL failed or SB3 not present
            return ZScoreThreshold(lookback=self.lookback).trade_signals(a, b)

        features = self._build_features(a, b)
        obs = features.values.astype(np.float32)

        signals = []
        for i in range(len(obs)):
            action, _states = self._model.predict(obs[i], deterministic=True)
            # Map action {0, 1, 2} -> signal {0, 1, -1}
            if action == 1: signals.append(1)
            elif action == 2: signals.append(-1)
            else: signals.append(0)

        return pd.Series(signals, index=features.index, dtype=int)
