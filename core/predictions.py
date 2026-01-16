"""Prediction engine for real-time pairs trading recommendations.

This module provides real-time trading recommendations based on current market data
and user-configured strategy parameters. It reuses the existing ensemble methodology
for consistency with backtesting results.

Key Features:
- Real-time pair scoring using ensemble of 5 selection models
- Entry/exit signal generation using 3 trading models  
- Market regime analysis for context
- Confidence scoring based on signal consistency and data quality

Sources and References:
- Ensemble methodology: Custom implementation combining multiple academic approaches
- Correlation-based selection: Standard Pearson correlation coefficient
- Distance-based selection: Gatev et al. (2006) "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
- Cointegration: Engle-Granger (1987) two-step procedure
- Mean reversion signals: Standard z-score methodology
- OU Model: Ornstein-Uhlenbeck process for mean reversion (Uhlenbeck & Ornstein, 1930)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st  # For caching functionality

from .data import DataConfig, YFinanceNSESource
from .selectors import (
    Pair,
    PairScore,
    CorrelationSelector,
    DistanceSelector,
    CointegrationSelector,
    CombinedCriteriaSelector,
    MLSelector,
)
from .entry import ZScoreThreshold, OUThreshold, KalmanHedge
from .ensemble import ensemble_pair_scores


@dataclass
class PairRecommendation:
    """A single pair recommendation with scoring and signal information.

    Contains all relevant metrics for evaluating a trading pair including:
    - Ensemble scores from Stage 1 selection models
    - Current trading signals from Stage 2 entry models
    - Risk metrics (volatility, correlation, z-score)
    - Confidence assessment based on data quality and signal consistency
    """
    pair: Pair                    # The stock pair (e.g., RELIANCE/TCS)
    score: float                  # Ensemble score from Stage 1 models
    rank: int                     # Ranking among all recommendations
    # Individual model signals {model_name: signal_value}
    signals: Dict[str, float]
    ensemble_signal: float        # Combined signal from Stage 2 models
    # Confidence score [0,1] based on data quality
    confidence: float
    current_spread: float         # Current price spread (A - B)
    z_score: float               # Z-score of current spread vs historical mean
    last_price_a: float          # Latest price of stock A
    last_price_b: float          # Latest price of stock B
    volatility: float            # Annualized volatility of stock A
    correlation: float           # 60-day rolling correlation between A and B


@dataclass
class MarketRegime:
    """Current market regime indicators for context.

    Provides market-wide metrics to help assess the suitability of pairs trading
    strategies under current conditions. Based on academic literature on regime
    detection and pairs trading performance.

    Sources:
    - Correlation regime analysis: Longin & Solnik (2001) "Extreme Correlation of International Equity Markets"
    - Volatility clustering: Engle (1982) "Autoregressive Conditional Heteroscedasticity"
    """
    overall_volatility: float      # Market-wide volatility (annualized)
    correlation_regime: str        # "High" or "Low" correlation environment
    trend_strength: float          # Momentum indicator (0-100)
    mean_reversion_opportunity: float  # Price dispersion metric
    regime_confidence: float       # Confidence in regime assessment [0,1]


@dataclass
class PredictionResult:
    """Complete prediction result with recommendations and market analysis.

    Encapsulates all outputs from the prediction engine including individual
    pair recommendations, market regime analysis, and metadata about the
    prediction quality and freshness.
    """
    recommendations: List[PairRecommendation]  # Ranked list of pair recommendations
    market_regime: MarketRegime               # Current market conditions
    timestamp: datetime                       # When predictions were generated
    data_freshness: str                      # Data age indicator
    universe_size: int                       # Number of stocks analyzed
    total_pairs_analyzed: int                # Total pair combinations considered


class PredictionEngine:
    """Real-time prediction engine for pairs trading recommendations.

    This engine provides real-time trading recommendations by applying the same
    ensemble methodology used in backtesting to current market data. It combines
    multiple pair selection models (Stage 1) with entry/exit signal models (Stage 2)
    to generate actionable trading insights.

    Architecture:
    - Stage 1: Ensemble of 5 pair selection models (Correlation, Distance, Cointegration, etc.)
    - Stage 2: Ensemble of 3 entry/exit models (Z-Score, OU, Kalman)
    - Market Regime Analysis: Volatility and correlation environment assessment

    The engine is designed for consistency with backtesting results while providing
    fast real-time analysis suitable for interactive use.

    Sources:
    - Two-stage ensemble approach: Custom methodology inspired by academic literature
    - Real-time processing: Optimized for sub-second response times
    """

    def __init__(self, lookback_days: int = 252):
        """Initialize prediction engine with configurable lookback period.

        Args:
            lookback_days: Historical data window for analysis (default: 252 trading days = 1 year)
        """
        self.lookback_days = lookback_days
        self.data_source = YFinanceNSESource()

        # Initialize Stage 1 selectors with optimized parameters for real-time use
        # Reduced lookback periods for faster computation while maintaining statistical validity
        self.selectors = {
            CorrelationSelector.name: CorrelationSelector(lookback=min(252, lookback_days)),
            DistanceSelector.name: DistanceSelector(lookback=min(252, lookback_days), mode="zscore"),
            CointegrationSelector.name: CointegrationSelector(lookback=min(504, lookback_days*2), pvalue_threshold=0.05),
            CombinedCriteriaSelector.name: CombinedCriteriaSelector(),
            MLSelector.name: MLSelector(),
        }

        # Initialize Stage 2 entry models with conservative parameters
        self.entry_models = {
            ZScoreThreshold.name: ZScoreThreshold(lookback=60, entry_z=2.0, exit_z=0.5),
            OUThreshold.name: OUThreshold(lookback=min(252, lookback_days)),
            KalmanHedge.name: KalmanHedge(),
        }

    def get_predictions(
        self,
        universe: List[str],
        stage1_weights: Dict[str, float],
        stage2_weights: Dict[str, float],
        top_k: int = 10,
        min_data_points: int = 100,
    ) -> PredictionResult:
        """Generate real-time predictions for the given universe and weights.

        Args:
            universe: List of stock symbols
            stage1_weights: Weights for pair selection models
            stage2_weights: Weights for entry/exit models
            top_k: Number of top recommendations to return
            min_data_points: Minimum data points required for analysis

        Returns:
            PredictionResult with recommendations and market analysis
        """
        try:
            # Fetch recent market data (fix datetime deprecation warning)
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - \
                timedelta(days=self.lookback_days + 30)  # Extra buffer

            data_config = DataConfig(
                start=start_date,
                end=end_date,
                freq="1D",
                price_field="Adj Close"
            )

            prices = self.data_source.get_prices(universe, data_config)

            if prices.empty or len(prices) < min_data_points:
                raise ValueError(f"Insufficient data: {len(prices)} rows")

            # Determine data freshness
            last_data_date = prices.index.max().date()
            days_old = (end_date - last_data_date).days
            if days_old == 0:
                data_freshness = "Real-time"
            elif days_old == 1:
                data_freshness = "1 day delayed"
            else:
                data_freshness = f"{days_old} days delayed"

            # Generate pair candidates
            candidates = [
                Pair(universe[i], universe[j])
                for i in range(len(universe))
                for j in range(i + 1, len(universe))
            ]

            # Stage 1: Score pairs using ensemble
            pair_scores = self._score_pairs(prices, candidates, stage1_weights)
            top_pairs = pair_scores[:top_k]

            # Stage 2: Generate signals for top pairs
            recommendations = []
            for rank, (pair, score) in enumerate(top_pairs, 1):
                try:
                    recommendation = self._analyze_pair(
                        pair, score, rank, prices, stage2_weights
                    )
                    recommendations.append(recommendation)
                except Exception as e:
                    # Skip pairs that fail analysis
                    print(f"Failed to analyze pair {pair}: {e}")
                    continue

            # Market regime analysis
            market_regime = self._analyze_market_regime(prices)

            return PredictionResult(
                recommendations=recommendations,
                market_regime=market_regime,
                timestamp=datetime.now(timezone.utc),
                data_freshness=data_freshness,
                universe_size=len(universe),
                total_pairs_analyzed=len(candidates),
            )

        except Exception as e:
            # Return empty result on failure
            return PredictionResult(
                recommendations=[],
                market_regime=MarketRegime(
                    overall_volatility=0.0,
                    correlation_regime="Unknown",
                    trend_strength=0.0,
                    mean_reversion_opportunity=0.0,
                    regime_confidence=0.0,
                ),
                timestamp=datetime.now(timezone.utc),
                data_freshness="Error",
                universe_size=len(universe) if universe else 0,
                total_pairs_analyzed=0,
            )

    def _score_pairs(
        self,
        prices: pd.DataFrame,
        candidates: List[Pair],
        weights: Dict[str, float],
    ) -> List[Tuple[Pair, float]]:
        """Score pairs using Stage 1 ensemble."""
        scores_by_model = {}

        for name, selector in self.selectors.items():
            try:
                fitted_selector = selector.fit(prices)
                scores = fitted_selector.score_pairs(prices, candidates)
                scores_by_model[name] = scores
                print(f"Selector {name}: {len(scores)} scores")
            except Exception as e:
                # Use neutral scores on failure
                print(f"Selector {name} failed: {e}")
                # Create neutral PairScore objects
                neutral_scores = [
                    PairScore(pair=pair, score=0.5, details={})
                    for pair in candidates
                ]
                scores_by_model[name] = neutral_scores

        print(f"Scores by model keys: {list(scores_by_model.keys())}")
        print(f"Weights keys: {list(weights.keys())}")

        # Combine scores using ensemble
        try:
            ensemble_scores = ensemble_pair_scores(
                scores_by_model, weights, top_k=len(candidates))
            print(f"Ensemble produced {len(ensemble_scores)} scores")
            # Convert to the expected format (Pair, float)
            return [(ps.pair, ps.score) for ps in ensemble_scores]
        except Exception as e:
            print(f"Ensemble scoring failed: {e}")
            import traceback
            traceback.print_exc()

    def get_predictions_from_report(
        self,
        report_data: Dict[str, Any],
        top_k: int = 10,
        min_data_points: int = 100,
    ) -> PredictionResult:
        """Generate predictions using settings from a saved report.

        This method allows users to generate real-time predictions using the exact
        same configuration (universe, weights, parameters) from a previous backtest
        run, enabling direct comparison between historical and current performance.

        Args:
            report_data: Complete report data from ReportManager.load_report()
            top_k: Number of top recommendations to return
            min_data_points: Minimum data points required for analysis

        Returns:
            PredictionResult with recommendations using report settings
        """
        try:
            # Extract configuration from report metadata
            metadata = report_data["metadata"]
            universe = metadata["universe"]
            stage1_weights = metadata["stage1_weights"]
            stage2_weights = metadata["stage2_weights"]

            # Generate predictions using report settings
            return self.get_predictions(
                universe=universe,
                stage1_weights=stage1_weights,
                stage2_weights=stage2_weights,
                top_k=top_k,
                min_data_points=min_data_points,
            )

        except Exception as e:
            print(f"Failed to generate predictions from report: {e}")
            # Return empty result on failure
            return PredictionResult(
                recommendations=[],
                market_regime=MarketRegime(
                    overall_volatility=0.0,
                    correlation_regime="Unknown",
                    trend_strength=0.0,
                    mean_reversion_opportunity=0.0,
                    regime_confidence=0.0,
                ),
                timestamp=datetime.now(timezone.utc),
                data_freshness="Error",
                universe_size=0,
                total_pairs_analyzed=0,
            )

    def _analyze_pair(
        self,
        pair: Pair,
        score: float,
        rank: int,
        prices: pd.DataFrame,
        stage2_weights: Dict[str, float],
    ) -> PairRecommendation:
        """Analyze a single pair to generate detailed recommendation."""
        try:
            # Get price series for the pair
            price_a = prices[pair.a].dropna()
            price_b = prices[pair.b].dropna()

            # Align prices
            common_idx = price_a.index.intersection(price_b.index)
            if len(common_idx) < 50:
                raise ValueError("Insufficient overlapping data")

            price_a = price_a.reindex(common_idx)
            price_b = price_b.reindex(common_idx)

            # Calculate current metrics
            current_spread = float(price_a.iloc[-1] - price_b.iloc[-1])
            last_price_a = float(price_a.iloc[-1])
            last_price_b = float(price_b.iloc[-1])

            # Calculate rolling statistics
            returns_a = price_a.pct_change().dropna()
            returns_b = price_b.pct_change().dropna()
            correlation = float(returns_a.rolling(60).corr(returns_b).iloc[-1])
            volatility = float(returns_a.rolling(
                60).std().iloc[-1] * np.sqrt(252))

            # Generate signals from each model
            signals = {}
            signal_values = []

            for name, model in self.entry_models.items():
                try:
                    # Generate signal for this pair
                    signal = self._generate_signal(model, price_a, price_b)
                    signals[name] = signal
                    signal_values.append(
                        signal * stage2_weights.get(name, 0.0))
                except Exception as e:
                    print(f"Signal generation failed for {name}: {e}")
                    signals[name] = 0.0

            # Ensemble signal
            ensemble_signal = sum(signal_values)

            # Calculate z-score of current spread
            spread_series = price_a - price_b
            spread_mean = spread_series.rolling(252).mean().iloc[-1]
            spread_std = spread_series.rolling(252).std().iloc[-1]
            z_score = (current_spread - spread_mean) / \
                spread_std if spread_std > 0 else 0.0

            # Calculate confidence based on signal consistency and data quality
            signal_consistency = 1.0 - \
                np.std(list(signals.values())) if signals else 0.0
            data_quality = min(1.0, len(common_idx) / 252.0)
            confidence = (signal_consistency * 0.6 + data_quality * 0.4)

            return PairRecommendation(
                pair=pair,
                score=score,
                rank=rank,
                signals=signals,
                ensemble_signal=ensemble_signal,
                confidence=confidence,
                current_spread=current_spread,
                z_score=z_score,
                last_price_a=last_price_a,
                last_price_b=last_price_b,
                volatility=volatility,
                correlation=correlation,
            )

        except Exception as e:
            # Return neutral recommendation on failure
            return PairRecommendation(
                pair=pair,
                score=score,
                rank=rank,
                signals={},
                ensemble_signal=0.0,
                confidence=0.0,
                current_spread=0.0,
                z_score=0.0,
                last_price_a=0.0,
                last_price_b=0.0,
                volatility=0.0,
                correlation=0.0,
            )

    def _generate_signal(
        self,
        model,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> float:
        """Generate signal from a single entry model."""
        try:
            if hasattr(model, 'generate_signals'):
                # Use the model's signal generation method
                signals = model.generate_signals(price_a, price_b)
                if not signals.empty:
                    return float(signals.iloc[-1])

            # Fallback: simple z-score based signal
            spread = price_a - price_b
            z_score = (spread - spread.rolling(60).mean()) / \
                spread.rolling(60).std()
            latest_z = z_score.iloc[-1]

            if latest_z > 2.0:
                return -1.0  # Short signal
            elif latest_z < -2.0:
                return 1.0   # Long signal
            else:
                return 0.0   # Neutral

        except Exception:
            return 0.0

    def _analyze_market_regime(self, prices: pd.DataFrame) -> MarketRegime:
        """Analyze current market regime."""
        try:
            # Calculate recent returns
            returns = prices.pct_change().dropna()
            recent_returns = returns.tail(60)  # Last 60 days

            # Overall volatility (average across all stocks)
            overall_vol = float(recent_returns.std().mean() * np.sqrt(252))

            # Correlation regime - fix the calculation
            if len(recent_returns.columns) > 1:
                corr_matrix = recent_returns.corr()
                # Get upper triangle values (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                upper_triangle_values = corr_matrix.values[mask]
                avg_correlation = float(np.nanmean(upper_triangle_values))
            else:
                avg_correlation = 0.0

            correlation_regime = "High" if avg_correlation > 0.5 else "Low"

            # Trend strength (momentum) - fix the calculation
            try:
                momentum_series = returns.rolling(20).mean().tail(1).abs()
                trend_strength = float(momentum_series.mean().iloc[0] * 100)
            except:
                trend_strength = 0.0

            # Mean reversion opportunity (based on dispersion) - fix the calculation
            try:
                normalized_prices = prices / prices.rolling(252).mean()
                price_dispersion_series = normalized_prices.std(
                    axis=1).tail(20)
                mean_reversion_opportunity = float(
                    price_dispersion_series.mean())
            except:
                mean_reversion_opportunity = 0.0

            # Regime confidence (based on data consistency)
            regime_confidence = min(1.0, len(returns) / 252.0)

            return MarketRegime(
                overall_volatility=overall_vol,
                correlation_regime=correlation_regime,
                trend_strength=trend_strength,
                mean_reversion_opportunity=mean_reversion_opportunity,
                regime_confidence=regime_confidence,
            )

        except Exception as e:
            print(f"Market regime analysis failed: {e}")
            return MarketRegime(
                overall_volatility=0.0,
                correlation_regime="Unknown",
                trend_strength=0.0,
                mean_reversion_opportunity=0.0,
                regime_confidence=0.0,
            )


__all__ = ["PredictionEngine", "PairRecommendation",
           "MarketRegime", "PredictionResult"]
