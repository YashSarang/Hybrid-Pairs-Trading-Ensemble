"""Core Pair Selection Module.

This module re-exports components from:
- selectors_base (Pair, PairScore, PairSelector abstract base)
- selectors_statistical (Correlation, Distance, Cointegration, CombinedCriteria)
- selectors_ml (MLSelector, LSTMSelector, TransformerSelector, GNNSelector)
"""
from __future__ import annotations

# Re-export all classes
from .selectors_base import Pair, PairScore, PairSelector
from .selectors_statistical import (
    CorrelationSelector,
    DistanceSelector,
    CointegrationSelector,
    CombinedCriteriaSelector
)
from .selectors_ml import (
    TrivialSelectorModel,
    MLSelector,
    LSTMSelector,
    TransformerSelector,
    GNNSelector
)

__all__ = [
    "Pair",
    "PairScore",
    "PairSelector",
    "CorrelationSelector",
    "DistanceSelector",
    "CointegrationSelector",
    "CombinedCriteriaSelector",
    "TrivialSelectorModel",
    "MLSelector",
    "LSTMSelector",
    "TransformerSelector",
    "GNNSelector"
]
