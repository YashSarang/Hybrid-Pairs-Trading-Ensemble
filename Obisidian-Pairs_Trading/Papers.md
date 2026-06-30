---
cssclasses:
---
---
# Sources of Code
# Classical Papers
### 1. Co-Integration and Error Correction: Representation, Estimation, and Testing
1987 - 55k cites https://www.jstor.org/stable/1913236?seq=1

### 2. Optimal pairs trading: A stochastic control approach
2008 - 79 cites - https://ieeexplore.ieee.org/abstract/document/4586628
**OU process** based stochastic control problem for Portfolio optimisation.

### 3. Pairs Trading: Performance of a Relative Value Arbitrage Rule 
2006 - 32 cites  - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615
**Distance method**
#### 3.1 Does Simple Pairs Trading Still Work
2010 - Cites 439 - https://www.jstor.org/stable/pdf/25741293.pdf
**Extended Distance** maybe (check properly)

### 4. Statistical Arbitrage in the U.S. Equities Market
40 cites  2010 - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505
PCA-OU

### 5. Enhancing a Pairs Trading strategy with the application of Machine Learning
2020 - 54 cites
Pairs Selection by - PCA and Clustering 

#### Multi-Criteria Spread Selection Rules
To qualify as a tradable pair, the resulting spread of the assets must pass the following multi-criteria rules: 

- **Cointegration:** The constituents must be cointegrated. The framework typically uses the [Engle-Granger test](https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/spread_selection/cointegration_spread_selection.html) to confirm this condition.
- **Mean Reversion (Hurst Exponent):** The spread's Hurst exponent must indicate a clear mean-reverting character.
- **Divergence Constraints:** The spread must diverge and converge within convenient, historically reliable periods.
- **Frequency of Reversion:** The spread must cross its mean at least twelve times per year to guarantee ample trading opportunities

# Deep Learning Papers
### 6. Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500
Cites 584 - https://www.sciencedirect.com/science/article/pii/S0377221716308657
* Deep Learning
* Gradient Boost
* Random Forests
* Their Ensemble (equal weighted only)
*For each day from December 1992 until October 2015, all constituents are ranked according to their out-of-sample probability forecast in descending order. The top _k_ stocks are bought and the flop _k_ stocks sold short.
### 7. Deep learning with long short-term memory networks for financial market predictions
2018 - 2150 cites 
LSTM Implementation Direct - No Pairs trading

### 8. A Transformer-based Framework for Multivariate Time Series Representation Learning
2020 - 1941 cites
Transformer Implementation - No Pairs trading involved

### 9. Exploring Graph Neural Networks for Stock Market Predictions with Rolling Window Analysis
2019 - Cites 171
GNN Implementation - No Pairs trading involved

### 10. Proximal Policy Optimization Algorithms
2017 - 41.5k cites
RL based implementation - No Pairs Trading involved

---

# Time-series based - Execution Engine
#### Pairs trading with time-series deep learning models
2 cites - Dec 2025- https://www.sciencedirect.com/science/article/pii/S2405918826000024 
Pairs Selection - 
* Co-integration
Exec Engines - 
* Autoformer 
* iTransformer
* Scaleformer
* Chronos


# Claude Generated IG

#### Enhancing Pairs Trading with Graph Neural Network-Based Pair Selection 
0 cites - https://ieeexplore.ieee.org/document/11413258/

### A hierarchical deep learning framework for pair trading with attention and graph networks
2025 - 2 cites
https://www.sciencedirect.com/science/article/pii/S0957417425021153

## Latest

#### A Hybrid Framework for Pairs Trading: Integrating Cointegration Analysis with Ensemble Learning for Enhanced Signal
2026 - 0 cites
https://dl.acm.org/doi/10.1145/3800000.3800094


# Extra Irrelevant 

#### Hardware Accelerator for Engle-Granger Cointegration in Pairs Trading
3 cites - 2020 https://ieeexplore.ieee.org/document/9180586 