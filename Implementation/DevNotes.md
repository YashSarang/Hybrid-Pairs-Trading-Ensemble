Run this in your terminal when ready — it uses the optimized settings (50k sequence cap + batch_size=256):
                                                                                                                                                              
  cd "C:\Code\Hybrid-Pairs-Trading-Ensemble"                                                                                                                    .venv/Scripts/activate                                                                                                                                      
  python experiments/walk_forward.py --mode full --s2 ou_only 
  


This repository is for my Master's thesis, I am pursuing a M.S. by Research. So I have to ensure the standard of work is upto the mark and that evetual whatever work I will present at the end will be worthy of being called a Thesis.

The aim is to build a Hybrid Pairs Trading Ensemble System.


High Level Architecture:

1. Data Ingestion (optional)
    1.1. Market Data
    1.2. Alternative / Fundamental Data
    1.3. Data Storage

2. Data Processing (important)
    2.1. Data Cleaning
    2.2. Normalization / Alignment
    2.3. Spread Construction
    2.4. Feature Engineering
    2.5. Regime Detection
    2.6. Data Storage

3. Candidate Pair Generation
    3.1. Universe Filtering
    3.2. Similarity / Sector Screening
    3.3. Candidate Pair Creation

4. Pairs Selection Models Ensemble 
    4.1. Statistical Models
        4.1.1 Correlation Selector (Rolling Pearson Correlation)
        4.1.2 Distance Selector (Gatev et al. 2006)
        4.1.3 Cointegration Selector (Engle-Granger Test + ADF Test)
        4.1.4 Combined Criteria Selector (Cointegration + Hurst Exponent)    
    4.2. Machine Learning Models
        4.2.1 XGBoost
        4.2.2 Gradient Boosting
        
    4.3. Deep Learning Models
        4.3.1 Graph Neural Networks
        4.3.2 LSTM/BiLSTM
        4.3.3 Transformers
    4.4. Ensemble Scoring / Ranking (optional)
    
5. Signal Generation / Trading Models
    5.1. Statistical Signal Models
    5.2. Machine Learning Signal Models
    5.3. Reinforcement Learning Models
    5.4. Signal Ensemble / Meta-Decision Layer

6. Execution & Trade Management
    6.1. Entry / Exit Logic
    6.2. Position Sizing
    6.3. Transaction Cost & Slippage Handling
    6.4. Order Execution Logic

7. Portfolio & Risk Management
    7.1. Portfolio Optimization
    7.2. Exposure Control
    7.3. Risk Limits / Stop Rules
    7.4. Performance Attribution

8. Backtesting & Evaluation
    8.1. Historical Simulation
    8.2. Walk-Forward Validation
    8.3. Stress Testing
    8.4. Benchmark Comparison

9. Deployment & Monitoring
    9.1. Production Deployment
    9.2. Live Monitoring
    9.3. Drift Detection
    9.4. Retraining / Maintenance



