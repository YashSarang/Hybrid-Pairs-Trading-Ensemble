"""
Avellaneda & Lee (2010) PCA-OU Statistical Arbitrage Implementation

Paper: "Statistical Arbitrage in the U.S. Equities Market"
Authors: Marco Avellaneda, Jeong-Hyun Lee
Published: Quantitative Finance, 10(7), 761-782 (2010)

Methodology:
1. Decompose stock returns into common factors (PCA) and idiosyncratic residuals
2. Model residuals as mean-reverting OU processes
3. Trade on deviations of residuals from their long-run mean
4. Market-neutral by construction (zero exposure to common factors)

Key Innovation:
- Separates systematic risk (factors) from idiosyncratic mean-reversion
- Only trades idiosyncratic component → true statistical arbitrage
- PCA learns factor structure from data (no assumed factor model)

Claimed Results (US S&P 500):
- Sharpe ratio: 1.5-2.0 (gross, 2003-2007)
- Works best in high-volatility periods
- Market-neutral (beta ≈ 0 by construction)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple, Dict
import yfinance as yf
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PCAOUConfig:
    """Configuration for PCA-OU strategy"""
    n_factors: int = 15  # Number of PCA factors (paper uses 15 for S&P 500)
    formation_window: int = 252  # Days to estimate PCA (1 year)
    ou_estimation_window: int = 60  # Days to estimate OU parameters
    entry_threshold: float = 1.5  # Enter when |s_t| > 1.5 * σ_eq
    exit_threshold: float = 0.5  # Exit when |s_t| < 0.5 * σ_eq
    min_half_life: int = 5  # Minimum half-life (days) for tradeable residual
    max_half_life: int = 120  # Maximum half-life (days) - RELAXED from 60
    adf_pvalue_threshold: float = 0.10  # ADF test for stationarity - RELAXED from 0.05


class OUProcess:
    """Ornstein-Uhlenbeck process parameter estimation"""
    
    def __init__(self):
        self.kappa = None  # Mean-reversion speed
        self.mu = None  # Long-run mean
        self.sigma = None  # Volatility
        self.half_life = None
        
    def fit(self, series: np.ndarray, dt: float = 1.0) -> bool:
        """
        Fit OU process parameters using AR(1) estimation
        
        dS_t = κ(μ - S_t)dt + σdW_t
        
        Discretized: S_{t+1} - S_t = κμΔt - κS_tΔt + σ√Δt ε_t
        Or: ΔS_t = a + b*S_t + ε_t
        
        where: κ = -b/Δt, μ = -a/b, σ = std(ε)/√Δt
        """
        if len(series) < 20:
            return False
            
        # AR(1) regression: ΔS = a + b*S_{t-1} + ε
        S_t = series[:-1]
        delta_S = np.diff(series)
        
        # Add constant for intercept
        X = np.column_stack([np.ones(len(S_t)), S_t])
        
        try:
            model = OLS(delta_S, X).fit()
            a, b = model.params
            
            # OU parameters
            self.kappa = -b / dt
            self.mu = -a / b if b != 0 else 0
            self.sigma = np.std(model.resid) / np.sqrt(dt)
            
            # Half-life: t_{1/2} = ln(2) / κ
            if self.kappa > 0:
                self.half_life = np.log(2) / self.kappa
            else:
                self.half_life = np.inf
                
            # Check mean-reversion (κ > 0)
            return self.kappa > 0 and np.isfinite(self.half_life)
            
        except:
            return False
    
    def s_score(self, current_value: float) -> float:
        """
        Calculate s-score: standardized deviation from mean
        s = (S_t - μ) / σ_eq
        where σ_eq = σ / √(2κ) is equilibrium volatility
        """
        if self.kappa is None or self.kappa <= 0:
            return 0.0
        
        sigma_eq = self.sigma / np.sqrt(2 * self.kappa)
        return (current_value - self.mu) / sigma_eq if sigma_eq > 0 else 0.0


class PCAOUStrategy:
    """
    Avellaneda-Lee (2010) PCA-OU Statistical Arbitrage
    """
    
    def __init__(self, config: PCAOUConfig):
        self.config = config
        self.pca = None
        self.scaler = StandardScaler()
        self.ou_models: Dict[str, OUProcess] = {}
        self.factor_loadings = None
        self.mean_returns = None
        
    def fit_pca(self, returns: pd.DataFrame) -> None:
        """
        Fit PCA on stock returns to extract common factors
        
        Args:
            returns: DataFrame of stock returns (stocks x time)
        """
        # Standardize returns
        returns_scaled = self.scaler.fit_transform(returns.T)
        
        # Fit PCA
        self.pca = PCA(n_components=self.config.n_factors)
        self.pca.fit(returns_scaled)
        
        # Store factor loadings (stocks x factors)
        self.factor_loadings = self.pca.components_.T
        self.mean_returns = returns.mean(axis=1)
        
        print(f"PCA fitted: {self.config.n_factors} factors explain "
              f"{self.pca.explained_variance_ratio_.sum():.1%} of variance")
    
    def compute_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute idiosyncratic residuals after removing common factors
        
        r_i,t = α_i + Σ_k β_{i,k} f_{k,t} + ε_{i,t}
        
        Residual: ε_{i,t} = r_i,t - r̂_i,t
        """
        if self.pca is None:
            raise ValueError("Must fit PCA first")
        
        # Standardize
        returns_scaled = self.scaler.transform(returns.T)
        
        # Project onto factor space and reconstruct
        factors = self.pca.transform(returns_scaled)  # (time x n_factors)
        reconstructed = self.pca.inverse_transform(factors)  # (time x stocks)
        
        # Residuals = actual - reconstructed
        residuals = returns_scaled - reconstructed
        
        # Convert back to DataFrame
        residuals_df = pd.DataFrame(
            residuals.T,
            index=returns.index,
            columns=returns.columns
        )
        
        return residuals_df
    
    def fit_ou_models(self, residuals: pd.DataFrame) -> List[str]:
        """
        Fit OU process to each stock's idiosyncratic residuals
        
        Returns: List of tradeable stocks (mean-reverting residuals)
        """
        tradeable_stocks = []
        failed_reasons = {'adf_fail': 0, 'ou_fit_fail': 0, 'half_life_fail': 0}
        
        for stock in residuals.index:
            series = residuals.loc[stock].values
            
            # Skip if insufficient data
            if len(series) < self.config.ou_estimation_window:
                continue
            
            # Test stationarity (ADF test)
            try:
                adf_result = adfuller(series, maxlag=1)
                if adf_result[1] > self.config.adf_pvalue_threshold:
                    failed_reasons['adf_fail'] += 1
                    continue  # Not stationary
            except:
                continue
            
            # Fit OU process
            ou = OUProcess()
            if ou.fit(series[-self.config.ou_estimation_window:]):
                # Check half-life constraints
                if self.config.min_half_life <= ou.half_life <= self.config.max_half_life:
                    self.ou_models[stock] = ou
                    tradeable_stocks.append(stock)
                else:
                    failed_reasons['half_life_fail'] += 1
            else:
                failed_reasons['ou_fit_fail'] += 1
        
        print(f"Fitted OU models for {len(tradeable_stocks)} / {len(residuals)} stocks")
        print(f"  Failed ADF test: {failed_reasons['adf_fail']}")
        print(f"  Failed OU fit: {failed_reasons['ou_fit_fail']}")
        print(f"  Failed half-life: {failed_reasons['half_life_fail']}")
        return tradeable_stocks
    
    def generate_signals(self, current_residuals: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on current residual s-scores
        
        Signal logic:
        - s > +entry_threshold → SHORT (residual too high, expect reversion)
        - s < -entry_threshold → LONG (residual too low, expect reversion)
        - |s| < exit_threshold → EXIT
        """
        signals = []
        
        for stock in self.ou_models.keys():
            if stock not in current_residuals.index:
                continue
            
            residual_value = current_residuals[stock]
            ou = self.ou_models[stock]
            
            s = ou.s_score(residual_value)
            
            # Determine signal
            if s > self.config.entry_threshold:
                signal = -1  # SHORT
            elif s < -self.config.entry_threshold:
                signal = +1  # LONG
            elif abs(s) < self.config.exit_threshold:
                signal = 0  # EXIT
            else:
                signal = None  # HOLD
            
            if signal is not None:
                signals.append({
                    'stock': stock,
                    's_score': s,
                    'signal': signal,
                    'residual': residual_value,
                    'half_life': ou.half_life
                })
        
        return pd.DataFrame(signals)
    
    def backtest(self, prices: pd.DataFrame, start_date: str, end_date: str) -> Dict:
        """
        Backtest PCA-OU strategy
        
        Args:
            prices: DataFrame of stock prices (time x stocks)
            start_date: Start of OOS test period
            end_date: End of OOS test period
        
        Returns: Performance metrics dictionary
        """
        # Compute returns
        returns = prices.pct_change().dropna()
        
        # Split into formation and trading periods
        formation_end = pd.to_datetime(start_date) - timedelta(days=1)
        formation_start = formation_end - timedelta(days=self.config.formation_window)
        
        # Formation period: fit PCA
        formation_returns = returns.loc[formation_start:formation_end].T
        self.fit_pca(formation_returns)
        
        # Trading period
        trading_returns = returns.loc[start_date:end_date]
        
        # Compute residuals for entire period (formation + trading)
        all_returns = returns.loc[formation_start:end_date].T
        all_residuals = self.compute_residuals(all_returns).T
        
        # Fit OU models on formation period residuals
        formation_residuals = all_residuals.loc[formation_start:formation_end].T
        tradeable_stocks = self.fit_ou_models(formation_residuals)
        
        if len(tradeable_stocks) == 0:
            return {'error': 'No tradeable stocks found'}
        
        # Generate signals day-by-day
        portfolio_returns = []
        positions = {}  # Current positions
        
        for date in trading_returns.index:
            current_residuals = all_residuals.loc[date]
            signals_df = self.generate_signals(current_residuals)
            
            # Update positions
            for _, row in signals_df.iterrows():
                stock = row['stock']
                signal = row['signal']
                
                if signal == 0:  # EXIT
                    if stock in positions:
                        del positions[stock]
                elif signal in [-1, +1]:  # ENTRY
                    positions[stock] = signal
            
            # Calculate portfolio return for this day
            if len(positions) > 0:
                daily_pnl = 0
                for stock, position in positions.items():
                    if stock in trading_returns.columns:
                        daily_pnl += position * trading_returns.loc[date, stock]
                
                # Equal-weight allocation
                portfolio_return = daily_pnl / len(positions)
                portfolio_returns.append(portfolio_return)
            else:
                portfolio_returns.append(0.0)
        
        # Calculate performance metrics
        portfolio_returns = np.array(portfolio_returns)
        
        metrics = {
            'total_return': np.prod(1 + portfolio_returns) - 1,
            'annualized_return': np.mean(portfolio_returns) * 252,
            'annualized_vol': np.std(portfolio_returns) * np.sqrt(252),
            'sharpe_ratio': (np.mean(portfolio_returns) / np.std(portfolio_returns)) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0,
            'max_drawdown': self._max_drawdown(portfolio_returns),
            'n_tradeable_stocks': len(tradeable_stocks),
            'avg_positions': np.mean([len(positions)]) if len(portfolio_returns) > 0 else 0,
            'pca_variance_explained': self.pca.explained_variance_ratio_.sum(),
            'daily_returns': portfolio_returns.tolist()
        }
        
        return metrics
    
    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())


def run_avellaneda_lee_reproduction():
    """
    Reproduce Avellaneda & Lee (2010) on NSE data
    """
    print("="*70)
    print("Avellaneda & Lee (2010) PCA-OU Reproduction on NSE")
    print("="*70)
    
    # NSE 35 stocks (corrected tickers)
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 'SBIN.NS', 'BHARTIARTL.NS',
        'AXISBANK.NS', 'ITC.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
        'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS',
        'SUNPHARMA.NS', 'TECHM.NS', 'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS',
        'M&M.NS', 'TATASTEEL.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'IOC.NS',
        'BPCL.NS', 'GRASIM.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'DIVISLAB.NS'
    ]
    
    # Download data
    print("\nDownloading NSE data...")
    data = yf.download(tickers, start='2019-01-01', end='2025-12-31', progress=False)
    
    # Handle yfinance format (MultiIndex with 'Close' not 'Adj Close')
    if isinstance(data.columns, pd.MultiIndex):
        # Try 'Close' column (NSE doesn't have Adj Close)
        prices = data['Close']
    else:
        prices = data
    
    prices = prices.dropna(axis=1, how='all')  # Drop columns with all NaN
    prices = prices.dropna()  # Drop rows with any NaN
    
    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Configuration (paper uses 15 factors for S&P 500)
    config = PCAOUConfig(
        n_factors=10,  # Fewer stocks (35 vs 500), so fewer factors
        formation_window=252,
        ou_estimation_window=60,
        entry_threshold=1.5,
        exit_threshold=0.5,
        min_half_life=5,
        max_half_life=60
    )
    
    strategy = PCAOUStrategy(config)
    
    # Backtest on multiple OOS periods
    oos_periods = [
        ('2020-01-01', '2020-12-31'),
        ('2021-01-01', '2021-12-31'),
        ('2022-01-01', '2022-12-31'),
        ('2023-01-01', '2023-12-31'),
        ('2024-01-01', '2024-12-31'),
    ]
    
    results = []
    
    for start, end in oos_periods:
        print(f"\n{'='*70}")
        print(f"Testing period: {start} to {end}")
        print(f"{'='*70}")
        
        metrics = strategy.backtest(prices, start, end)
        
        if 'error' not in metrics:
            print(f"\nResults:")
            print(f"  Total Return: {metrics['total_return']:.2%}")
            print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
            print(f"  Annualized Vol: {metrics['annualized_vol']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  PCA Variance Explained: {metrics['pca_variance_explained']:.1%}")
            print(f"  Tradeable Stocks: {metrics['n_tradeable_stocks']}")
            
            results.append({
                'period': f"{start} to {end}",
                **metrics
            })
        else:
            print(f"  ERROR: {metrics['error']}")
    
    # Aggregate results
    if results:
        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS ACROSS ALL PERIODS")
        print(f"{'='*70}")
        
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_return = np.mean([r['annualized_return'] for r in results])
        avg_vol = np.mean([r['annualized_vol'] for r in results])
        
        print(f"\nAverage Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"Average Annualized Return: {avg_return:.2%}")
        print(f"Average Annualized Vol: {avg_vol:.2%}")
        
        # Compare to claimed results
        print(f"\n{'='*70}")
        print("COMPARISON TO PAPER CLAIMS")
        print(f"{'='*70}")
        print(f"\nAvellaneda & Lee (2010) on S&P 500:")
        print(f"  Claimed Sharpe Ratio: 1.5 - 2.0")
        print(f"  Claimed Period: 2003-2007 (US developed market)")
        print(f"\nOur Results on NSE 35 stocks:")
        print(f"  Actual Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"  Period: 2020-2024 (Indian emerging market)")
        
        if avg_sharpe >= 0.5:
            print(f"\n✅ METHOD WORKS on NSE (Sharpe > 0.5)")
        elif avg_sharpe >= 0.0:
            print(f"\n⚠️  PARTIAL SUCCESS (Sharpe positive but < 0.5)")
        else:
            print(f"\n❌ METHOD FAILED on NSE (negative Sharpe)")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_json('results.json', indent=2)
    print(f"\n✅ Results saved to results.json")
    
    return results


if __name__ == "__main__":
    results = run_avellaneda_lee_reproduction()
