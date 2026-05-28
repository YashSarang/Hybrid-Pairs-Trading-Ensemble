#!/usr/bin/env python3
"""
fetch_market_data.py

Download historical price data for all 4 markets (India, US, Brazil, UK) using yfinance.
Saves to Parquet cache under experimental-ablation/data/{market}/.

Usage:
    python fetch_market_data.py --markets india us brazil uk
    python fetch_market_data.py --market us  # Single market
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_config(market: str) -> Dict:
    """Load market config YAML."""
    config_path = Path(__file__).parent.parent / "configs" / f"{market}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker."""
    try:
        print(f"  Fetching {ticker}...", end="", flush=True)
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            show_errors=False
        )
        
        if data.empty:
            print(f" ❌ No data")
            return None
        
        # Extract Close price
        if isinstance(data.columns, pd.MultiIndex):
            # yfinance sometimes returns MultiIndex for single ticker
            close = data[('Close', ticker)] if ('Close', ticker) in data.columns else data['Close'].iloc[:, 0]
        else:
            close = data['Close']
        
        close = close.dropna()
        
        # Data quality check
        pct_available = len(close) / len(pd.date_range(start_date, end_date, freq='D')) * 100
        
        if pct_available < 50:
            print(f" ⚠️  Only {pct_available:.1f}% data available")
            return None
        
        print(f" ✅ {len(close)} days ({pct_available:.1f}%)")
        return pd.DataFrame({ticker: close})
    
    except Exception as e:
        print(f" ❌ Error: {str(e)[:50]}")
        return None

def fetch_market_data(market: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch all tickers for a market and return combined DataFrame.
    
    Returns:
        (prices_df, failed_tickers)
    """
    config = load_config(market)
    tickers = config['universe']['tickers']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    cache_dir = Path(__file__).parent.parent.parent / config['data']['cache_dir']
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"prices_{start_date}_{end_date}.parquet"
    
    # Check cache
    if cache_file.exists() and not force_refresh:
        print(f"\n✅ Cache found: {cache_file}")
        print(f"   Use --force to refresh\n")
        df = pd.read_parquet(cache_file)
        return df, []
    
    print(f"\n📊 Fetching {len(tickers)} tickers for {config['market']['name']}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Cache: {cache_file}\n")
    
    all_data = []
    failed = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}]", end=" ")
        df = fetch_ticker_data(ticker, start_date, end_date)
        
        if df is not None:
            all_data.append(df)
        else:
            failed.append(ticker)
    
    if not all_data:
        raise ValueError(f"❌ No data fetched for {market}!")
    
    # Combine all tickers into single DataFrame
    prices = pd.concat(all_data, axis=1)
    prices.index.name = 'Date'
    
    # Forward-fill missing values (holidays, etc.)
    prices = prices.ffill()
    
    # Final quality check
    coverage = (prices.notna().sum() / len(prices) * 100).mean()
    
    print(f"\n✅ Combined DataFrame: {prices.shape[0]} days × {prices.shape[1]} tickers")
    print(f"   Average coverage: {coverage:.1f}%")
    print(f"   Failed tickers: {len(failed)}")
    
    if failed:
        print(f"   ⚠️  {', '.join(failed)}")
    
    # Save to Parquet
    prices.to_parquet(cache_file)
    print(f"\n💾 Saved to {cache_file}\n")
    
    return prices, failed

def main():
    parser = argparse.ArgumentParser(description="Fetch multi-market price data")
    parser.add_argument(
        '--markets',
        nargs='+',
        choices=['india', 'us', 'brazil', 'uk'],
        help='Markets to fetch (space-separated)'
    )
    parser.add_argument(
        '--market',
        choices=['india', 'us', 'brazil', 'uk'],
        help='Single market to fetch'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if cache exists'
    )
    
    args = parser.parse_args()
    
    # Determine markets to fetch
    if args.market:
        markets = [args.market]
    elif args.markets:
        markets = args.markets
    else:
        markets = ['india', 'us', 'brazil', 'uk']
    
    print("=" * 60)
    print("Multi-Market Data Fetcher")
    print("=" * 60)
    
    results = {}
    
    for market in markets:
        try:
            df, failed = fetch_market_data(market, force_refresh=args.force)
            results[market] = {
                'success': True,
                'tickers': df.shape[1],
                'days': df.shape[0],
                'failed': len(failed)
            }
        except Exception as e:
            print(f"\n❌ {market.upper()} failed: {e}\n")
            results[market] = {'success': False, 'error': str(e)}
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for market, res in results.items():
        status = "✅" if res.get('success') else "❌"
        if res.get('success'):
            print(f"{status} {market.upper():8s} — {res['tickers']} tickers, {res['days']} days, {res['failed']} failed")
        else:
            print(f"{status} {market.upper():8s} — {res.get('error', 'Unknown error')}")
    
    print("=" * 60)
    print("Done! 🎯")

if __name__ == "__main__":
    main()
