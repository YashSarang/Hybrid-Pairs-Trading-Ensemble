#!/usr/bin/env python3
"""
Download all required data for experiments and cache it locally.
Run this once before submitting jobs to avoid yfinance issues on compute nodes.
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

# Tickers from config.py
NSE_UNIVERSE = [
    # Banking & Financial Services (6)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS",
    # Information Technology (5)
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    # Automobiles & Components (5)
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    # FMCG & Consumer Staples (4)
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    # Pharma & Healthcare (4)
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    # Energy & Oil & Gas (4)
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",
    # Metals & Mining (4)
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    # Cement & Infrastructure (3)
    "ULTRACEMCO.NS", "ACC.NS", "SHREECEM.NS",
]

# Date ranges
MAIN_START = "2016-01-01"
MAIN_END = "2026-03-31"

# For hourly data (730 day limit)
HOURLY_END = datetime.now()
HOURLY_START = HOURLY_END - timedelta(days=700)

def download_ticker_individually(ticker, start, end, interval="1d"):
    """Download a single ticker with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            if not data.empty:
                return data
            time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  ✗ Failed {ticker}: {e}")
                return None
            time.sleep(2)
    return None

def download_daily_data():
    """Download daily data for all tickers, one by one."""
    print("=" * 60)
    print("Downloading DAILY data...")
    print(f"Period: {MAIN_START} to {MAIN_END}")
    print(f"Tickers: {len(NSE_UNIVERSE)}")
    print("=" * 60)
    
    all_data = {}
    failed = []
    
    for i, ticker in enumerate(NSE_UNIVERSE, 1):
        print(f"[{i}/{len(NSE_UNIVERSE)}] Downloading {ticker}...", end=" ")
        data = download_ticker_individually(ticker, MAIN_START, MAIN_END, "1d")
        
        if data is not None and not data.empty:
            # Store with ticker as key
            all_data[ticker] = data
            print(f"✓ ({len(data)} rows)")
        else:
            failed.append(ticker)
            print("✗ Failed")
        
        time.sleep(0.5)  # Rate limiting
    
    if not all_data:
        raise RuntimeError("No data downloaded successfully!")
    
    # Combine all tickers into a multi-index DataFrame
    combined = pd.concat(all_data, axis=1)
    
    # Save to parquet (better for MultiIndex)
    os.makedirs("data_cache", exist_ok=True)
    combined.to_parquet("data_cache/daily_prices.parquet")
    
    print(f"\n✓ Saved daily data: {combined.shape}")
    print(f"  File: data_cache/daily_prices.parquet")
    print(f"  Success: {len(all_data)}/{len(NSE_UNIVERSE)} tickers")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    
    return combined

def download_hourly_data():
    """Download hourly data for all tickers, one by one."""
    print("\n" + "=" * 60)
    print("Downloading HOURLY data...")
    print(f"Period: {HOURLY_START.date()} to {HOURLY_END.date()}")
    print(f"Tickers: {len(NSE_UNIVERSE)}")
    print("=" * 60)
    
    all_data = {}
    failed = []
    
    for i, ticker in enumerate(NSE_UNIVERSE, 1):
        print(f"[{i}/{len(NSE_UNIVERSE)}] Downloading {ticker}...", end=" ")
        data = download_ticker_individually(ticker, HOURLY_START, HOURLY_END, "1h")
        
        if data is not None and not data.empty:
            all_data[ticker] = data
            print(f"✓ ({len(data)} rows)")
        else:
            failed.append(ticker)
            print("✗ Failed")
        
        time.sleep(0.5)  # Rate limiting
    
    if not all_data:
        raise RuntimeError("No data downloaded successfully!")
    
    # Combine all tickers into a multi-index DataFrame
    combined = pd.concat(all_data, axis=1)
    
    # Save to parquet (better for MultiIndex)
    os.makedirs("data_cache", exist_ok=True)
    combined.to_parquet("data_cache/hourly_prices.parquet")
    
    print(f"\n✓ Saved hourly data: {combined.shape}")
    print(f"  File: data_cache/hourly_prices.parquet")
    print(f"  Success: {len(all_data)}/{len(NSE_UNIVERSE)} tickers")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    
    return combined

def main():
    print("\n" + "=" * 60)
    print("NSE Pairs Trading - Data Download")
    print("=" * 60)
    
    try:
        # Download daily data
        daily_data = download_daily_data()
        
        # Download hourly data
        hourly_data = download_hourly_data()
        
        print("\n" + "=" * 60)
        print("✓ Data download complete!")
        print("=" * 60)
        print("\nData files created:")
        print("  - data_cache/daily_prices.parquet")
        print("  - data_cache/hourly_prices.parquet")
        print("\nNote: Some tickers may have failed due to yfinance issues.")
        print("The experiments will work with the available tickers.")
        print("\nYou can now run the experiments.")
        
    except Exception as e:
        print(f"\n✗ Error downloading data: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try: pip install --upgrade yfinance")
        print("  3. Some tickers may be delisted - check NSE website")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
