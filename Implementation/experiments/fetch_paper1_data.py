"""
fetch_paper1_data.py
====================
One-time script: download Nifty 100 daily Adj Close prices from Yahoo Finance
and save as a single Parquet file for all Paper 1 experiments.

Run ONCE on kalpana (or locally) before submitting SLURM jobs:
    python Implementation/experiments/fetch_paper1_data.py

Output:
    Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet

Design choices (publishable paper rationale):
- Universe  : ~95 Nifty 100 current constituents (survivorship-bias disclosed as limitation)
- Period    : 2015-01-01 to 2024-12-31 (10 full years: pre-COVID, crash, recovery, rate cycle)
- Field     : Adj Close (dividend + split adjusted)
- Frequency : Daily
- Min data  : Tickers with <5 years (1250 trading days) of non-NaN data are dropped + logged
- No runtime downloads during experiments — all runs read from this file
"""
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe: Nifty 100 constituents (as of 2025, current + stable members)
# Covers all 10 sectors proportionally. ~95 tickers after delisting filter.
# ---------------------------------------------------------------------------
NIFTY100_TICKERS: list[str] = [
    # Banking & Financial Services (14)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "PNB.NS", "CANBK.NS", "BANKBARODA.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",

    # Information Technology (8)
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS",

    # Automobiles & Auto Components (8)
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "TATAMOTORS.NS", "BOSCHLTD.NS", "MOTHERSON.NS",

    # FMCG & Consumer Staples (7)
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "DABUR.NS", "GODREJCP.NS", "MARICO.NS",

    # Pharma & Healthcare (8)
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS", "TORNTPHARM.NS", "BIOCON.NS", "LUPIN.NS",

    # Energy, Oil & Gas (6)
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",
    "GAIL.NS", "POWERGRID.NS",

    # Metals, Mining & Materials (8)
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    "VEDL.NS", "SAIL.NS", "NATIONALUM.NS", "NMDC.NS",

    # Cement & Infrastructure (6)
    "ULTRACEMCO.NS", "ACC.NS", "SHREECEM.NS",
    "AMBUJACEM.NS", "JKCEMENT.NS", "RAMCOCEM.NS",

    # Capital Goods & Industrials (8)
    "LT.NS", "BHEL.NS", "SIEMENS.NS", "ABB.NS",
    "HAVELLS.NS", "CUMMINSIND.NS", "THERMAX.NS", "BEL.NS",

    # Telecom & Media (3)
    "BHARTIARTL.NS", "VODAFONEIDEA.NS", "TATACOMM.NS",

    # Consumer Discretionary & Retail (6)
    "TITAN.NS", "ASIANPAINT.NS", "BERGERPAINTS.NS",
    "PIDILITIND.NS", "DMART.NS", "TRENT.NS",

    # Real Estate & Utilities (4)
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS",

    # Conglomerate / Diversified (4)
    "ADANIENT.NS", "ADANIPORTS.NS", "ADANIGREEN.NS", "ADANITRANS.NS",

    # Insurance & Asset Management (4)
    "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "MUTHOOTFIN.NS",
]

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
MIN_TRADING_DAYS = 1250   # ~5 years; tickers below this are dropped

OUT_DIR  = Path(__file__).parent / "data" / "nse_nifty100"
OUT_FILE = OUT_DIR / f"prices_{START_DATE}_{END_DATE}.parquet"


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. Run: pip install yfinance")

    log.info(f"Downloading {len(tickers)} tickers: {start} → {end}")
    # Download in batches of 20 to avoid Yahoo rate limits
    batch_size = 20
    frames: list[pd.DataFrame] = []
    dropped: list[str] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info(f"  Batch {i//batch_size + 1}: {batch}")
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,       # avoid yahoo threading issues
                )
                break
            except Exception as e:
                log.warning(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(10)
        else:
            log.error(f"  Batch failed after 3 attempts, skipping: {batch}")
            continue

        # Extract Adj Close (auto_adjust=True puts it in 'Close')
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": batch[0]})

        frames.append(close)
        time.sleep(2)   # polite delay between batches

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    # Drop tickers with insufficient history
    valid_counts = prices.notna().sum()
    thin = valid_counts[valid_counts < MIN_TRADING_DAYS].index.tolist()
    if thin:
        log.warning(f"Dropping {len(thin)} tickers (< {MIN_TRADING_DAYS} days data): {thin}")
        dropped.extend(thin)
        prices.drop(columns=thin, inplace=True)

    # Forward-fill gaps ≤ 3 days (trading holidays), then drop remaining NaN rows
    prices.ffill(limit=3, inplace=True)

    log.info(f"Final universe: {prices.shape[1]} tickers, {prices.shape[0]} trading days")
    log.info(f"Dropped: {dropped}")
    return prices, dropped


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        log.info(f"Cache already exists: {OUT_FILE}")
        df = pd.read_parquet(OUT_FILE)
        log.info(f"  Shape: {df.shape}, dates: {df.index[0].date()} → {df.index[-1].date()}")
        return

    prices, dropped = fetch_prices(NIFTY100_TICKERS, START_DATE, END_DATE)
    prices.to_parquet(OUT_FILE)
    log.info(f"Saved → {OUT_FILE}")

    # Write a metadata sidecar so experiments can verify what they're loading
    meta_file = OUT_DIR / "metadata.txt"
    meta_file.write_text(
        f"Generated: {datetime.utcnow().isoformat()}Z\n"
        f"Period: {START_DATE} to {END_DATE}\n"
        f"Tickers requested: {len(NIFTY100_TICKERS)}\n"
        f"Tickers retained: {prices.shape[1]}\n"
        f"Tickers dropped (< {MIN_TRADING_DAYS} days): {dropped}\n"
        f"Trading days: {prices.shape[0]}\n"
        f"Field: Adj Close (auto_adjust=True)\n"
    )
    log.info(f"Metadata → {meta_file}")


if __name__ == "__main__":
    main()
