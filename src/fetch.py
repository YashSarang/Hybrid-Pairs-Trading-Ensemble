"""
fetch.py  – Robust Yahoo Finance downloader with smart cache
===========================================================
2025‑07‑30 patch
----------------
* Explicitly sets ``auto_adjust=False`` to maintain presence of **Adj Close**
  (YFinance 0.2.37+ switched default to True).
* Gracefully falls back to **Close** when *Adj Close* is absent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, List, Optional, Union
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_tickers(tickers: Sequence[str]) -> List[str]:
    return sorted({t.upper().strip() for t in tickers})


def _cache_path(tickers: Sequence[str], interval: str) -> Path:
    tick_str = "_".join(_sanitize_tickers(tickers))
    return DATA_DIR / f"prices_{tick_str}_{interval}.csv"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.sort_index()

# ---------------------------------------------------------------------------
# Core util
# ---------------------------------------------------------------------------


def _select_price_column(raw: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of adjusted close (fallback close)."""
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        use = "Adj Close" if "Adj Close" in level0 else "Close"
        df = raw[use]
    else:
        # Single ticker → columns like ['Open', 'High', ...]
        if "Adj Close" in raw.columns:
            df = raw["Adj Close"].to_frame()
        else:
            df = raw["Close"].to_frame()
    return df


def download_prices(
    tickers: Sequence[str],
    start: Union[str, datetime] = "2015-01-01",
    end: Optional[Union[str, datetime]] = None,
    interval: str = "1d",
    cache: bool = True,
    force: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    tickers = _sanitize_tickers(tickers)
    cache_path = _cache_path(tickers, interval) if cache else None

    # ---------------------- read existing cache ---------------------------
    if cache and cache_path and cache_path.exists() and not force:
        df_cache = _read_cache(cache_path)
        if end is None or pd.Timestamp(end) <= df_cache.index[-1]:
            return df_cache.loc[pd.to_datetime(start): end]
        start_new = (df_cache.index[-1] +
                     pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("Updating cache from %s to %s", start_new, end or "today")
        raw = yf.download(
            tickers,
            start=start_new,
            end=end,
            interval=interval,
            group_by="column",
            auto_adjust=False,
            progress=show_progress,
        )
        df_new = _select_price_column(raw)
        df_all = pd.concat([df_cache, df_new])
        df_all.to_csv(cache_path)
        return df_all.loc[pd.to_datetime(start): end]

    # ---------------------- fresh download -------------------------------
    logger.info("Downloading %d tickers (%s) from scratch …",
                len(tickers), interval)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="column",
        auto_adjust=False,
        progress=show_progress,
    )
    df = _select_price_column(raw)

    if cache and cache_path:
        df.to_csv(cache_path)
        logger.info("Cached to %s", cache_path.name)

    return df
