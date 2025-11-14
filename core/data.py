"""Data layer for the Pairs Trading app (NSE-first).

Contains:
- DataConfig: parameters for data retrieval/resampling
- DataSource: abstract base
- YFinanceNSESource: fetches OHLCV from Yahoo Finance for NSE (.NS) tickers
- CSVUploadSource: loads from user CSV/Parquet files (wide or per-ticker)
- Helpers: _to_datetime_index, FREQ_TO_YF_INTERVAL

Notes
-----
* We avoid any Streamlit imports here to keep the core layer UI-agnostic.
* yfinance intraday history is limited by Yahoo; the app should surface warnings in the UI.
* We default to `Adj Close` where available but allow override via DataConfig.price_field.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import os
from typing import List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # handled in code paths

# ---------------------------------------------
# Helpers & Config
# ---------------------------------------------

FREQ_TO_YF_INTERVAL = {
    "1D": "1d",
    "1H": "60m",
    "1min": "1m",
}


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tz-naive, ascending DatetimeIndex, inferring from common columns if needed."""
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("datetime", "timestamp", "date", "Date", "time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                df = df.set_index(col)
                break
    df = df.sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        # drop tz to simplify downstream resampling/alignment
        try:
            df.index = pd.DatetimeIndex(df.index).tz_convert(None)
        except Exception:
            # already tz-naive or conversion not applicable
            df.index = pd.DatetimeIndex(df.index)
    return df


# ---------------------------------------------
# Data Model
# ---------------------------------------------

@dataclass
class DataConfig:
    start: datetime
    end: datetime
    freq: str  # '1D', '1H', '1min'
    price_field: str = "Adj Close"


class DataSource:
    """Abstract price source.

    Implementations must return a wide DataFrame with DatetimeIndex and
    columns = tickers (no ".NS" suffix), values = prices at the requested
    frequency, clipped to [start, end], forward-filled, with no all-NaN columns.
    """

    def get_prices(self, universe: List[str], cfg: DataConfig) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceNSESource(DataSource):
    """Fetch NSE prices from Yahoo Finance via yfinance.

    * Automatically appends ".NS" if not provided.
    * Supports daily, hourly, and minute intervals (subject to Yahoo limits).
    * Returns a wide DataFrame with columns named **without** the .NS suffix.
    """

    def get_prices(self, universe: List[str], cfg: DataConfig) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed in this environment.")
        if cfg.freq not in FREQ_TO_YF_INTERVAL:
            raise ValueError(f"Unsupported freq: {cfg.freq}. Use one of {list(FREQ_TO_YF_INTERVAL)}")

        # Normalize tickers to Yahoo format, but we'll return plain codes.
        tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in universe]
        interval = FREQ_TO_YF_INTERVAL[cfg.freq]
        start = pd.Timestamp(cfg.start)
        end = pd.Timestamp(cfg.end) + pd.Timedelta(days=1)

        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        # Build a wide price frame for the selected field (fallback to Close)
        def _strip_ns(t: str) -> str:
            return t[:-3] if t.endswith(".NS") else t

        if isinstance(data.columns, pd.MultiIndex):
            fields = list(data.columns.levels[1])
            field = cfg.price_field if cfg.price_field in fields else "Close"
            frames = []
            for t in tickers:
                col = (t, field)
                if col in data.columns:
                    s = data[col].rename(_strip_ns(t))
                    frames.append(s)
            if not frames:
                raise RuntimeError("No matching price field found in Yahoo data.")
            wide = pd.concat(frames, axis=1)
        else:
            # Single ticker shape
            field = cfg.price_field if cfg.price_field in data.columns else "Close"
            name = _strip_ns(tickers[0])
            wide = data[[field]].rename(columns={field: name})

        wide = _to_datetime_index(wide)
        # Clip and resample to exact requested frequency (safety)
        wide = wide.loc[(wide.index >= pd.Timestamp(cfg.start)) & (wide.index <= pd.Timestamp(cfg.end))]
        if cfg.freq == "1D":
            wide = wide.resample("1D").last()
        elif cfg.freq == "1H":
            wide = wide.resample("1H").last()
        elif cfg.freq == "1min":
            wide = wide.resample("1min").last()

        return wide.ffill().dropna(axis=1, how="any")

    def estimate_adv(self, universe: List[str], cfg: DataConfig, lookback: int = 60) -> pd.DataFrame:
        """Estimate Average Daily Value (₹) over the last `lookback` bars.

        Returns a DataFrame with index=tickers (no .NS), column 'ADV'.
        """
        if yf is None:
            raise RuntimeError("yfinance is not installed in this environment.")
        tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in universe]
        interval = FREQ_TO_YF_INTERVAL.get(cfg.freq, "1d")
        start = pd.Timestamp(cfg.end) - pd.Timedelta(days=lookback * 2)
        end = pd.Timestamp(cfg.end) + pd.Timedelta(days=1)
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        adv = {}
        for t in tickers:
            name = t[:-3] if t.endswith(".NS") else t
            try:
                close = data[(t, "Close")].dropna()
                vol = data[(t, "Volume")].reindex(close.index).fillna(0)
            except Exception:
                # Single-ticker or missing data: try flat columns
                try:
                    close = data["Close"].dropna()
                    vol = data["Volume"].reindex(close.index).fillna(0)
                except Exception:
                    continue
            val = (close * vol).tail(lookback)
            if len(val) > 0:
                adv[name] = float(val.mean())
        return pd.DataFrame.from_dict(adv, orient="index", columns=["ADV"]) 


class CSVUploadSource(DataSource):
    """Load prices from CSV/Parquet files.

    Supports either a wide file containing multiple tickers as columns or
    per-ticker files with OHLCV columns and an optional `Ticker` column.
    """

    def __init__(self, files: List[io.BytesIO]):
        self.files = files

    def get_prices(self, universe: List[str], cfg: DataConfig) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for f in self.files:
            try:
                df = pd.read_csv(f)
            except Exception:
                f.seek(0)
                df = pd.read_parquet(f)
            df = _to_datetime_index(df)

            # Wide file: directly select intersection of columns with universe
            cols = [c for c in df.columns if c in universe]
            if cols:
                dfs.append(df[cols])
                continue

            # Per-ticker file: detect the ticker
            ticker = None
            if "Ticker" in df.columns:
                ticker = str(df["Ticker"].iloc[0])
            if ticker is None and hasattr(f, "name"):
                base = os.path.basename(getattr(f, "name", ""))
                ticker = os.path.splitext(base)[0]

            field = cfg.price_field if cfg.price_field in df.columns else "Close"
            if field in df.columns and ticker:
                dfs.append(df[[field]].rename(columns={field: ticker}))

        if not dfs:
            raise ValueError("No valid data found in uploaded files.")

        wide = pd.concat(dfs, axis=1).sort_index()
        wide = wide.loc[(wide.index >= pd.Timestamp(cfg.start)) & (wide.index <= pd.Timestamp(cfg.end))]
        if cfg.freq != "1D":
            wide = wide.resample(cfg.freq).last().dropna(how="all")
        return wide.ffill().dropna(axis=1, how="any")
