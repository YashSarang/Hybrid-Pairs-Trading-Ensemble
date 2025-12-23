# src/trade_logger.py
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import pandas as pd
from pathlib import Path
from datetime import datetime

@dataclass
class Trade:
    dt_open: pd.Timestamp
    dt_close: Optional[pd.Timestamp]
    pair: str
    side: str                 # 'long' or 'short'
    entry_price_x: float
    entry_price_y: float
    exit_price_x: Optional[float]
    exit_price_y: Optional[float]
    qty_x: float
    qty_y: float
    pnl: Optional[float]
    reason_open: str          # e.g., 'signal'
    reason_close: Optional[str]  # e.g., 'stop', 'tp', 'time', 'signal_flip'
    meta: Optional[Dict] = None  # anything extra (zscore, thresholds, etc.)

class TradeLogger:
    def __init__(self):
        self._events: List[Trade] = []

    def log_open(self, *, dt, pair, side, px_x, px_y, qty_x, qty_y, reason_open, meta=None):
        self._events.append(
            Trade(
                dt_open=pd.Timestamp(dt),
                dt_close=None,
                pair=pair,
                side=side,
                entry_price_x=float(px_x),
                entry_price_y=float(px_y),
                exit_price_x=None,
                exit_price_y=None,
                qty_x=float(qty_x),
                qty_y=float(qty_y),
                pnl=None,
                reason_open=str(reason_open),
                reason_close=None,
                meta=meta or {}
            )
        )

    def log_close(self, *, dt, pair, px_x, px_y, pnl, reason_close, meta=None):
        # match the most recent open trade for that pair that is still open
        for tr in reversed(self._events):
            if tr.pair == pair and tr.dt_close is None:
                tr.dt_close = pd.Timestamp(dt)
                tr.exit_price_x = float(px_x)
                tr.exit_price_y = float(px_y)
                tr.pnl = float(pnl)
                tr.reason_close = str(reason_close)
                if meta:
                    tr.meta = {**(tr.meta or {}), **meta}
                break

    def to_frame(self) -> pd.DataFrame:
        if not self._events:
            return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
        rows = [asdict(e) for e in self._events]
        df = pd.json_normalize(rows, sep=".")
        # ensure datetime dtypes
        if "dt_open" in df: df["dt_open"] = pd.to_datetime(df["dt_open"])
        if "dt_close" in df: df["dt_close"] = pd.to_datetime(df["dt_close"])
        return df

    def flush(self, out_dir: str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_path = out / f"trades_{ts}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
