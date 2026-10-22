"""Price-history hygiene for broker candles that may not be split-adjusted.

The backtest uses NSE's own adjustment factors (from the cash bhavcopy), so
this heuristic only runs on live broker data. It is deliberately strict:
it only accepts the common split/bonus ratios, needs a near-exact match,
and, when lot sizes are known, a matching F&O lot-size change (NSE scales
the lot by the same factor on a split or bonus).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# new/old price ratios: 1:1 bonus or 2-for-1 split = 1/2, 2:1 bonus = 1/3,
# 1:2 bonus = 2/3, splits 10->2 = 1/5, 10->1 = 1/10, 1:3 bonus = 1/4.
RATIOS = np.array([1 / 2, 1 / 3, 2 / 3, 1 / 4, 1 / 5, 1 / 10])


def adjust_splits(px: pd.DataFrame, lots: pd.Series | None = None, tol: float = 0.015,
                  max_ratio: float = 0.70) -> pd.DataFrame:
    """Back-adjust OHLCV for splits/bonuses detected from overnight gaps.

    A gap only counts when open[t]/close[t-1] is within `tol` of a ratio in
    RATIOS (all imply a 30%+ fall) and, if `lots` (F&O lot size by date) is
    given, the lot grew by the inverse ratio (+-5%) on that day.
    Adds close_raw (unadjusted close) and corp_action (1 on detected days).
    """
    px = px.sort_index().copy()
    px["close_raw"] = px["close"].astype(float)
    o, c = px["open"].astype(float), px["close"].astype(float)
    ratio = (o / c.shift(1)).to_numpy()
    lot_ratio = None
    if lots is not None and len(lots):
        lots = lots[~lots.index.duplicated(keep="last")].sort_index()
        lr = lots.reindex(px.index).ffill()
        lot_ratio = (lr / lr.shift(1)).to_numpy()
    factor = np.ones(len(px))
    for i in range(1, len(px)):
        r = ratio[i]
        if not np.isfinite(r) or r > max_ratio:
            continue
        j = np.argmin(np.abs(r / RATIOS - 1))
        if abs(r / RATIOS[j] - 1) > tol:
            continue
        if lot_ratio is not None and np.isfinite(lot_ratio[i]):
            if abs(lot_ratio[i] * RATIOS[j] - 1) > 0.05:
                continue   # lot did not change: treat as a real price move
        factor[i] = RATIOS[j]
    cum = np.cumprod(factor[::-1])[::-1]
    cum = np.append(cum[1:], 1.0)          # factor at t applies to rows before t
    for k in ("open", "high", "low", "close"):
        px[k] = px[k].astype(float) * cum
    px["volume"] = px["volume"].astype(float) / cum
    px["corp_action"] = (factor != 1).astype(int)
    return px
