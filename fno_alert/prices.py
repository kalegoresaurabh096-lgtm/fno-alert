"""Price-history hygiene for broker candles that may not be split-adjusted."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Common Indian split/bonus price ratios (new/old): 1:1 bonus = 1/2, 1:2 bonus = 2/3,
# 2:1 bonus = 1/3, split 10->5 = 1/2, 10->2 = 1/5, 10->1 = 1/10, 5->1 = 1/5, 4:1 bonus = 1/5 ...
RATIOS = np.array([1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4, 1 / 5, 2 / 5, 1 / 10, 1 / 1.5, 4 / 5, 1 / 6, 1 / 8])


def adjust_splits(px: pd.DataFrame, tol: float = 0.035, min_gap: float = 0.18) -> pd.DataFrame:
    """Detect overnight price gaps that match a split/bonus ratio and back-adjust.

    A real 20%+ overnight crash that also lands within 3.5% of a split ratio
    is rare for F&O stocks; flagged days are kept in `corp_action` so they can
    be reviewed and OI comparisons around them are masked downstream.
    """
    px = px.sort_index().copy()
    o, c = px["open"].astype(float), px["close"].astype(float)
    ratio = (o / c.shift(1)).to_numpy()
    factor = np.ones(len(px))
    for i in range(1, len(px)):
        r = ratio[i]
        if not np.isfinite(r) or abs(r - 1) < min_gap:
            continue
        cand = RATIOS if r < 1 else 1 / RATIOS
        j = np.argmin(np.abs(r / cand - 1))
        if abs(r / cand[j] - 1) <= tol:
            factor[i] = cand[j]
    cum = np.cumprod(factor[::-1])[::-1]
    cum = np.append(cum[1:], 1.0)          # factor at t applies to rows before t
    for k in ("open", "high", "low", "close"):
        px[k] = px[k].astype(float) * cum
    px["volume"] = px["volume"].astype(float) / cum
    px["corp_action"] = (factor != 1).astype(int)
    return px
