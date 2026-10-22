"""What counts as a "big move", measured from the first price you could act on.

The signal is computed after the close of day t (the end-of-day email goes
out after 15:30), so the earliest realistic entry is the open of day t+1.
The overnight gap from close[t] to open[t+1] is therefore *not* credited.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5        # trading days after the signal
SIGMA_MULT = 2.0   # threshold = SIGMA_MULT x the stock's normal H-day move
MIN_MOVE = 0.04    # ...but never less than 4%


def forward_labels(close: pd.Series, open_: pd.Series, sigma60: pd.Series, horizon: int = HORIZON,
                   sigma_mult: float = SIGMA_MULT, min_move: float = MIN_MOVE) -> pd.DataFrame:
    """For each signal day t:

    big      1 if any close in t+1..t+H is at least `thr` away from open[t+1]
    dir      +1 if the first close to cross the threshold was above, -1 below, 0 none
    max_up / max_dn  largest up/down close excursion from open[t+1]
    end_ret  close[t+H] / open[t+1] - 1
    gap_next open[t+1] / close[t] - 1 (the part you could not trade)
    thr      the threshold used (fraction)

    The last H rows have NaN labels (future unknown).
    """
    c = close.to_numpy(float)
    o = open_.to_numpy(float)
    n = len(c)
    thr = np.maximum(min_move, sigma_mult * sigma60.to_numpy(float) * np.sqrt(horizon))
    big, direction, max_up, max_dn, end_ret, gap = (np.full(n, np.nan) for _ in range(6))
    for t in range(n - horizon):
        ref = o[t + 1]
        if not (np.isfinite(ref) and ref > 0 and np.isfinite(thr[t]) and np.isfinite(c[t])):
            continue
        path = c[t + 1:t + 1 + horizon] / ref - 1
        if np.isnan(path).any():
            continue
        gap[t] = ref / c[t] - 1
        max_up[t] = path.max()
        max_dn[t] = path.min()
        end_ret[t] = path[-1]
        hit = np.flatnonzero(np.abs(path) >= thr[t])
        if len(hit):
            big[t] = 1.0
            direction[t] = np.sign(path[hit[0]])
        else:
            big[t] = 0.0
            direction[t] = 0.0
    return pd.DataFrame({"big": big, "dir": direction, "max_up": max_up, "max_dn": max_dn,
                         "end_ret": end_ret, "gap_next": gap, "thr": thr}, index=close.index)
