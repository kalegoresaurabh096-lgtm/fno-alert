"""What counts as a "big move", measured strictly after the signal day."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5        # trading days after the signal
SIGMA_MULT = 2.0   # threshold = SIGMA_MULT x the stock's normal H-day move
MIN_MOVE = 0.04    # ...but never less than 4%


def forward_labels(close: pd.Series, sigma60: pd.Series, horizon: int = HORIZON,
                   sigma_mult: float = SIGMA_MULT, min_move: float = MIN_MOVE) -> pd.DataFrame:
    """For each day t (signal at the close of t):

    big      1 if any close in t+1..t+H is at least `thr` away from close[t]
    dir      +1 if the first close to cross the threshold was above, -1 below, 0 none
    max_up / max_dn  largest favourable/unfavourable close-to-close excursion
    thr      the threshold used (fraction)

    The last H rows have NaN labels (future unknown).
    """
    c = close.to_numpy(float)
    n = len(c)
    thr = np.maximum(min_move, sigma_mult * sigma60.to_numpy(float) * np.sqrt(horizon))
    big = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    max_up = np.full(n, np.nan)
    max_dn = np.full(n, np.nan)
    end_ret = np.full(n, np.nan)
    for t in range(n - horizon):
        if np.isnan(c[t]) or np.isnan(thr[t]):
            continue
        path = c[t + 1:t + 1 + horizon] / c[t] - 1
        if np.isnan(path).any():
            continue
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
                         "end_ret": end_ret, "thr": thr}, index=close.index)
