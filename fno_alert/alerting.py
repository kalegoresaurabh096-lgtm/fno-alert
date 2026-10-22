"""Which scores become emails. Shared by the backtest and the live scanner so
both count alerts the same way."""
from __future__ import annotations

import numpy as np
import pandas as pd

COOLDOWN_DAYS = 5      # trading days: do not re-alert a stock inside one label horizon...
REALERT_BUMP = 0.10    # ...unless its probability rose by at least this much


def emit(df: pd.DataFrame, threshold: float, score: str = "p_big", max_per_day: int | None = 10,
         cooldown_days: int = COOLDOWN_DAYS, bump: float = REALERT_BUMP) -> pd.DataFrame:
    """Rows of df (date, symbol, score) that would have been emailed."""
    d = df[df[score] >= threshold].sort_values(["date", score], ascending=[True, False])
    if d.empty:
        return d
    if max_per_day:
        d = d.groupby("date", sort=False).head(max_per_day)
    days = np.sort(df["date"].unique())
    day_idx = {v: i for i, v in enumerate(days)}
    last = {}
    keep = []
    for i, r in zip(d.index, d[["date", "symbol", score]].itertuples(index=False)):
        t = day_idx[r.date]
        prev = last.get(r.symbol)
        if prev is None or t - prev[0] > cooldown_days or r[2] >= prev[1] + bump:
            keep.append(i)
            last[r.symbol] = (t, r[2])
    return d.loc[keep]
