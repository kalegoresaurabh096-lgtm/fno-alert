"""Join prices, derivatives and events into one research panel."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import features as feat
from . import labels as lab


def stock_frame(px: pd.DataFrame, snap: pd.DataFrame | None, mkt_f: pd.DataFrame | None,
                results, with_labels: bool = True) -> pd.DataFrame:
    """All features (and labels) for one stock. px/snap indexed by date."""
    tech = feat.technical_features(px)
    parts = [tech]
    if snap is not None and len(snap):
        snap = snap[~snap.index.duplicated(keep="last")]
        # Align to the stock's trading days so shift(5) always means 5 sessions,
        # even when a day's snapshot is missing.
        snap = snap.reindex(tech.index)
        raw = px["close_raw"] if "close_raw" in px else None
        parts.append(feat.option_features(snap, tech, raw))
        keep = ["F", "max_pain", "call_wall", "put_wall", "atm_strike", "focus_expiry", "lot", "fut_close"]
        parts.append(snap[[k for k in keep if k in snap.columns]])
    df = pd.concat(parts, axis=1)
    if "corp_action" in px:
        ca = px["corp_action"].reindex(df.index).fillna(0)
        # OI/volume comparisons are meaningless across a split/bonus day.
        for c in ("fut_oi_chg", "fut_oi_chg5", "opt_oi_chg", "fut_oi_chg_z", "buildup_long",
                  "buildup_short", "short_covering", "long_unwinding", "near_oi_chg"):
            if c in df:
                df.loc[ca.rolling(6, min_periods=1).max().astype(bool), c] = np.nan
        df["corp_action"] = ca
    if mkt_f is not None and len(mkt_f):
        df = df.join(mkt_f, how="left")
    df = df.join(feat.event_features(df.index, results), how="left")
    if with_labels:
        df = df.join(lab.forward_labels(df["close"], px["open"].reindex(df.index), df["sigma60"]))
    return df


def build_panel(processed: str = "data/processed", min_turnover_cr: float = 10.0) -> pd.DataFrame:
    ohlcv = pd.read_parquet(os.path.join(processed, "ohlcv.parquet"))
    snap = pd.read_parquet(os.path.join(processed, "opt_daily.parquet"))
    mkt_path = os.path.join(processed, "market.parquet")
    mkt = pd.read_parquet(mkt_path) if os.path.exists(mkt_path) else pd.DataFrame()
    mkt_f = feat.market_features(mkt) if len(mkt) else None
    ev_path = os.path.join(processed, "events.parquet")
    ev = pd.read_parquet(ev_path) if os.path.exists(ev_path) else pd.DataFrame(columns=["symbol", "date"])

    frames = []
    snap_by = dict(tuple(snap.groupby("symbol")))
    ev_by = {k: g[["date", "announced"]] for k, g in ev.groupby("symbol")} if len(ev) else {}
    for sym, px in ohlcv.groupby("symbol"):
        px = px.set_index("date").sort_index()
        px = px[~px.index.duplicated(keep="last")]
        if len(px) < 80 or sym not in snap_by:
            continue
        s = snap_by[sym].set_index("date").sort_index()
        df = stock_frame(px, s, mkt_f, ev_by.get(sym))
        df["symbol"] = sym
        # Universe = days on which the stock actually had F&O contracts trading.
        df["in_fno"] = df.index.isin(s.index[s["F"].notna()] if "F" in s else s.index)
        frames.append(df)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    panel = panel[panel["in_fno"] & (panel["dvol_cr"] >= min_turnover_cr)]
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)
