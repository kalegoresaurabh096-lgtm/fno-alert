"""Day-to-day log of every scan, for the dashboard.

Each run appends one row to state/dash/runs.csv and one row per scored stock
to state/dash/scores.csv. Only the last KEEP_SESSIONS trading days of scores
are kept so the file stays small.
"""
from __future__ import annotations

import datetime as dt
import math
import os

import numpy as np
import pandas as pd

from . import explain
from . import features as feat

KEEP_SESSIONS = 20
SCORE_COLS = ["run_time", "date", "mode", "symbol", "score", "p_up", "p_dn", "points", "close", "ret1",
              "atm_iv", "iv_rank", "iv_hv", "pcr_oi", "fut_oi_chg", "rvol", "squeeze_days", "buildup",
              "alerted", "action", "reasons"]


def _num(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def _append(path: str, new: pd.DataFrame, keep_after: pd.Timestamp | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    if keep_after is not None and "date" in out:
        out = out[pd.to_datetime(out["date"]) >= keep_after]
    out.to_csv(path, index=False)


def record(root: str, now: dt.datetime, mode: str, df: pd.DataFrame, alerts: list[dict], threshold: float,
           model: str, scanned: int, errors: int, api_calls: int, nifty=None, vix=None):
    run_time = now.strftime("%Y-%m-%d %H:%M")
    day = now.strftime("%Y-%m-%d")
    by_sym = {a["symbol"]: a for a in alerts}
    rows = []
    for r in df.to_dict("records"):
        pts, why, _ = explain.reasons(r)
        a = by_sym.get(r["symbol"])
        rows.append({
            "run_time": run_time, "date": day, "mode": mode, "symbol": r["symbol"],
            "score": _num(r.get("p_big")), "p_up": _num(r.get("p_up")), "p_dn": _num(r.get("p_dn")),
            "points": _num(pts, 1), "close": _num(r.get("spot", r.get("close")), 2), "ret1": _num(r.get("ret1")),
            "atm_iv": _num(r.get("atm_iv")), "iv_rank": _num(r.get("iv_rank")), "iv_hv": _num(r.get("iv_hv")),
            "pcr_oi": _num(r.get("pcr_oi")), "fut_oi_chg": _num(r.get("fut_oi_chg")), "rvol": _num(r.get("rvol"), 2),
            "squeeze_days": _num(r.get("squeeze_days"), 0),
            "buildup": feat.buildup_label(r.get("ret1", np.nan), r.get("fut_oi_chg", np.nan)),
            "alerted": int(a is not None), "action": a["action"] if a else "",
            "reasons": " | ".join((a["reasons"] if a else why)[:3]),
        })
    scores = pd.DataFrame(rows, columns=SCORE_COLS)
    dates = sorted(set(pd.read_csv(os.path.join(root, "dash", "scores.csv"))["date"])
                   if os.path.exists(os.path.join(root, "dash", "scores.csv")) else []) + [day]
    dates = sorted(set(dates))
    keep_after = pd.Timestamp(dates[-KEEP_SESSIONS]) if len(dates) > KEEP_SESSIONS else None
    _append(os.path.join(root, "dash", "scores.csv"), scores, keep_after)

    top = scores.sort_values("score", ascending=False).head(1)
    run = pd.DataFrame([{
        "run_time": run_time, "date": day, "mode": mode, "model": model, "threshold": _num(threshold),
        "scanned": scanned, "errors": errors, "alerts": len(alerts), "api_calls": api_calls,
        "near_threshold": int((scores["score"] >= threshold * 0.8).sum()),
        "top_symbol": top["symbol"].iloc[0] if len(top) else "", "top_score": _num(top["score"].iloc[0]) if len(top) else None,
        "nifty": _num(nifty, 2), "vix": _num(vix, 2),
    }])
    _append(os.path.join(root, "dash", "runs.csv"), run)
