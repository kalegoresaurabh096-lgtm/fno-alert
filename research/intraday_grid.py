"""Sensitivity of the intraday selling rules: change one rule at a time around the
base rules (config.toml [intraday]) and check both halves of the history.

    PYTHONPATH=. python research/intraday_grid.py --data data/intraday --out reports/intraday
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from fno_alert.cli import load_config
from fno_alert.intraday import backtest
from fno_alert.intraday.rules import Rules, load_events

ARGS = None


def variants(base: Rules):
    v = [("Your rules (base)", {})]
    v.append(("Stop-loss only, no hedge", {"hedged": False}))
    for d in (0.10, 0.15, 0.25, 0.30):
        v.append((f"Short delta {d:.2f}", {"short_delta": d}))
    for w in (50, 150, 200):
        v.append((f"Wing {w} pts (SENSEX x3)", {"wing": {"NIFTY": w, "SENSEX": 3 * w}}))
    for t in ("09:30", "10:30", "11:30", "13:00"):
        v.append((f"Entry {t}", {"entry_time": t}))
    for d in (0, 2, 4):
        v.append((f"Expiry within {d} sessions", {"max_dte": d}))
    for tp in (0.4, 0.6, 0.8):
        v.append((f"Profit target {int(tp * 100)}%", {"take_profit": tp}))
    for s in (0.5, 1.5, 2.0):
        v.append((f"Stop at {s:g}x premium", {"stop_mult": s}))
    v.append(("No stop (hedge only)", {"stop_mult": 0.0}))
    v.append(("No trend filter (always both sides)", {"trend": "none"}))
    for tp in (0.15, 0.5):
        v.append((f"Trend = {tp}% move since open", {"trend_pct": tp}))
    v.append(("No skip filters", {"gap_max_pct": 99, "vix_max": 99, "or_max_pct": 99}))
    v.append(("No event-day skip", {"_no_events": True}))
    v.append(("Stops on worst print in the minute", {"stop_trigger": "high"}))
    v.append(("Risk 1% per trade", {"risk_pct": 0.01}))
    v.append(("Two trades a day allowed", {"max_trades_per_day": 2}))
    v.append(("Unhedged, stops on worst print", {"hedged": False, "stop_trigger": "high"}))
    return v


def one(job):
    name, over, under = job
    over = dict(over)
    no_events = over.pop("_no_events", False)
    base = Rules.from_config(load_config(ARGS["config"]).get("intraday"))
    r = base.with_(**over)
    hists = [backtest.History(ARGS["data"], u) for u in under]
    tr, log = backtest.run(hists, r, {} if no_events else load_events(), log_days=(name.startswith("Your") or
                                                                                    name.startswith("Stop-loss only")))
    s = backtest.summary(tr, r)
    if len(tr):
        mid = pd.Timestamp(ARGS["mid"]).date()
        s["first_half_pnl"] = int(tr.loc[tr["date"] < mid, "pnl"].sum())
        s["second_half_pnl"] = int(tr.loc[tr["date"] >= mid, "pnl"].sum())
    return name, "+".join(under), s, tr, log


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="reports/intraday")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--underlyings", default="NIFTY,SENSEX")
    ap.add_argument("--mid", default="2025-11-01", help="split date for the two halves")
    a = ap.parse_args()
    ARGS = vars(a)
    os.makedirs(a.out, exist_ok=True)
    base = Rules.from_config(load_config(a.config).get("intraday"))
    sets = [tuple(a.underlyings.split(","))] + [(u,) for u in a.underlyings.split(",") if "," in a.underlyings]
    jobs = []
    for under in sets:
        for name, over in variants(base):
            if len(under) > 1 or len(sets) == 1 or name.startswith("Your") or name.startswith("Stop-loss only"):
                jobs.append((name, over, under))
    rows = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 0)) as ex:
        for name, under, s, tr, log in ex.map(one, jobs):
            rows.append({"variant": name, "markets": under, **{k: v for k, v in s.items() if k != "exit_reasons"},
                         "exit_reasons": json.dumps(s.get("exit_reasons", {}))})
            print(f"{under:13s} {name:40s} trades {s.get('trades', 0):4d}  P&L {s.get('net_pnl', 0):>9,}  "
                  f"halves {s.get('first_half_pnl', 0):>8,} / {s.get('second_half_pnl', 0):>8,}  "
                  f"DD {s.get('max_drawdown_pct', 0)}%", flush=True)
            tag = {"Your rules (base)": "base", "Stop-loss only, no hedge": "stoponly"}.get(name)
            if tag:
                suffix = "" if "+" in under else f"_{under}"
                tr.to_csv(os.path.join(a.out, f"trades_{tag}{suffix}.csv"), index=False)
                log.to_csv(os.path.join(a.out, f"days_{tag}{suffix}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "variants.csv"), index=False)


if __name__ == "__main__":
    main()
