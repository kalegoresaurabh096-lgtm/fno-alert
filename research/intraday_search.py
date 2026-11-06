"""Search for an intraday selling setup that makes money, without peeking at the holdout.

Development period: up to DEV_END. Selection rule (fixed before running):
  * at least 40 development trades
  * profit factor >= 1.15 on the development period
  * positive P&L in both halves of the development period
  * ranked by development P&L
The top 2 are then tested once on the holdout (after DEV_END) with --holdout.

    PYTHONPATH=. python research/intraday_search.py --out reports/intraday/search
    PYTHONPATH=. python research/intraday_search.py --out reports/intraday/search --holdout
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from fno_alert.cli import load_config
from fno_alert.intraday import backtest
from fno_alert.intraday.rules import Rules, load_events

DEV_END = pd.Timestamp("2025-10-31").date()
DEV_MID = pd.Timestamp("2025-04-30").date()
GRID = {
    "entry_time": ["09:45", "11:30", "13:00", "14:00"],
    "max_dte": [0, 1],
    "iv_rv_min": [0.0, 1.0, 1.25],
    "neutral": ["both", "skip"],
    "short_delta": [0.15, 0.20, 0.30],
    "hedged": [True, False],
}
ARGS = {}


def configs():
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        yield dict(zip(keys, vals))


def one(over):
    base = Rules.from_config(load_config("config.toml").get("intraday"))
    r = base.with_(**over)
    hists = [backtest.History(ARGS["data"], u) for u in ("NIFTY", "SENSEX")]
    if ARGS["holdout"]:
        tr, _ = backtest.run(hists, r, load_events(), start=DEV_END + pd.Timedelta(days=1).to_pytimedelta(),
                             log_days=False)
    else:
        tr, _ = backtest.run(hists, r, load_events(), end=DEV_END, log_days=False)
    s = backtest.summary(tr, r)
    s.pop("exit_reasons", None)
    if len(tr):
        s["half1"] = int(tr.loc[tr["date"] <= DEV_MID, "pnl"].sum())
        s["half2"] = int(tr.loc[tr["date"] > DEV_MID, "pnl"].sum())
    return {**over, **s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="reports/intraday/search")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    ARGS.update(vars(a))
    os.makedirs(a.out, exist_ok=True)
    if a.holdout:
        picks = json.load(open(os.path.join(a.out, "picks.json")))
        with ProcessPoolExecutor(os.cpu_count()) as ex:
            res = list(ex.map(one, [p["config"] for p in picks]))
        pd.DataFrame(res).to_csv(os.path.join(a.out, "holdout.csv"), index=False)
        print(pd.DataFrame(res).to_string())
        return
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        res = list(ex.map(one, configs(), chunksize=4))
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(a.out, "dev_grid.csv"), index=False)
    ok = df[(df["trades"] >= 40) & (df["profit_factor"].fillna(0) >= 1.15) & (df["half1"] > 0) & (df["half2"] > 0)]
    ok = ok.sort_values("net_pnl", ascending=False)
    print(f"{len(df)} configs, {len(ok)} pass the selection rule")
    cols = list(GRID) + ["trades", "net_pnl", "half1", "half2", "win_rate", "profit_factor", "max_drawdown_pct",
                         "worst_trade", "costs"]
    print(df.sort_values("net_pnl", ascending=False)[cols].head(15).to_string())
    picks = [{"config": {k: (bool(v) if k == "hedged" else v) for k, v in row[list(GRID)].items()}}
             for _, row in ok.head(2).iterrows()]
    json.dump(picks, open(os.path.join(a.out, "picks.json"), "w"), indent=1, default=float)
    print("picks:", picks)


if __name__ == "__main__":
    main()
