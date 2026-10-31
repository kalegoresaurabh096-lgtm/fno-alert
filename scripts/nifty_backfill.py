"""Download expired NIFTY options (weekly and monthly expiries) from Upstox
(needs Upstox Plus) for the index option-selling backtest, resumably.

    python scripts/nifty_backfill.py --out data/nifty --max-minutes 300

Layout:
    index.csv                   NIFTY 50 and India VIX daily OHLC
    expiries.json               expired NIFTY expiries
    opt/<YYYY-MM-DD>.parquet    daily candles of that expiry's options, strikes that
                                are multiples of 100 within BAND of NIFTY's range over
                                the contract's last LOOKBACK_DAYS
    status.json                 {"left": expiries still to fetch}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fno_alert.upstox import IST, RateLimiter, Upstox, UpstoxError  # noqa: E402
from scripts.upstox_backfill import daily_history  # noqa: E402

NIFTY = "NSE_INDEX|Nifty 50"
VIX = "NSE_INDEX|India VIX"
BAND = 0.08            # strikes within 8% of NIFTY's range
STEP = 100             # keep strikes that are multiples of 100
LOOKBACK_DAYS = 45     # weekly condors open 1-4 weeks before expiry


def fetch_expiry(ux: Upstox, expiry: dt.date, idx: pd.DataFrame) -> pd.DataFrame:
    start = expiry - dt.timedelta(days=LOOKBACK_DAYS)
    win = idx[(idx["date"] >= pd.Timestamp(start)) & (idx["date"] <= pd.Timestamp(expiry))]
    if win.empty:
        return pd.DataFrame()
    lo, hi = win["low"].min() * (1 - BAND), win["high"].max() * (1 + BAND)
    oc = ux.expired_contracts(NIFTY, expiry, "option")
    if oc.empty:
        return pd.DataFrame()
    k = oc["strike_price"].astype(float)
    oc = oc[(k >= lo) & (k <= hi) & (k % STEP == 0)]
    frames = []
    for r in oc.to_dict("records"):
        c = ux.expired_candles(r["instrument_key"], start, expiry)
        if len(c):
            c = c.assign(expiry=expiry, strike=float(r["strike_price"]), opt=r["instrument_type"],
                         lot=float(r.get("lot_size") or 0))
            frames.append(c[c["date"] <= pd.Timestamp(expiry)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/nifty")
    ap.add_argument("--max-minutes", type=float, default=300)
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + a.max_minutes * 60
    now = dt.datetime.utcnow()
    market = now.weekday() < 5 and dt.time(4, 15) <= now.time() <= dt.time(11, 15)
    per30 = 1400 if market else 1950
    ux = Upstox(limiter=RateLimiter(per_sec=5, per_min=per30 // 25, per_30min=per30),
                cache_dir=os.path.join(a.out, ".cache"))
    today = dt.datetime.now(IST).date()
    os.makedirs(os.path.join(a.out, "opt"), exist_ok=True)

    ip = os.path.join(a.out, "index.csv")
    n = daily_history(ux, NIFTY, dt.date(2023, 1, 1), today)
    v = daily_history(ux, VIX, dt.date(2023, 1, 1), today)[["date", "close"]].rename(columns={"close": "vix"})
    idx = n[["date", "open", "high", "low", "close"]].merge(v, on="date", how="left")
    idx.to_csv(ip, index=False)

    exps = [d.isoformat() for d in ux.expired_expiries(NIFTY)]
    json.dump(exps, open(os.path.join(a.out, "expiries.json"), "w"), indent=0)
    todo = [e for e in sorted(exps, reverse=True) if not os.path.exists(os.path.join(a.out, "opt", f"{e}.parquet"))]
    print(f"{len(exps)} expired NIFTY expiries, {len(todo)} to fetch", flush=True)
    done = 0
    for e in todo:
        if time.time() > deadline:
            break
        try:
            df = fetch_expiry(ux, dt.date.fromisoformat(e), idx)
        except UpstoxError as ex:
            print(f"  {e}: {ex}", flush=True)
            if "401" in str(ex):
                raise
            continue
        df.to_parquet(os.path.join(a.out, "opt", f"{e}.parquet"), index=False)
        done += 1
        print(f"  {e}: {len(df)} rows, {ux.calls} calls, {(time.time() - t0) / 60:.0f} min", flush=True)
    left = len(todo) - done
    print(f"Finished this run: {done} expiries, {left} left, {ux.calls} API calls.", flush=True)
    json.dump({"left": left, "updated": dt.datetime.now(IST).isoformat(timespec="minutes")},
              open(os.path.join(a.out, "status.json"), "w"))


if __name__ == "__main__":
    main()
