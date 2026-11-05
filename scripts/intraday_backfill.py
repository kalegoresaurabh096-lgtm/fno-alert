"""Download 1-minute history for the intraday NIFTY / SENSEX option-selling backtest
(needs Upstox Plus), resumably.

    python scripts/intraday_backfill.py --underlying NIFTY --out data/intraday --max-minutes 60

Layout (under <out>/<underlying>/):
    index_1m.parquet          index and India VIX 1-minute candles (ts, open, high, low, close, vix)
    expiries.json             expired weekly expiries
    opt/<YYYY-MM-DD>.parquet  1-minute candles of that expiry's options over its last DAYS
                              calendar days, strikes that are multiples of the step within BAND
                              of the index's range in that window
    status.json               {"left": expiries still to fetch}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.parse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fno_alert.upstox import IST, RateLimiter, Upstox, UpstoxError  # noqa: E402

UNDERLYINGS = {  # name: (instrument key, strike step kept)
    "NIFTY": ("NSE_INDEX|Nifty 50", 50),
    "SENSEX": ("BSE_INDEX|SENSEX", 100),
}
VIX = "NSE_INDEX|India VIX"
BAND = 0.05
DAYS = 8          # weekly expiries: every session up to 8 calendar days before expiry
START = dt.date(2024, 9, 1)
COLS = ["ts", "open", "high", "low", "close", "volume", "oi"]


def frame(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame([c[:7] + [0] * (7 - len(c[:7])) for c in candles], columns=COLS)
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(IST).dt.tz_localize(None)
    return df.sort_values("ts").drop_duplicates("ts", keep="last")


def index_minutes(ux: Upstox, key: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    parts, s = [], start
    while s <= end:
        e = min(s + dt.timedelta(days=27), end)
        path = f"/v3/historical-candle/{urllib.parse.quote(key, safe='')}/minutes/1/{e}/{s}"
        parts.append(frame((ux.get(path) or {}).get("candles", [])))
        s = e + dt.timedelta(days=1)
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).drop_duplicates("ts").sort_values("ts") if parts else pd.DataFrame(columns=COLS)


def fetch_expiry(ux: Upstox, key: str, step: int, expiry: dt.date, idx: pd.DataFrame) -> pd.DataFrame:
    start = expiry - dt.timedelta(days=DAYS)
    win = idx[(idx["ts"] >= pd.Timestamp(start)) & (idx["ts"] < pd.Timestamp(expiry + dt.timedelta(days=1)))]
    if win.empty:
        return pd.DataFrame()
    lo, hi = win["low"].min() * (1 - BAND), win["high"].max() * (1 + BAND)
    oc = ux.expired_contracts(key, expiry, "option")
    if oc.empty:
        return pd.DataFrame()
    k = oc["strike_price"].astype(float)
    oc = oc[(k >= lo) & (k <= hi) & (k % step == 0)]
    frames = []
    for r in oc.to_dict("records"):
        path = (f"/v2/expired-instruments/historical-candle/{urllib.parse.quote(r['instrument_key'], safe='')}"
                f"/1minute/{expiry}/{start}")
        c = frame((ux.get(path) or {}).get("candles", []))
        if len(c):
            c = c.assign(strike=float(r["strike_price"]), opt=r["instrument_type"],
                         lot=float(r.get("lot_size") or 0))
            frames.append(c)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["expiry"] = pd.Timestamp(expiry)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    for col in ["volume", "oi"]:
        df[col] = df[col].astype("int64")
    df["strike"] = df["strike"].astype("float32")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="NIFTY", choices=sorted(UNDERLYINGS))
    ap.add_argument("--out", default="data/intraday")
    ap.add_argument("--max-minutes", type=float, default=300)
    a = ap.parse_args()
    key, step = UNDERLYINGS[a.underlying]
    a.out = os.path.join(a.out, a.underlying)
    t0 = time.time()
    deadline = t0 + a.max_minutes * 60
    now = dt.datetime.utcnow()
    market = now.weekday() < 5 and dt.time(3, 30) <= now.time() <= dt.time(10, 15)
    per30 = 1400 if market else 1950
    ux = Upstox(limiter=RateLimiter(per_sec=5, per_min=per30 // 25, per_30min=per30),
                cache_dir=os.path.join(a.out, ".cache"))
    today = dt.datetime.now(IST).date()
    os.makedirs(os.path.join(a.out, "opt"), exist_ok=True)

    ip = os.path.join(a.out, "index_1m.parquet")
    if os.path.exists(ip):
        idx = pd.read_parquet(ip)
        last = idx["ts"].max().date()
    else:
        idx, last = pd.DataFrame(), START - dt.timedelta(days=1)
    if last < today - dt.timedelta(days=1):
        n = index_minutes(ux, key, last + dt.timedelta(days=1), today)
        v = index_minutes(ux, VIX, last + dt.timedelta(days=1), today)[["ts", "close"]].rename(columns={"close": "vix"})
        new = n[["ts", "open", "high", "low", "close"]].merge(v, on="ts", how="left")
        idx = pd.concat([idx, new]).drop_duplicates("ts").sort_values("ts") if len(idx) else new
        idx.to_parquet(ip, index=False)
    print(f"index: {len(idx)} minutes, {idx['ts'].min()} .. {idx['ts'].max()}", flush=True)

    exps = [d.isoformat() for d in ux.expired_expiries(key) if d >= START]
    json.dump(exps, open(os.path.join(a.out, "expiries.json"), "w"), indent=0)
    todo = [e for e in sorted(exps, reverse=True) if not os.path.exists(os.path.join(a.out, "opt", f"{e}.parquet"))]
    print(f"{len(exps)} expired {a.underlying} expiries, {len(todo)} to fetch", flush=True)
    done = 0
    for e in todo:
        if time.time() > deadline:
            break
        try:
            df = fetch_expiry(ux, key, step, dt.date.fromisoformat(e), idx)
        except UpstoxError as ex:
            print(f"  {e}: {ex}", flush=True)
            if "401" in str(ex):
                raise
            continue
        df.to_parquet(os.path.join(a.out, "opt", f"{e}.parquet"), index=False, compression="zstd")
        done += 1
        print(f"  {e}: {len(df)} rows, {ux.calls} calls, {(time.time() - t0) / 60:.0f} min", flush=True)
    left = len(todo) - done
    print(f"Finished this run: {done} expiries, {left} left, {ux.calls} API calls.", flush=True)
    json.dump({"left": left, "updated": dt.datetime.now(IST).isoformat(timespec="minutes")},
              open(os.path.join(a.out, "status.json"), "w"))


if __name__ == "__main__":
    main()
