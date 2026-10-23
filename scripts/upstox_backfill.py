"""Download expired F&O contracts, stock prices, NIFTY and VIX from Upstox
(expired contracts need Upstox Plus) into data/upstox/, resumably.

    python scripts/upstox_backfill.py --out data/upstox --max-minutes 320

Layout (one file per finished unit, so a stopped run picks up where it left off):
    eq/<SYMBOL>.csv                 daily OHLCV, not adjusted for splits
    index/nifty.csv, index/vix.csv  daily closes
    fo/<SYMBOL>/<YYYY-MM-DD>.parquet  daily candles of that expiry's futures and
                                      options within STRIKE_BAND of the futures price
    expiries.json                   expired expiries per stock
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
from fno_alert.chain import STRIKE_BAND  # noqa: E402
from fno_alert.upstox import IST, RateLimiter, Upstox, UpstoxError  # noqa: E402

INDEX_KEYS = {"nifty": "NSE_INDEX|Nifty 50", "vix": "NSE_INDEX|India VIX"}
PRICE_START = dt.date(2023, 1, 1)   # about a year of warm-up before the first expired expiry
LOOKBACK_DAYS = 100                 # contract life fetched before each expiry
REF_DAYS = 70                       # window whose futures prices set the strike band


def year_chunks(start: dt.date, end: dt.date):
    s = start
    while s <= end:
        e = min(dt.date(s.year, 12, 31), end)
        yield s, e
        s = e + dt.timedelta(days=1)


def daily_history(ux: Upstox, key: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    parts = [ux.daily_candles(key, s, e) for s, e in year_chunks(start, end)]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "oi"])
    return pd.concat(parts).drop_duplicates("date", keep="last").sort_values("date")


def contract_rows(df: pd.DataFrame, sym: str, inst: str, expiry: dt.date, strike, lot: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame({
        "date": df["date"].dt.date, "symbol": sym, "inst": inst, "expiry": expiry,
        "strike": strike, "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "settle": df["close"], "volume": df["volume"] / lot if lot else df["volume"], "oi": df["oi"],
        "lot": lot,
    })
    return out[out["date"] <= expiry]


def fetch_unit(ux: Upstox, sym: str, ukey: str, expiry: dt.date, eq: pd.DataFrame) -> pd.DataFrame:
    start = expiry - dt.timedelta(days=LOOKBACK_DAYS)
    frames = []
    fc = ux.expired_contracts(ukey, expiry, "future")
    ref = pd.Series(dtype=float)
    for r in fc.to_dict("records"):
        lot = float(r.get("lot_size") or 0)
        c = ux.expired_candles(r["instrument_key"], start, expiry)
        frames.append(contract_rows(c, sym, "FUT", expiry, None, lot))
        if len(c):
            ref = c.set_index("date")["close"]
    win = (ref.index >= pd.Timestamp(expiry - dt.timedelta(days=REF_DAYS))) if len(ref) else []
    ref = ref[win] if len(ref) else ref
    if ref.empty and len(eq):
        e = eq[(eq["date"] >= pd.Timestamp(expiry - dt.timedelta(days=REF_DAYS))) & (eq["date"] <= pd.Timestamp(expiry))]
        ref = e["close"]
    oc = ux.expired_contracts(ukey, expiry, "option")
    if len(oc) and len(ref):
        lo, hi = float(ref.min()) * (1 - STRIKE_BAND), float(ref.max()) * (1 + STRIKE_BAND)
        oc = oc[(oc["strike_price"].astype(float) >= lo) & (oc["strike_price"].astype(float) <= hi)]
        for r in oc.to_dict("records"):
            c = ux.expired_candles(r["instrument_key"], start, expiry)
            frames.append(contract_rows(c, sym, r["instrument_type"], expiry, float(r["strike_price"]),
                                        float(r.get("lot_size") or 0)))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "inst", "expiry", "strike", "open", "high", "low",
                                     "close", "settle", "volume", "oi", "lot"])
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/upstox")
    ap.add_argument("--max-minutes", type=float, default=320)
    ap.add_argument("--symbols", help="comma-separated subset")
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + a.max_minutes * 60
    # Upstox allows 2000 requests per 30 minutes. Leave room for the live scans
    # during market hours (about 04:15-11:15 UTC on weekdays); use nearly all of it otherwise.
    now = dt.datetime.utcnow()
    market = now.weekday() < 5 and dt.time(4, 15) <= now.time() <= dt.time(11, 15)
    per30 = 1400 if market else 1950
    ux = Upstox(limiter=RateLimiter(per_sec=5, per_min=per30 // 25, per_30min=per30),
                cache_dir=os.path.join(a.out, ".cache"))
    today = dt.datetime.now(IST).date()
    uni = ux.fno_universe(ux.instruments(), today)
    if a.symbols:
        uni = uni[uni["symbol"].isin([s.strip().upper() for s in a.symbols.split(",")])]
    for d in ("eq", "index", "fo"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    # 1. Index and stock prices (cheap: a few calls per stock).
    for name, key in INDEX_KEYS.items():
        p = os.path.join(a.out, "index", f"{name}.csv")
        if not os.path.exists(p):
            daily_history(ux, key, PRICE_START, today)[["date", "close"]].to_csv(p, index=False)
    for u in uni.itertuples(index=False):
        p = os.path.join(a.out, "eq", f"{u.symbol}.csv")
        if not os.path.exists(p):
            daily_history(ux, u.underlying_key, PRICE_START, today).drop(columns="oi").to_csv(p, index=False)
    print(f"prices done ({ux.calls} calls, {(time.time() - t0) / 60:.1f} min)", flush=True)

    # 2. Expired expiries per stock.
    ep = os.path.join(a.out, "expiries.json")
    exps = json.load(open(ep)) if os.path.exists(ep) else {}
    for u in uni.itertuples(index=False):
        if u.symbol not in exps:
            try:
                exps[u.symbol] = [d.isoformat() for d in ux.expired_expiries(u.underlying_key)]
            except UpstoxError as e:
                print(f"  {u.symbol}: expiries failed: {e}", flush=True)
                exps[u.symbol] = []
    json.dump(exps, open(ep, "w"), indent=0, sort_keys=True)

    # 3. Contracts, newest expiry first, every stock per expiry.
    todo = sorted({(e, u.symbol, u.underlying_key) for u in uni.itertuples(index=False)
                   for e in exps.get(u.symbol, [])}, reverse=True)
    todo = [t for t in todo if not os.path.exists(os.path.join(a.out, "fo", t[1], f"{t[0]}.parquet"))]
    print(f"{len(todo)} stock-expiries left", flush=True)
    done = 0
    eq_cache = {}
    for e, sym, key in todo:
        if time.time() > deadline:
            break
        if sym not in eq_cache:
            p = os.path.join(a.out, "eq", f"{sym}.csv")
            eq_cache[sym] = pd.read_csv(p, parse_dates=["date"]) if os.path.exists(p) else pd.DataFrame()
        try:
            df = fetch_unit(ux, sym, key, dt.date.fromisoformat(e), eq_cache[sym])
        except UpstoxError as ex:
            print(f"  {sym} {e}: {ex}", flush=True)
            if "401" in str(ex):
                raise
            continue
        os.makedirs(os.path.join(a.out, "fo", sym), exist_ok=True)
        df.to_parquet(os.path.join(a.out, "fo", sym, f"{e}.parquet"), index=False)
        done += 1
        if done % 25 == 0:
            print(f"  {done} done, at {e} {sym}, {ux.calls} calls, {(time.time() - t0) / 60:.0f} min", flush=True)
    left = len(todo) - done
    print(f"Finished this run: {done} stock-expiries, {left} left, {ux.calls} API calls.", flush=True)
    with open(os.path.join(a.out, "status.json"), "w") as fh:
        json.dump({"left": left, "updated": dt.datetime.now(IST).isoformat(timespec="minutes")}, fh)


if __name__ == "__main__":
    main()
