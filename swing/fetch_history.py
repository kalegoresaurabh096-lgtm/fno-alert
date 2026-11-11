"""Download long daily price history for the swing backtest (runs on GitHub Actions).

Universe: the top 800 companies in AMFI's market cap ranking (state/mfwatch/mcap.csv),
which comfortably covers every company above Rs 5,000 crore. ISINs are mapped to NSE
symbols with Upstox's public instrument list. Prices come from Yahoo Finance
(split/bonus adjusted, not dividend adjusted). Shares outstanding (today's) are saved so
the backtest can estimate market cap on each past day as price x shares.

    python swing/fetch_history.py --out swing-data
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

UPSTOX_NSE = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
START = "2008-01-01"


def symbols(top: int) -> pd.DataFrame:
    amfi = pd.read_csv("state/mfwatch/mcap.csv")
    amfi = amfi[amfi["rank"] <= top]
    raw = requests.get(UPSTOX_NSE, timeout=60).content
    inst = pd.DataFrame(json.load(io.BytesIO(gzip.decompress(raw))))
    inst = inst[(inst.segment == "NSE_EQ") & (inst.instrument_type == "EQ")]
    m = amfi.merge(inst[["isin", "trading_symbol"]].drop_duplicates("isin"), on="isin", how="left")
    print(f"{len(amfi)} companies, {m.trading_symbol.notna().sum()} mapped to NSE symbols")
    return m.dropna(subset=["trading_symbol"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=800)
    a = ap.parse_args()
    (a.out / "eq").mkdir(parents=True, exist_ok=True)

    uni = symbols(a.top)
    tickers = [s + ".NS" for s in uni.trading_symbol]
    meta, failed = [], []
    for k in range(0, len(tickers), 50):
        batch = tickers[k:k + 50]
        df = yf.download(batch, start=START, auto_adjust=False, actions=False, group_by="ticker",
                         threads=True, progress=False)
        for t in batch:
            try:
                d = df[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
            except KeyError:
                d = pd.DataFrame()
            if len(d) < 100:
                failed.append(t)
                continue
            d.columns = ["open", "high", "low", "close", "volume"]
            d.index.name = "date"
            d.round(2).to_csv(a.out / "eq" / f"{t[:-3]}.csv")
        print(f"batch {k // 50 + 1}: {len(failed)} failed so far", flush=True)
        time.sleep(2)

    for _, r in uni.iterrows():
        t = r.trading_symbol + ".NS"
        if t in failed:
            continue
        shares = None
        for _ in range(3):
            try:
                shares = yf.Ticker(t).fast_info["shares"]
                break
            except Exception:
                time.sleep(3)
        meta.append(dict(symbol=r.trading_symbol, isin=r.isin, name=r["name"], amfi_rank=r["rank"], shares=shares))
        time.sleep(0.3)
    pd.DataFrame(meta).to_csv(a.out / "universe.csv", index=False)

    n = yf.download("^NSEI", start=START, auto_adjust=False, progress=False)
    n.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in n.columns]
    n = n[["open", "high", "low", "close"]].dropna()
    n.index.name = "date"
    n.round(2).to_csv(a.out / "nifty.csv")
    print(f"done: {len(meta)} stocks saved, {len(failed)} failed: {failed[:30]}")


if __name__ == "__main__":
    main()
