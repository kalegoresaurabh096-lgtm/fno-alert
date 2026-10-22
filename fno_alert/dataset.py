"""Turn the raw downloads (data/raw) into clean research tables (data/processed).

    fo_contracts.parquet  near-the-money option prices, for trade backtests
    opt_daily.parquet     one derivatives snapshot per stock per day
    ohlcv.parquet         corporate-action adjusted daily OHLCV per stock
    market.parquet        NIFTY and India VIX closes
    events.parquet        results board-meeting dates per stock (if available)
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import chain

FO_DTYPES = {"symbol": "string", "inst": "string", "fmt": "string"}


def read_fo_month(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=FO_DTYPES)
    df = df.drop_duplicates(subset=["date", "symbol", "inst", "expiry", "strike"], keep="last")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    for c in ("strike", "open", "high", "low", "close", "settle", "volume", "oi", "chg_oi", "lot",
              "underlying", "turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return normalise_units(df)


def _gcd_lot(oi: pd.Series) -> float:
    """Lot size = greatest common divisor of OI values (OI is a whole number of lots, in shares)."""
    v = oi[(oi > 0)].round().astype("int64").to_numpy()
    if len(v) < 3:
        return np.nan
    g = int(np.gcd.reduce(v))
    return float(g) if g > 0 else np.nan


def normalise_units(df: pd.DataFrame) -> pd.DataFrame:
    """Both bhavcopy formats already report volume in contracts and OI in shares
    (checked against real files), so nothing is rescaled. Only the lot size is
    filled in for old-format files, which do not carry it."""
    df = df.copy()
    old = df["fmt"] == "old"
    if old.any():
        lots = df[old].groupby(["date", "symbol"])["oi"].apply(_gcd_lot)
        # Cross-check with the turnover estimate; fall back to it when the GCD looks off.
        fut = old & (df["inst"] == "FUT") & (df["volume"] > 0)
        px = df.loc[fut, ["open", "high", "low", "close"]].mean(axis=1)
        est = (df.loc[fut, "turnover"] * 1e5 / (df.loc[fut, "volume"] * px)).groupby(
            [df.loc[fut, "date"], df.loc[fut, "symbol"]]).median()
        est = est.reindex(lots.index)
        bad = lots.isna() | ((est.notna()) & ((lots / est - 1).abs() > 0.25))
        lots = lots.where(~bad, est.round())
        key = pd.MultiIndex.from_arrays([df.loc[old, "date"], df.loc[old, "symbol"]])
        df.loc[old, "lot"] = lots.reindex(key).to_numpy()
    df.loc[~(df["volume"] > 0), "close"] = np.nan
    df.loc[df["close"] <= 0, "close"] = np.nan
    return df


def _snapshots_for_month(path: str):
    df = read_fo_month(path)
    rows = []
    contracts = []
    futs = df[df["inst"] == "FUT"][["date", "symbol", "expiry", "close", "oi", "chg_oi", "volume", "lot"]]
    for d, day in df.groupby("date"):
        for sym, g in day.groupby("symbol"):
            fut = g[g["inst"] == "FUT"][["expiry", "close", "oi", "chg_oi", "volume"]]
            opt = g[g["inst"] != "FUT"].rename(columns={"inst": "opt"})
            opt.loc[~(opt["volume"] > 0), "open"] = np.nan
            snap = chain.snapshot(opt[["expiry", "strike", "opt", "close", "oi", "chg_oi", "volume"]], fut, d)
            snap.update({"date": d, "symbol": sym, "lot": g["lot"].median()})
            rows.append(snap)
            # Keep near-the-money prices of the two nearest expiries for trade tests.
            exps = sorted(e for e in opt["expiry"].unique() if e >= d)[:2]
            near = opt[opt["expiry"].isin(exps)]
            contracts.append(near[["date", "symbol", "expiry", "strike", "opt", "open", "close", "settle",
                                   "volume", "oi"]])
    con = pd.concat(contracts) if contracts else pd.DataFrame()
    return rows, con, futs


def build_derivatives(raw_dir: str, out_dir: str, workers: int = 4) -> None:
    files = sorted(glob.glob(os.path.join(raw_dir, "fo", "fo_*.csv.gz")))
    if not files:
        raise SystemExit(f"No F&O files in {raw_dir}/fo")
    rows, cons, futs = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for i, (r, c, f) in enumerate(ex.map(_snapshots_for_month, files)):
            rows.extend(r)
            cons.append(c)
            futs.append(f)
            print(f"  derivatives: {i + 1}/{len(files)} months", flush=True)
    snap = pd.DataFrame(rows)
    snap["date"] = pd.to_datetime(snap["date"])
    snap["focus_expiry"] = pd.to_datetime(snap["focus_expiry"])
    snap.sort_values(["symbol", "date"]).to_parquet(os.path.join(out_dir, "opt_daily.parquet"), index=False)
    con = pd.concat(cons, ignore_index=True)
    con["date"] = pd.to_datetime(con["date"])
    con["expiry"] = pd.to_datetime(con["expiry"])
    con["symbol"] = con["symbol"].astype("category")
    con["opt"] = con["opt"].astype("category")
    con.to_parquet(os.path.join(out_dir, "fo_contracts.parquet"), index=False)
    fu = pd.concat(futs, ignore_index=True)
    fu["date"] = pd.to_datetime(fu["date"])
    fu["expiry"] = pd.to_datetime(fu["expiry"])
    fu.to_parquet(os.path.join(out_dir, "fut_contracts.parquet"), index=False)


def build_equity(raw_dir: str, out_dir: str, symbols: set[str]) -> pd.DataFrame:
    """Adjusted OHLCV from NSE's cash bhavcopy.

    NSE adjusts PREVCLOSE for splits/bonuses on the ex-date, so
    prev_close[t] / close[t-1] is the adjustment factor for everything
    before t. Yahoo data fills symbols or dates the bhavcopy does not cover.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "cm", "cm_*.csv.gz")))
    frames = [pd.read_csv(f) for f in files]
    out = []
    if frames:
        cm = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
        cm = cm[cm["symbol"].isin(symbols)]
        cm["date"] = pd.to_datetime(cm["date"])
        for sym, g in cm.sort_values("date").groupby("symbol"):
            g = g.set_index("date")
            factor = (g["prev_close"] / g["close"].shift(1)).fillna(1.0)
            # Ignore tiny rounding differences; real corporate actions are big.
            factor = factor.where((factor - 1).abs() > 0.02, 1.0)
            # Cumulative factor applied to all earlier rows.
            cum = factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
            adj = pd.DataFrame({
                "open": g["open"] * cum, "high": g["high"] * cum, "low": g["low"] * cum,
                "close": g["close"] * cum, "volume": g["volume"] / cum, "close_raw": g["close"],
                "corp_action": (factor != 1.0).astype(int),
            })
            adj["symbol"] = sym
            out.append(adj.reset_index())
    eq = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume", "corp_action", "symbol", "close_raw"])
    eq["source"] = "nse"
    # Fill from Yahoo where NSE CM data is missing.
    ydir = os.path.join(raw_dir, "yahoo")
    have = set(eq["symbol"].unique())
    extra = []
    for sym in sorted(symbols - have):
        p = os.path.join(ydir, f"{sym}.csv")
        if os.path.exists(p):
            y = pd.read_csv(p, parse_dates=["date"]).dropna(subset=["close"])
            y["symbol"] = sym
            y["corp_action"] = 0
            y["source"] = "yahoo"
            y["close_raw"] = np.nan   # Yahoo closes are split-adjusted
            extra.append(y[["date", "open", "high", "low", "close", "volume", "corp_action", "symbol", "source",
                            "close_raw"]])
    if extra:
        eq = pd.concat([eq] + extra, ignore_index=True)
    eq = eq.sort_values(["symbol", "date"])
    eq.to_parquet(os.path.join(out_dir, "ohlcv.parquet"), index=False)
    return eq


def build_market(raw_dir: str, out_dir: str) -> pd.DataFrame:
    ydir = os.path.join(raw_dir, "yahoo")
    cols = {}
    for name, sym in (("nifty", "NIFTY"), ("vix", "INDIAVIX")):
        p = os.path.join(ydir, f"{sym}.csv")
        if os.path.exists(p):
            y = pd.read_csv(p, parse_dates=["date"]).dropna(subset=["close"]).drop_duplicates("date", keep="last")
            cols[name] = y.set_index("date")["close"]
    mkt = pd.DataFrame(cols)
    mkt.to_parquet(os.path.join(out_dir, "market.parquet"))
    return mkt


def build_events(raw_dir: str, out_dir: str) -> pd.DataFrame:
    p = os.path.join(raw_dir, "events", "board_meetings.json")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["symbol", "date", "announced"])
    rows = json.load(open(p))
    recs = []
    for r in rows:
        purpose = f"{r.get('bm_purpose', '')} {r.get('bm_desc', '')}".lower()
        if "result" not in purpose and "financial" not in purpose:
            continue
        d = pd.to_datetime(r.get("bm_date"), dayfirst=True, errors="coerce")
        a = pd.to_datetime(r.get("bm_timestamp") or r.get("sys_time"), dayfirst=True, errors="coerce")
        if pd.notna(d):
            recs.append({"symbol": r.get("bm_symbol"), "date": d.normalize(), "announced": a})
    ev = pd.DataFrame(recs).drop_duplicates(["symbol", "date"])
    ev.to_parquet(os.path.join(out_dir, "events.parquet"), index=False)
    return ev


def build_all(raw_dir: str = "data/raw", out_dir: str = "data/processed", workers: int = 4) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print("Building derivatives tables ...", flush=True)
    build_derivatives(raw_dir, out_dir, workers)
    syms = set(pd.read_parquet(os.path.join(out_dir, "opt_daily.parquet"), columns=["symbol"])["symbol"])
    print("Building adjusted price history ...", flush=True)
    build_equity(raw_dir, out_dir, syms)
    build_market(raw_dir, out_dir)
    build_events(raw_dir, out_dir)
    print("Done:", out_dir)
