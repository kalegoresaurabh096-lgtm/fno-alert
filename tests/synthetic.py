"""Synthetic NSE-like raw data for pipeline tests (no network needed).

Volatility switches between calm and stormy regimes. Implied vol and
option volume rise a few days *before* a storm, so a working pipeline
should find a real (if artificial) signal, and a broken one won't.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import os

import numpy as np

from fno_alert import black76

FO_FIELDS = ["date", "symbol", "inst", "expiry", "strike", "open", "high", "low", "close",
             "settle", "volume", "oi", "chg_oi", "lot", "underlying", "turnover", "fmt"]
CM_FIELDS = ["date", "symbol", "series", "open", "high", "low", "close", "prev_close", "volume", "turnover"]


def last_thursdays(start: dt.date, end: dt.date):
    out = []
    y, m = start.year, start.month
    while dt.date(y, m, 1) <= end + dt.timedelta(days=120):
        nm = dt.date(y + (m // 12), m % 12 + 1, 1)
        d = nm - dt.timedelta(days=1)
        while d.weekday() != 3:
            d -= dt.timedelta(days=1)
        out.append(d)
        y, m = nm.year, nm.month
    return out


def generate(raw_dir: str, n_symbols: int = 30, start=dt.date(2022, 1, 3), n_days: int = 520, seed: int = 7):
    rng = np.random.default_rng(seed)
    days = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    expiries = last_thursdays(start, days[-1])
    os.makedirs(os.path.join(raw_dir, "fo"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "cm"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "yahoo"), exist_ok=True)
    fo_rows: dict[str, list] = {}
    cm_rows: dict[str, list] = {}
    for s in range(n_symbols):
        sym = f"SYM{s:02d}"
        lot = int(rng.choice([250, 500, 1000, 1500]))
        base_vol = rng.uniform(0.012, 0.025)
        storm = np.zeros(n_days, bool)
        t = 30
        while t < n_days:
            t += int(rng.integers(25, 70))
            storm[t:t + int(rng.integers(3, 8))] = True
        sig = np.where(storm, base_vol * 3.0, base_vol)
        # Pre-storm warning appears 1-4 days ahead in IV and option volume.
        warn = np.zeros(n_days)
        for i in np.flatnonzero(storm & ~np.roll(storm, 1)):
            lead = int(rng.integers(1, 5))
            warn[max(i - lead, 0):i] = 1
        ret = rng.standard_t(5, n_days) / np.sqrt(5 / 3) * sig
        close = 500 * np.exp(np.cumsum(ret)) * rng.uniform(0.5, 4)
        if s == 0:  # one 1:2 split to test corporate-action adjustment
            close[n_days // 2:] /= 2
        opn = close / np.exp(ret) * np.exp(rng.normal(0, sig / 4))
        high = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0, sig / 2)))
        low = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0, sig / 2)))
        vol = (rng.lognormal(13, 0.3, n_days) * (1 + 1.5 * storm + 0.6 * warn)).astype(int)
        iv_base = base_vol * np.sqrt(252) * 1.1
        iv = iv_base * (1 + 0.8 * warn + 1.2 * storm) * np.exp(rng.normal(0, 0.05, n_days))
        fut_oi = lot * 2000.0
        for i, day in enumerate(days):
            key = day.strftime("%Y%m")
            prev = close[i - 1] if i else close[i]
            if s == 0 and i == n_days // 2:
                prev = close[i - 1] / 2   # NSE adjusts PREVCLOSE on the ex-date
            cm_rows.setdefault(key, []).append([day.isoformat(), sym, "EQ", opn[i], high[i], low[i], close[i],
                                                prev, vol[i], vol[i] * close[i]])
            live = [e for e in expiries if e >= day][:3]
            step = 10 ** np.floor(np.log10(close[i] * 0.025))
            step = step * (5 if close[i] * 0.025 / step > 5 else 2 if close[i] * 0.025 / step > 2 else 1)
            atm = round(close[i] / step) * step
            for e in live:
                T = max((e - day).days, 0.5) / 365
                F = close[i] * np.exp(0.065 * T)
                d_oi = lot * round(rng.normal(0.01 + 0.04 * warn[i], 0.02) * fut_oi / lot)
                fut_oi = max(fut_oi + d_oi, lot * 100.0)
                fo_rows.setdefault(key, []).append(
                    [day.isoformat(), sym, "FUT", e.isoformat(), "", F, F, F, F, F,
                     int(vol[i] / lot / 3), fut_oi, d_oi, lot, close[i], "", "udiff"])
                for k in range(-10, 11):
                    K = atm + k * step
                    if K <= 0:
                        continue
                    for opt in ("CE", "PE"):
                        smile = iv[i] * (1 + 0.5 * ((K / F - 1) ** 2) * 10)
                        px = float(black76.price(F, K, T, smile, opt == "CE"))
                        traded = abs(k) <= 6 or rng.random() < 0.3
                        v = int(rng.lognormal(4, 1) * (1 + 2 * warn[i] + 2 * storm[i])) if traded else 0
                        oi = lot * round(rng.integers(50, 600) / (1 + abs(k) / 3))
                        chg = lot * round(rng.normal(0, 0.05) * oi / lot)
                        fo_rows[key].append([day.isoformat(), sym, opt, e.isoformat(), K, px, px, px,
                                             round(px, 2), round(px, 2), v, oi, chg,
                                             lot, close[i], "", "udiff"])
    for key, rows in fo_rows.items():
        with gzip.open(os.path.join(raw_dir, "fo", f"fo_{key}.csv.gz"), "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FO_FIELDS)
            w.writerows(rows)
    for key, rows in cm_rows.items():
        with gzip.open(os.path.join(raw_dir, "cm", f"cm_{key}.csv.gz"), "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CM_FIELDS)
            w.writerows(rows)
    # Market series
    nifty = 18000 * np.exp(np.cumsum(rng.normal(0, 0.009, n_days)))
    vix = 14 * np.exp(np.cumsum(rng.normal(0, 0.03, n_days)) * 0.3)
    for name, series in (("NIFTY", nifty), ("INDIAVIX", vix)):
        with open(os.path.join(raw_dir, "yahoo", f"{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "open", "high", "low", "close", "adjclose", "volume"])
            for day, v in zip(days, series):
                w.writerow([day.isoformat(), v, v, v, v, v, 0])
    return days
