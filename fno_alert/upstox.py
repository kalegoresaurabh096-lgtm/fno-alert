"""Upstox API client (read-only market data).

Uses the 1-year read-only *Analytics Token* (Upstox Developer Apps ->
Analytics -> Generate Token). It can only read market data; it cannot place
orders or see your holdings. Data APIs need no static IP.

Docs: https://upstox.com/developer/api-documentation/
"""
from __future__ import annotations

import collections
import datetime as dt
import gzip
import io
import json
import math
import os
import time
import urllib.parse

import numpy as np
import pandas as pd
import requests

BASE = "https://api.upstox.com"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


class UpstoxError(RuntimeError):
    pass


class RateLimiter:
    """Sliding-window limiter kept safely under Upstox's per-user limits
    (50/s, 500/min, 2000/30min for standard APIs)."""

    def __init__(self, per_sec=8, per_min=400, per_30min=1800):
        self.windows = [(1.0, per_sec), (60.0, per_min), (1800.0, per_30min)]
        self.calls = collections.deque()

    def wait(self):
        while True:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 1800:
                self.calls.popleft()
            sleep_for = 0.0
            for span, limit in self.windows:
                recent = [t for t in self.calls if now - t <= span]
                if len(recent) >= limit:
                    sleep_for = max(sleep_for, span - (now - recent[0]) + 0.01)
            if sleep_for <= 0:
                self.calls.append(now)
                return
            time.sleep(min(sleep_for, 60))


class Upstox:
    def __init__(self, token: str | None = None, session: requests.Session | None = None,
                 limiter: RateLimiter | None = None, cache_dir: str = "state/cache"):
        self.token = token or os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        if not self.token:
            raise UpstoxError("Set UPSTOX_ACCESS_TOKEN (the read-only Analytics Token).")
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.token}", "Accept": "application/json"})
        self.limiter = limiter or RateLimiter()
        self.cache_dir = cache_dir
        self.calls = 0

    # ------------------------------------------------------------------ http
    def get(self, path: str, params: dict | None = None, retries: int = 4):
        url = path if path.startswith("http") else BASE + path
        for attempt in range(retries):
            self.limiter.wait()
            self.calls += 1
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise UpstoxError(f"network error on {path}: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                js = r.json()
                if js.get("status") not in (None, "success"):
                    raise UpstoxError(f"{path}: {js}")
                return js.get("data")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * (5 if r.status_code == 429 else 1))
                continue
            if r.status_code == 401:
                raise UpstoxError("Upstox rejected the token (401). The Analytics Token may have "
                                  "expired or been revoked: generate a new one and update the secret.")
            raise UpstoxError(f"{path}: HTTP {r.status_code} {r.text[:300]}")
        raise UpstoxError(f"{path}: gave up after {retries} attempts")

    # ----------------------------------------------------------- instruments
    def instruments(self, max_age_hours: float = 12) -> pd.DataFrame:
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, "upstox_NSE.json.gz")
        if not (os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_hours * 3600):
            r = requests.get(INSTRUMENTS_URL, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as fh:
                fh.write(r.content)
        with gzip.open(path, "rt") as fh:
            data = json.load(fh)
        return pd.DataFrame(data)

    @staticmethod
    def fno_universe(instr: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
        """One row per F&O *stock* (indices excluded)."""
        today = today or dt.datetime.now(IST).date()
        fo = instr[(instr["segment"] == "NSE_FO") & (instr.get("underlying_type") == "EQUITY")].copy()
        fo["expiry_date"] = pd.to_datetime(fo["expiry"], unit="ms", utc=True).dt.tz_convert(IST).dt.date
        fo = fo[fo["expiry_date"] >= today]
        rows = []
        for sym, g in fo.groupby("underlying_symbol"):
            fut = g[g["instrument_type"] == "FUT"].sort_values("expiry_date")
            opt = g[g["instrument_type"].isin(["CE", "PE"])]
            if fut.empty or opt.empty:
                continue
            rows.append({
                "symbol": sym, "underlying_key": g["underlying_key"].iloc[0],
                "lot": float(fut["lot_size"].iloc[0]),
                "futures": list(zip(fut["expiry_date"], fut["instrument_key"])),
                "option_expiries": sorted(opt["expiry_date"].unique()),
                "name": g["name"].iloc[0],
            })
        return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    # ------------------------------------------------------------ market data
    def option_chain(self, underlying_key: str, expiry: dt.date) -> list:
        return self.get("/v2/option/chain", {"instrument_key": underlying_key,
                                             "expiry_date": expiry.isoformat()}) or []

    def quotes(self, keys: list[str]) -> dict:
        """Full market quotes, returned keyed by instrument key (NSE_FO|12345)."""
        out = {}
        for i in range(0, len(keys), 450):
            chunk = keys[i:i + 450]
            data = self.get("/v2/market-quote/quotes", {"instrument_key": ",".join(chunk)}) or {}
            for _, v in data.items():
                out[v.get("instrument_token")] = v
        return out

    def daily_candles(self, key: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        path = (f"/v3/historical-candle/{urllib.parse.quote(key, safe='')}/days/1/"
                f"{end.isoformat()}/{start.isoformat()}")
        data = self.get(path) or {}
        return candles_to_frame(data.get("candles", []))

    def expired_expiries(self, underlying_key: str) -> list[dt.date]:
        data = self.get("/v2/expired-instruments/expiries", {"instrument_key": underlying_key}) or []
        return sorted(dt.date.fromisoformat(d) for d in data)

    def expired_contracts(self, underlying_key: str, expiry: dt.date, kind: str = "option") -> pd.DataFrame:
        data = self.get(f"/v2/expired-instruments/{kind}/contract",
                        {"instrument_key": underlying_key, "expiry_date": expiry.isoformat()}) or []
        return pd.DataFrame(data)

    def expired_candles(self, expired_key: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        path = (f"/v2/expired-instruments/historical-candle/{urllib.parse.quote(expired_key, safe='')}/day/"
                f"{end.isoformat()}/{start.isoformat()}")
        data = self.get(path) or {}
        return candles_to_frame(data.get("candles", []))


def candles_to_frame(candles: list) -> pd.DataFrame:
    cols = ["ts", "open", "high", "low", "close", "volume", "oi"]
    if not candles:
        return pd.DataFrame(columns=["date"] + cols[1:])
    df = pd.DataFrame([c[:7] + [0] * (7 - len(c[:7])) for c in candles], columns=cols)
    df["date"] = pd.to_datetime(df["ts"].str[:10])
    return df.drop(columns="ts").sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def chain_to_frames(chain: list, lot: float):
    """Upstox option chain -> (options frame for chain.snapshot, spot, upstox IV frame)."""
    rows = []
    spot = math.nan
    for item in chain:
        K = float(item.get("strike_price"))
        exp = dt.date.fromisoformat(item.get("expiry"))
        spot = float(item.get("underlying_spot_price") or spot)
        for side, opt in (("call_options", "CE"), ("put_options", "PE")):
            leg = item.get(side) or {}
            md = leg.get("market_data") or {}
            gk = leg.get("option_greeks") or {}
            if not md:
                continue
            oi = float(md.get("oi") or 0)
            prev = float(md.get("prev_oi") or 0)
            vol = float(md.get("volume") or 0)
            ltp = float(md.get("ltp") or 0)
            rows.append({"expiry": exp, "strike": K, "opt": opt, "close": ltp if (ltp > 0 and vol > 0) else np.nan,
                         "oi": oi, "chg_oi": oi - prev, "volume": vol, "upstox_iv": gk.get("iv"),
                         "bid": md.get("bid_price"), "ask": md.get("ask_price")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, spot
    # Normalise units to the bhavcopy convention: volume in contracts, OI in shares.
    if lot and lot > 1:
        traded = df["volume"] > 0
        if traded.sum() >= 10 and ((df.loc[traded, "volume"] % lot) == 0).mean() > 0.97:
            df["volume"] = df["volume"] / lot
        has_oi = df["oi"] > 0
        if has_oi.sum() >= 10 and ((df.loc[has_oi, "oi"] % lot) == 0).mean() < 0.5:
            df["oi"] = df["oi"] * lot
            df["chg_oi"] = df["chg_oi"] * lot
    return df, spot
