"""NSE end-of-day F&O bhavcopy: download and normalise both file formats.

* Old format, until early July 2024:
  .../content/historical/DERIVATIVES/2024/JUN/fo28JUN2024bhav.csv.zip
* UDiFF format, from 8 July 2024:
  .../content/fo/BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip

Both are normalised to one long table of *stock* futures and options only
(index contracts are dropped):

date, symbol, inst ('FUT'|'CE'|'PE'), expiry, strike, open, high, low, close,
settle, volume, oi, chg_oi, lot, underlying

`close` is NaN when the contract did not trade that day; `volume` is in
contracts, `oi` and `chg_oi` in shares.
"""
from __future__ import annotations

import datetime as dt
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

HOSTS = ("https://nsearchives.nseindia.com", "https://archives.nseindia.com")
UDIFF_START = dt.date(2024, 7, 8)
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
COLUMNS = ["date", "symbol", "inst", "expiry", "strike", "open", "high", "low", "close",
           "settle", "volume", "oi", "chg_oi", "lot", "underlying"]


def urls_for(d: dt.date) -> list[str]:
    mon = d.strftime("%b").upper()
    old = f"/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"
    new = f"/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    paths = [new, old] if d >= UDIFF_START else [old, new]
    return [h + p for p in paths for h in HOSTS]


class NotPublished(Exception):
    """No file for this date (holiday, weekend, or not yet uploaded)."""


def fetch_raw(d: dt.date, session: requests.Session | None = None, timeout: int = 30,
              retries: int = 3) -> pd.DataFrame:
    s = session or requests.Session()
    last_err = None
    for url in urls_for(d):
        for attempt in range(retries):
            try:
                r = s.get(url, headers=HEADERS, timeout=timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                break
            if r.status_code != 200 or not r.content.startswith(b"PK"):
                last_err = RuntimeError(f"{url}: HTTP {r.status_code}")
                time.sleep(2 ** attempt)
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
                return pd.read_csv(z.open(name), low_memory=False)
    if last_err:
        raise last_err
    raise NotPublished(str(d))


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def normalise(raw: pd.DataFrame, d: dt.date | None = None) -> pd.DataFrame:
    cols = {c.strip(): c for c in raw.columns}
    raw = raw.rename(columns={v: k for k, v in cols.items()})
    if "INSTRUMENT" in raw.columns:
        return _normalise_old(raw)
    if "FinInstrmTp" in raw.columns:
        return _normalise_udiff(raw)
    raise ValueError(f"Unknown bhavcopy columns: {list(raw.columns)[:10]}")


def _normalise_old(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["INSTRUMENT"].isin(["FUTSTK", "OPTSTK"])].copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["TIMESTAMP"], format="%d-%b-%Y").dt.date,
        "symbol": raw["SYMBOL"].str.strip(),
        "inst": np.where(raw["INSTRUMENT"] == "FUTSTK", "FUT", raw["OPTION_TYP"].str.strip()),
        "expiry": pd.to_datetime(raw["EXPIRY_DT"], format="%d-%b-%Y").dt.date,
        "strike": _num(raw["STRIKE_PR"]),
        "open": _num(raw["OPEN"]), "high": _num(raw["HIGH"]), "low": _num(raw["LOW"]),
        "close": _num(raw["CLOSE"]), "settle": _num(raw["SETTLE_PR"]),
        "volume": _num(raw["CONTRACTS"]),
        "oi": _num(raw["OPEN_INT"]), "chg_oi": _num(raw["CHG_IN_OI"]),
        "lot": np.nan, "underlying": np.nan,
    })
    out.loc[out["inst"] == "FUT", "strike"] = np.nan
    # Lot size is not in the old file: back it out of futures turnover
    # (VAL_INLAKH = contracts * lot * avg price / 1e5) and share it per symbol.
    fut = raw["INSTRUMENT"] == "FUTSTK"
    vwap = (_num(raw["OPEN"]) + _num(raw["HIGH"]) + _num(raw["LOW"]) + _num(raw["CLOSE"])) / 4
    lot_est = _num(raw["VAL_INLAKH"]) * 1e5 / (_num(raw["CONTRACTS"]) * vwap)
    lot_by_sym = lot_est[fut & (_num(raw["CONTRACTS"]) > 0)].groupby(raw["SYMBOL"].str.strip()[fut]).median()
    out["lot"] = out["symbol"].map(lot_by_sym)
    return _finish(out)


def _normalise_udiff(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["FinInstrmTp"].isin(["STF", "STO"])].copy()
    vol = _num(raw["TtlTradgVol"])
    lot = _num(raw.get("NewBrdLotQty"))
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["TradDt"]).dt.date,
        "symbol": raw["TckrSymb"].str.strip(),
        "inst": np.where(raw["FinInstrmTp"] == "STF", "FUT", raw["OptnTp"].astype(str).str.strip()),
        "expiry": pd.to_datetime(raw["XpryDt"]).dt.date,
        "strike": _num(raw["StrkPric"]),
        "open": _num(raw["OpnPric"]), "high": _num(raw["HghPric"]), "low": _num(raw["LwPric"]),
        "close": _num(raw["ClsPric"]), "settle": _num(raw["SttlmPric"]),
        "volume": vol, "oi": _num(raw["OpnIntrst"]), "chg_oi": _num(raw["ChngInOpnIntrst"]),
        "lot": lot, "underlying": _num(raw.get("UndrlygPric")),
    })
    out.loc[out["inst"] == "FUT", "strike"] = np.nan
    # UDiFF reports volume in shares on some days/versions and contracts on
    # others; detect it from the lot size (volume in shares is a lot multiple).
    traded = (out["volume"] > 0) & (out["lot"] > 0)
    if traded.any():
        frac_multiple = ((out.loc[traded, "volume"] % out.loc[traded, "lot"]) == 0).mean()
        big_share = (out.loc[traded, "volume"] >= out.loc[traded, "lot"]).mean()
        if frac_multiple > 0.98 and big_share > 0.98:
            out.loc[traded, "volume"] = out.loc[traded, "volume"] / out.loc[traded, "lot"]
    return _finish(out)


def _finish(out: pd.DataFrame) -> pd.DataFrame:
    out.loc[~(out["volume"] > 0), "close"] = np.nan
    out.loc[out["close"] <= 0, "close"] = np.nan
    keep = (out["oi"] > 0) | (out["volume"] > 0)
    out = out[keep]
    return out[COLUMNS].reset_index(drop=True)


def split_for_snapshot(day: pd.DataFrame):
    """Yield (symbol, options_df, futures_df) in the shape chain.snapshot wants."""
    for sym, g in day.groupby("symbol", sort=True):
        fut = g[g["inst"] == "FUT"][["expiry", "close", "oi", "chg_oi", "volume"]]
        opt = g[g["inst"] != "FUT"].rename(columns={"inst": "opt"})[
            ["expiry", "strike", "opt", "close", "oi", "chg_oi", "volume"]]
        yield sym, opt, fut
