#!/usr/bin/env python3
"""Download NSE history for backtesting. Standard library only (Python 3.8+).

Run it from the project folder on a normal home internet connection:

    python3 scripts/download_history.py --start 2021-01-01

It is safe to stop (Ctrl+C) and re-run: finished days are skipped.

What it saves under data/raw/:
  fo/fo_YYYYMM.csv.gz   stock futures + options (index contracts dropped,
                        option strikes beyond +-25% of the price dropped)
  cm/cm_YYYYMM.csv.gz   cash-market daily OHLCV for F&O stocks (unadjusted)
  yahoo/<SYMBOL>.csv    adjusted daily OHLCV from Yahoo Finance
  events/board_meetings.json   results board meetings (best effort)
  progress.json         which days are done / were holidays
"""
import argparse
import csv
import datetime as dt
import gzip
import http.cookiejar
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HOSTS = ("https://nsearchives.nseindia.com", "https://archives.nseindia.com")
UDIFF_START = dt.date(2024, 7, 8)
STRIKE_BAND = 0.35
FO_FIELDS = ["date", "symbol", "inst", "expiry", "strike", "open", "high", "low", "close",
             "settle", "volume", "oi", "chg_oi", "lot", "underlying", "turnover", "fmt"]
CM_FIELDS = ["date", "symbol", "series", "open", "high", "low", "close", "prev_close",
             "volume", "turnover"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Http:
    def __init__(self):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def get(self, url, referer="https://www.nseindia.com/", timeout=40, accept="*/*"):
        req = urllib.request.Request(url, headers={
            "User-Agent": UA, "Accept": accept, "Accept-Language": "en-US,en;q=0.9",
            "Referer": referer})
        with self.opener.open(req, timeout=timeout) as r:
            return r.read()


def fetch_zip_csv(http, urls, retries=3):
    """Return CSV text from the first URL that works, None if all are 404."""
    saw_error = None
    for url in urls:
        for attempt in range(retries):
            try:
                data = http.get(url)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                saw_error = e
            except Exception as e:  # network hiccup
                saw_error = e
            else:
                if data[:2] != b"PK":
                    saw_error = RuntimeError("not a zip: " + url)
                else:
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
                        return z.read(name).decode("utf-8", "replace")
            time.sleep(2 ** attempt + random.random())
    if saw_error is not None:
        raise saw_error
    return None


def fo_urls(d):
    mon = d.strftime("%b").upper()
    old = "/content/historical/DERIVATIVES/%d/%s/fo%02d%s%dbhav.csv.zip" % (d.year, mon, d.day, mon, d.year)
    new = "/content/fo/BhavCopy_NSE_FO_0_0_0_%s_F_0000.csv.zip" % d.strftime("%Y%m%d")
    paths = [new, old] if d >= UDIFF_START else [old, new]
    return [h + p for p in paths for h in HOSTS]


def cm_urls(d):
    mon = d.strftime("%b").upper()
    old = "/content/historical/EQUITIES/%d/%s/cm%02d%s%dbhav.csv.zip" % (d.year, mon, d.day, mon, d.year)
    new = "/content/cm/BhavCopy_NSE_CM_0_0_0_%s_F_0000.csv.zip" % d.strftime("%Y%m%d")
    paths = [new, old] if d >= UDIFF_START else [old, new]
    return [h + p for p in paths for h in HOSTS]


def _f(x):
    x = (x or "").strip()
    if x in ("", "-", "NA", "nan"):
        return None
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return None


def _iso(s, fmts=("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d%b%Y")):
    s = (s or "").strip()
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    return s


def parse_fo(text, d):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return []
    rows = [{k.strip(): (v or "") for k, v in r.items() if k} for r in rows]
    out = []
    if "INSTRUMENT" in rows[0]:
        for r in rows:
            ins = r["INSTRUMENT"].strip()
            if ins not in ("FUTSTK", "OPTSTK"):
                continue
            out.append({
                "date": d.isoformat(), "symbol": r["SYMBOL"].strip(),
                "inst": "FUT" if ins == "FUTSTK" else r["OPTION_TYP"].strip(),
                "expiry": _iso(r["EXPIRY_DT"]), "strike": _f(r["STRIKE_PR"]) if ins == "OPTSTK" else None,
                "open": _f(r["OPEN"]), "high": _f(r["HIGH"]), "low": _f(r["LOW"]), "close": _f(r["CLOSE"]),
                "settle": _f(r["SETTLE_PR"]), "volume": _f(r["CONTRACTS"]), "oi": _f(r["OPEN_INT"]),
                "chg_oi": _f(r["CHG_IN_OI"]), "lot": None, "underlying": None,
                "turnover": _f(r.get("VAL_INLAKH")), "fmt": "old"})
    else:
        for r in rows:
            tp = r.get("FinInstrmTp", "").strip()
            if tp not in ("STF", "STO"):
                continue
            out.append({
                "date": d.isoformat(), "symbol": r["TckrSymb"].strip(),
                "inst": "FUT" if tp == "STF" else r["OptnTp"].strip(),
                "expiry": _iso(r["XpryDt"]), "strike": _f(r["StrkPric"]) if tp == "STO" else None,
                "open": _f(r["OpnPric"]), "high": _f(r["HghPric"]), "low": _f(r["LwPric"]),
                "close": _f(r["ClsPric"]), "settle": _f(r["SttlmPric"]), "volume": _f(r["TtlTradgVol"]),
                "oi": _f(r["OpnIntrst"]), "chg_oi": _f(r["ChngInOpnIntrst"]),
                "lot": _f(r.get("NewBrdLotQty")), "underlying": _f(r.get("UndrlygPric")),
                "turnover": _f(r.get("TtlTrfVal")), "fmt": "udiff"})
    # Drop dead contracts and far strikes to keep files small.
    ref = {}
    for r in out:
        if r["inst"] == "FUT" and r["close"]:
            e = r["expiry"]
            if r["symbol"] not in ref or e < ref[r["symbol"]][0]:
                ref[r["symbol"]] = (e, r["close"])
    kept = []
    for r in out:
        if not ((r["oi"] or 0) > 0 or (r["volume"] or 0) > 0):
            continue
        if r["inst"] != "FUT":
            px = (ref.get(r["symbol"]) or (None, r["underlying"]))[1]
            if px and r["strike"] and abs(r["strike"] / px - 1) > STRIKE_BAND:
                continue
        kept.append(r)
    return kept


def parse_cm(text, d, symbols):
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        r = {k.strip(): (v or "") for k, v in r.items() if k}
        if "SYMBOL" in r:
            sym, ser = r["SYMBOL"].strip(), r["SERIES"].strip()
            rec = [r["OPEN"], r["HIGH"], r["LOW"], r["CLOSE"], r["PREVCLOSE"], r["TOTTRDQTY"], r["TOTTRDVAL"]]
        else:
            sym, ser = r.get("TckrSymb", "").strip(), r.get("SctySrs", "").strip()
            rec = [r.get("OpnPric"), r.get("HghPric"), r.get("LwPric"), r.get("ClsPric"),
                   r.get("PrvsClsgPric"), r.get("TtlTradgVol"), r.get("TtlTrfVal")]
        if ser != "EQ" or (symbols and sym not in symbols):
            continue
        vals = [_f(x) for x in rec]
        out.append(dict(zip(CM_FIELDS, [d.isoformat(), sym, ser] + vals)))
    return out


def append_gz(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with gzip.open(path, "at", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def load_progress():
    p = os.path.join(RAW, "progress.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {"fo": {}, "cm": {}}


def save_progress(prog):
    p = os.path.join(RAW, "progress.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prog, fh)
    os.replace(tmp, p)


def weekdays(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def download_bhavcopies(http, start, end, with_cm, delay):
    prog = load_progress()
    symbols = set()
    days = list(weekdays(start, end))
    t0 = time.time()
    for i, d in enumerate(days):
        key = d.isoformat()
        month = d.strftime("%Y%m")
        if key not in prog["fo"]:
            try:
                text = fetch_zip_csv(http, fo_urls(d))
            except Exception as e:
                log("FO", key, "error, will retry next run:", e)
                time.sleep(5)
                continue
            if text is None:
                prog["fo"][key] = "holiday"
            else:
                rows = parse_fo(text, d)
                append_gz(os.path.join(RAW, "fo", "fo_%s.csv.gz" % month), FO_FIELDS, rows)
                prog["fo"][key] = len(rows)
                symbols.update(r["symbol"] for r in rows)
            save_progress(prog)
            time.sleep(delay)
        if with_cm and key not in prog["cm"] and prog["fo"].get(key) != "holiday":
            try:
                text = fetch_zip_csv(http, cm_urls(d))
            except Exception as e:
                log("CM", key, "error, will retry next run:", e)
                time.sleep(5)
                continue
            if text is None:
                prog["cm"][key] = "holiday"
            else:
                rows = parse_cm(text, d, None)
                append_gz(os.path.join(RAW, "cm", "cm_%s.csv.gz" % month), CM_FIELDS, rows)
                prog["cm"][key] = len(rows)
            save_progress(prog)
            time.sleep(delay)
        if i % 20 == 0:
            done = sum(1 for k in prog["fo"] if k <= key)
            eta = (time.time() - t0) / (i + 1) * (len(days) - i - 1) / 60
            log("bhavcopy %s  (%d/%d days, about %.0f min left)" % (key, i + 1, len(days), eta))
    return prog


def symbols_from_fo():
    syms = set()
    d = os.path.join(RAW, "fo")
    if not os.path.isdir(d):
        return syms
    for name in sorted(os.listdir(d)):
        with gzip.open(os.path.join(d, name), "rt") as fh:
            for r in csv.DictReader(fh):
                syms.add(r["symbol"])
    return syms


def yahoo_symbol(sym):
    return {"NIFTY": "^NSEI", "INDIAVIX": "^INDIAVIX"}.get(sym, sym + ".NS")


def download_yahoo(http, symbols, start):
    out_dir = os.path.join(RAW, "yahoo")
    os.makedirs(out_dir, exist_ok=True)
    p1 = int(time.mktime((start - dt.timedelta(days=500)).timetuple()))
    p2 = int(time.time()) + 86400
    failed = []
    for i, sym in enumerate(sorted(symbols)):
        path = os.path.join(out_dir, sym + ".csv")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600:
            continue
        ys = urllib.parse.quote(yahoo_symbol(sym))
        url = ("https://query2.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d"
               "&interval=1d&events=div%%7Csplit&includeAdjustedClose=true" % (ys, p1, p2))
        try:
            js = json.loads(http.get(url, referer="https://finance.yahoo.com/", accept="application/json"))
            res = js["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(ts))
            splits = res.get("events", {}).get("splits", {})
        except Exception as e:
            failed.append(sym)
            log("yahoo", sym, "failed:", e)
            time.sleep(2)
            continue
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "open", "high", "low", "close", "adjclose", "volume"])
            for j, t in enumerate(ts):
                day = dt.datetime.utcfromtimestamp(t + 19800).date().isoformat()  # IST
                w.writerow([day, q["open"][j], q["high"][j], q["low"][j], q["close"][j], adj[j], q["volume"][j]])
        if splits:
            with open(os.path.join(out_dir, sym + ".splits.json"), "w") as fh:
                json.dump(splits, fh)
        if i % 25 == 0:
            log("yahoo %d/%d %s" % (i + 1, len(symbols), sym))
        time.sleep(0.4)
    return failed


def download_board_meetings(http, start, end):
    """Best effort: NSE's website API needs a cookie from the home page first."""
    out = os.path.join(RAW, "events", "board_meetings.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        http.get("https://www.nseindia.com/", accept="text/html")
        http.get("https://www.nseindia.com/companies-listing/corporate-filings-board-meetings", accept="text/html")
    except Exception as e:
        log("board meetings: could not open nseindia.com:", e)
        return False
    allrows = []
    d = start
    while d <= end:
        e = min(d + dt.timedelta(days=89), end)
        url = ("https://www.nseindia.com/api/corporate-board-meetings?index=equities&from_date=%s&to_date=%s"
               % (d.strftime("%d-%m-%Y"), e.strftime("%d-%m-%Y")))
        try:
            js = json.loads(http.get(url, referer="https://www.nseindia.com/companies-listing/corporate-filings-board-meetings",
                                     accept="application/json"))
            rows = js if isinstance(js, list) else js.get("data", [])
            allrows.extend(rows)
            log("board meetings %s..%s: %d" % (d, e, len(rows)))
        except Exception as ex:
            log("board meetings %s..%s failed: %s" % (d, e, ex))
        d = e + dt.timedelta(days=1)
        time.sleep(1.5)
    with open(out, "w") as fh:
        json.dump(allrows, fh)
    return bool(allrows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between NSE requests")
    ap.add_argument("--no-cm", action="store_true")
    ap.add_argument("--skip-bhavcopy", action="store_true")
    ap.add_argument("--skip-yahoo", action="store_true")
    ap.add_argument("--skip-events", action="store_true")
    a = ap.parse_args()
    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end)
    http = Http()
    os.makedirs(RAW, exist_ok=True)
    if not a.skip_bhavcopy:
        log("Downloading NSE bhavcopies %s to %s ..." % (start, end))
        download_bhavcopies(http, start, end, not a.no_cm, a.delay)
    if not a.skip_yahoo:
        syms = symbols_from_fo() | {"NIFTY", "INDIAVIX"}
        log("Downloading Yahoo daily prices for %d symbols ..." % len(syms))
        failed = download_yahoo(http, syms, start)
        if failed:
            log("Yahoo failed for:", ", ".join(failed))
    if not a.skip_events:
        download_board_meetings(http, start, end)
    log("Done. Data is in", RAW)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Stopped. Run the same command again to resume.")
