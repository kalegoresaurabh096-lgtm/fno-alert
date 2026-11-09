"""Monthly alert: which mutual funds started a new position in a small cap stock.

Every fund house publishes the full portfolio of each scheme as an Excel file
about 10 days after month-end. This module reads those files, keeps one month's
holdings per fund house in state/mfwatch/, and emails the small cap stocks a
fund house holds this month but did not hold the month before.

"Small cap" follows SEBI/AMFI: listed companies ranked 251st onwards by
average market cap, from AMFI's half-yearly list.
"""
from __future__ import annotations

import html
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ISIN_RE = re.compile(r"^IN[A-Z0-9]{9}[0-9]$")


def is_equity_isin(isin: str) -> bool:
    """Indian company shares: INE + issuer code + '01' (equity) + serial + check digit."""
    return bool(ISIN_RE.match(isin)) and isin[2] == "E" and isin[7:9] == "01"


# ---------------------------------------------------------------- reading files

@dataclass
class Holding:
    amc: str
    scheme: str
    isin: str
    name: str
    qty: float
    value_lakh: float
    pct_nav: float


def _norm(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip() if x is not None and x == x else ""


def _num(x) -> float:
    if isinstance(x, (int, float)):
        return float(x) if x == x else 0.0
    s = _norm(x).replace(",", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _find_col(header: list[str], *keys: str, avoid: tuple[str, ...] = ()) -> int | None:
    for i, h in enumerate(header):
        hl = h.lower()
        if any(k in hl for k in keys) and not any(a in hl for a in avoid):
            return i
    return None


def _scheme_name(rows: list[list], header_at: int, sheet: str) -> str:
    """The scheme name is usually in the first few rows above the table."""
    for r in rows[:header_at]:
        cells = [_norm(c) for c in r]
        for k, t in enumerate(cells):
            tl = t.lower()
            if re.match(r"(scheme name|name of the scheme|scheme)\s*[:\-]?\s*$", tl):
                nxt = next((x for x in cells[k + 1:] if x), "")
                if nxt:
                    return nxt
            if len(t) > 6 and ("fund" in tl or "scheme" in tl or "etf" in tl or "plan" in tl) \
                    and "portfolio statement" not in tl and "investment manager" not in tl \
                    and not tl.endswith(("mutual fund", "asset management company limited",
                                         "asset management co. ltd.", "mutual fund)")):
                t = re.sub(r"(?i)^(scheme name|name of the scheme)\s*[:\-]\s*", "", t)
                t = re.sub(r"(?i)\s*\(an open[- ]ended.*$", "", t)
                t = re.sub(r"(?i)\s*portfolio (as on|statement).*$", "", t)
                t = re.sub(r"(?i)^(monthly )?portfolio of\s+", "", t)
                t = t.strip(" :-")
                if t:
                    return t
    return sheet


_DEBUG_SEEN: dict[str, bool] = {}


def parse_sheet(rows: list[list], amc: str, sheet: str) -> list[Holding]:
    """Find the holdings table in one sheet (one scheme) and return its equity rows."""
    header_at = None
    for i, r in enumerate(rows[:60]):
        cells = [_norm(c).lower() for c in r]
        if any(c == "isin" or c.startswith("isin") for c in cells):
            header_at = i
            break
    if header_at is None:
        return []
    header = [_norm(c) for c in rows[header_at]]
    c_isin = _find_col(header, "isin")
    c_name = _find_col(header, "name of", "instrument", "company", "issuer", "security", avoid=("isin", "rating"))
    c_qty = _find_col(header, "quantity", "qty", "no. of shares", "units")
    c_val = (_find_col(header, "market value", "market/fair value", "fair value", "mkt val", "mkt. val",
                       "value (rs", "value(rs")
             or _find_col(header, "value", avoid=("face", "%")))
    c_pct = _find_col(header, "% to nav", "% to net", "%to nav", "% of nav", "% to aum", "% of net", "nav")
    if c_pct == c_val:
        c_pct = None
    # Values are usually in Rs lakh; some files use Rs crore or plain rupees.
    vh = header[c_val].lower() if c_val is not None else ""
    unit = 100.0 if re.search(r"crore|\bcr\b", vh) else (1e-5 if re.search(r"\(\s*(rs\.?|inr|₹)\s*\)$", vh) else 1.0)
    if os.environ.get("MFWATCH_DEBUG") and not _DEBUG_SEEN.get(amc):
        _DEBUG_SEEN[amc] = True
        print(f"  [{amc}] header: {header}  -> isin={c_isin} name={c_name} qty={c_qty} value={c_val} pct={c_pct}")
    scheme = _scheme_name(rows, header_at, sheet)
    out = []
    for r in rows[header_at + 1:]:
        if c_isin is None or c_isin >= len(r):
            continue
        isin = _norm(r[c_isin]).upper()
        if not is_equity_isin(isin):
            continue
        get = lambda c: r[c] if c is not None and c < len(r) else None  # noqa: E731
        out.append(Holding(amc, scheme, isin, _norm(get(c_name)), _num(get(c_qty)), _num(get(c_val)) * unit,
                           _num(get(c_pct))))
    # Some fund houses store 1.23% as 0.0123: if the whole scheme adds up to at most 1, scale it.
    total = sum(h.pct_nav for h in out)
    if out and 0 < total <= 1.01:
        for h in out:
            h.pct_nav *= 100
    return out


def _sheets(data: bytes, filename: str) -> dict[str, list[list]]:
    name = filename.lower()
    if name.endswith(".zip"):
        out = {}
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                if n.lower().endswith((".xls", ".xlsx", ".xlsb")):
                    for k, v in _sheets(z.read(n), n).items():
                        out[f"{Path(n).stem}|{k}"] = v
        return out
    if data[:2] == b"PK" and not name.endswith(".xlsb"):
        engine = "openpyxl"          # .xlsx, even when the file is named .xls
    elif data[:4] == b"\xd0\xcf\x11\xe0":
        engine = "xlrd"              # real old-style .xls
    else:
        engine = "pyxlsb" if name.endswith(".xlsb") else "openpyxl"
    book = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine=engine, dtype=object)
    return {k: v.values.tolist() for k, v in book.items()}


def parse_file(data: bytes, filename: str, amc: str) -> list[Holding]:
    out = []
    for sheet, rows in _sheets(data, filename).items():
        out += parse_sheet(rows, amc, sheet.split("|")[-1])
    return out


def to_frame(holdings: list[Holding]) -> pd.DataFrame:
    cols = ["amc", "scheme", "isin", "name", "qty", "value_lakh", "pct_nav"]
    return pd.DataFrame([h.__dict__ for h in holdings], columns=cols)


# ---------------------------------------------------------------- small cap list

def parse_mcap_list(data: bytes, filename: str) -> pd.DataFrame:
    """AMFI's average-market-cap list -> isin, name, rank, category (Large/Mid/Small Cap)."""
    for rows in _sheets(data, filename).values():
        for i, r in enumerate(rows[:20]):
            header = [_norm(c) for c in r]
            c_isin = _find_col(header, "isin")
            c_cat = _find_col(header, "categorization", "category")
            c_rank = _find_col(header, "sr", "rank")
            c_name = _find_col(header, "company name", "name")
            if c_isin is None:
                continue
            recs = []
            for j, row in enumerate(rows[i + 1:]):
                isin = _norm(row[c_isin]).upper() if c_isin < len(row) else ""
                if not ISIN_RE.match(isin):
                    continue
                rank = int(_num(row[c_rank])) if c_rank is not None else j + 1
                cat = _norm(row[c_cat]) if c_cat is not None else ""
                if not cat:
                    cat = "Large Cap" if rank <= 100 else "Mid Cap" if rank <= 250 else "Small Cap"
                recs.append({"isin": isin, "name": _norm(row[c_name]) if c_name is not None else "",
                             "rank": rank, "category": cat})
            if recs:
                return pd.DataFrame(recs).drop_duplicates("isin")
    raise ValueError("no ISIN table found in the AMFI market cap file")


# ---------------------------------------------------------------- finding new positions

def _name_key(name) -> str:
    n = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    n = re.sub(r"\b(ltd|limited|the|india|co|company|corporation|corp)\b", " ", n)
    return re.sub(r"\s+", "", n)


PASSIVE_RE = re.compile(r"\b(?:ETF|INDEX|NIFTY|SENSEX|BSE|FOF|FUND OF FUNDS?|EXCHANGE TRADED|ARBITRAGE)\b", re.I)
MIN_PCT_NAV = 0.25      # ignore token positions below 0.25% of the scheme...
MIN_VALUE_LAKH = 1000   # ...unless they are worth at least Rs 10 crore


def new_positions(prev: pd.DataFrame, cur: pd.DataFrame, mcap: pd.DataFrame | None) -> pd.DataFrame:
    """Small cap stocks a fund house holds now (in any scheme) but held in none last month.

    One row per fund house and stock, listing the active schemes that bought it. Skipped:
    index funds and ETFs (they buy whatever enters an index), arbitrage funds, schemes
    that were not in last month's file (new schemes, or a sheet we could not read then),
    and token positions. Stocks missing from AMFI's list (listed since it was published,
    or unlisted) are kept with category "Not in AMFI list".
    """
    out_cols = ["amc", "isin", "name", "qty", "value_lakh", "top_pct_nav", "category", "mcap_rank",
                "schemes", "n_amcs"]
    if cur.empty or prev.empty:
        return pd.DataFrame(columns=out_cols)
    held_before = set(zip(prev["amc"], prev["isin"]))
    schemes_before = set(zip(prev["amc"], prev["scheme"].str.upper()))
    c = cur[cur["amc"].isin(set(prev["amc"]))]
    c = c[~c["scheme"].fillna("").str.contains(PASSIVE_RE)]
    c = c[pd.Series([(a, sc.upper()) in schemes_before for a, sc in zip(c["amc"], c["scheme"].fillna(""))],
                    index=c.index, dtype=bool)]
    c = c[pd.Series([(a, i) not in held_before for a, i in zip(c["amc"], c["isin"])], index=c.index, dtype=bool)]
    if mcap is not None and not mcap.empty:
        m = mcap.set_index("isin")
        # A share split or bonus changes the ISIN, so fall back to the company name.
        names = mcap["name"] if "name" in mcap else pd.Series("", index=mcap.index)
        by_name = mcap.assign(k=names.map(_name_key)).query("k != ''").drop_duplicates("k").set_index("k")
        keys = c["name"].map(_name_key)
        c = c.assign(category=c["isin"].map(m["category"]).fillna(keys.map(by_name["category"]))
                     .fillna("Not in AMFI list"),
                     mcap_rank=c["isin"].map(m["rank"]).fillna(keys.map(by_name["rank"])))
        c = c[c["category"].str.lower().str.startswith(("small", "not in"))]
    else:
        c = c.assign(category="", mcap_rank=float("nan"))
    if c.empty:
        return pd.DataFrame(columns=out_cols)
    g = c.groupby(["amc", "isin"], as_index=False).agg(
        name=("name", "first"), qty=("qty", "sum"), value_lakh=("value_lakh", "sum"),
        top_pct_nav=("pct_nav", "max"), category=("category", "first"), mcap_rank=("mcap_rank", "first"),
        schemes=("scheme", lambda s: "; ".join(sorted(set(s)))))
    g = g[(g["top_pct_nav"] >= MIN_PCT_NAV) | (g["value_lakh"] >= MIN_VALUE_LAKH)]
    # How many fund houses started the same stock this month: a crowd is a stronger signal.
    g["n_amcs"] = g.groupby("isin")["amc"].transform("nunique")
    return g.sort_values(["n_amcs", "value_lakh"], ascending=False).reset_index(drop=True)[out_cols]


# ---------------------------------------------------------------- email

def _section(new: pd.DataFrame, describe) -> tuple[list[str], str]:
    td = "padding:6px 8px;border-top:1px solid #eee"
    text, rows = [], []
    for isin, grp in new.groupby("isin", sort=False):
        first = grp.iloc[0]
        tag = describe(first)
        text.append(f"{first['name']} ({tag})")
        for _, r in grp.iterrows():
            text.append(f"  {r['amc']}: Rs {r['value_lakh'] / 100:,.1f} cr, up to {r['top_pct_nav']:.2f}% of a scheme"
                        f" | {r['schemes']}")
            rows.append(f"<tr><td style='{td}'><b>{html.escape(first['name'])}</b>"
                        f"<br><span style='color:#777;font-size:12px'>{html.escape(tag)}</span></td>"
                        f"<td style='{td}'>{html.escape(r['amc'])}</td>"
                        f"<td style='{td};text-align:right'>{r['value_lakh'] / 100:,.1f}</td>"
                        f"<td style='{td};text-align:right'>{r['top_pct_nav']:.2f}%</td>"
                        f"<td style='{td};font-size:12px;color:#444'>{html.escape(r['schemes'])}</td></tr>")
        text.append("")
    th = "padding:6px 8px"
    table = (f"<table style='border-collapse:collapse;font-size:14px;width:100%'>"
             f"<tr style='text-align:left;color:#555'><th style='{th}'>Stock</th><th style='{th}'>Fund house</th>"
             f"<th style='{th};text-align:right'>Rs cr</th><th style='{th};text-align:right'>Biggest % of a scheme</th>"
             f"<th style='{th}'>Schemes</th></tr>" + "".join(rows) + "</table>")
    return text, table


def render_email(new: pd.DataFrame, month: str, amcs_read: list[str], amcs_failed: list[str]) -> tuple[str, str, str]:
    ranked = new[new["mcap_rank"].notna()] if not new.empty else new
    other = new[new["mcap_rank"].isna()] if not new.empty else new
    n = ranked["isin"].nunique() if not ranked.empty else 0
    subject = f"Fund houses' new small cap buys, {month}: {n} stock{'s' if n != 1 else ''}"
    if not other.empty:
        subject += f" (+{other['isin'].nunique()} IPO/unlisted)"
    text = [subject, ""]
    parts = []
    if not ranked.empty:
        t, table = _section(ranked, lambda r: f"small cap, rank {int(r['mcap_rank'])} by market cap")
        text += t
        parts.append(table)
    else:
        text.append("No fund house started a new position in an AMFI-listed small cap.")
        parts.append("<p>No fund house started a new position in an AMFI-listed small cap.</p>")
    if not other.empty:
        text += ["Also new: recent IPOs and unlisted shares (not in AMFI's list yet, so size unknown)", ""]
        t, table = _section(other, lambda r: "recent IPO or unlisted")
        text += t
        parts.append("<h3 style='margin-top:28px'>Also new: recent IPOs and unlisted shares</h3>"
                     "<p style='font-size:13px;color:#555'>Not in AMFI's market cap list yet, so they may not "
                     "be small caps.</p>" + table)
    cover = f"Fund houses read: {', '.join(amcs_read) or 'none'}."
    if amcs_failed:
        cover += f" Could not read: {', '.join(amcs_failed)}."
    note = ("A 'new position' means the fund house held none of the stock, in any scheme, the month before. "
            "Index funds, ETFs, arbitrage funds, newly launched schemes and positions under 0.25% of a scheme "
            "(unless worth Rs 10 cr or more) are left out. Portfolios come out about 10 days after month-end, "
            "so the buying may still be going on. Not investment advice.")
    text += [cover, note]
    body = (f"<html><body style='max-width:820px;margin:auto;font-family:Arial,sans-serif'>"
            f"<h2>New small cap positions by fund houses &middot; {html.escape(month)}</h2>{''.join(parts)}"
            f"<p style='font-size:12px;color:#777;margin-top:24px'>{html.escape(cover)}<br>{html.escape(note)}</p>"
            f"</body></html>")
    return subject, "\n".join(text), body


# ---------------------------------------------------------------- finding and downloading files

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_RE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s_\-.,'%20]*"
                       r"(?:\d{1,2}(?:st|nd|rd|th)?[\s_\-,]*)?((?:19|20)\d{2}|\d{2})(?!\d|st|nd|rd|th)", re.I)
_DAY_MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})[\s_\-.]?(\d{1,2})[\s_\-.]?((?:20)\d{2})(?!\d)")


def month_of(text: str) -> str | None:
    """'monthend-portfolios_31_july_2026.zip' -> '2026-07'. Returns None if no month is named."""
    t = requests_unquote(text)
    m = None
    for m in _MONTH_RE.finditer(t):
        pass  # the last month named is usually the portfolio month (paths often start with an upload month)
    if m:
        y = int(m.group(2))
        y = y + 2000 if y < 100 else y
        return f"{y:04d}-{MONTHS[m.group(1).lower()[:3]]:02d}"
    d = None
    for d in _DAY_MONTH_RE.finditer(t):
        pass
    if d and 1 <= int(d.group(2)) <= 12:
        return f"{int(d.group(3)):04d}-{int(d.group(2)):02d}"
    return None


def requests_unquote(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s)


UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126 Safari/537.36")
FILE_RE = re.compile(r"\.(xlsx?|xlsb|zip)(\?|$)", re.I)


def page_links(url: str, browser: bool) -> list[tuple[str, str]]:
    """All (absolute href, link text) on a page. browser=True renders JavaScript first."""
    from urllib.parse import urljoin
    if browser:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_context(user_agent=UA, locale="en-IN").new_page()
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_timeout(8000)
            links = page.eval_on_selector_all("a[href]", "els => els.map(e => [e.href, e.innerText.trim()])")
            b.close()
        return [(h, t) for h, t in links]
    import requests
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    out = []
    for m in re.finditer(r'<a\b[^>]*?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r.text, re.I | re.S):
        out.append((urljoin(r.url, html.unescape(m.group(1))), re.sub(r"<[^>]+>|\s+", " ", m.group(2)).strip()))
    # Links that sit in scripts or data attributes rather than <a> tags.
    for m in re.finditer(r'["\']([^"\'\s<>]+?\.(?:xlsx?|xlsb|zip)(?:\?[^"\'\s<>]*)?)["\']', r.text, re.I):
        out.append((urljoin(r.url, html.unescape(m.group(1))), ""))
    return out


def monthly_files(src: dict) -> list[tuple[str, list[str]]]:
    """[(YYYY-MM, [urls])] of a fund house's monthly portfolio files, newest first.

    One file per month, or every file for the month when the source has multi=True."""
    inc = re.compile(src.get("include", r"portfolio"), re.I)
    exc = re.compile(src.get("exclude", r"debt|fortnight|half[-_ ]?year|factsheet|riskometer|commission|isin-debt"),
                     re.I)
    best: dict[str, list[str]] = {}
    for href, text in page_links(src["page"], src.get("browser", False)):
        label = f"{href} {text}"
        fname = href.split("?")[0].rsplit("/", 1)[-1]
        if not FILE_RE.search(href) or not inc.search(label) or exc.search(f"{fname} {text}"):
            continue
        m = month_of(text) or month_of(href.rsplit("/", 1)[-1]) or month_of(href)
        if not m:
            continue
        urls = best.setdefault(m, [])
        if href not in urls and (src.get("multi") or not urls):
            urls.append(href)
    return sorted(best.items(), reverse=True)


def read_month(urls: list[str], amc: str) -> pd.DataFrame:
    out = []
    for u in urls:
        out += parse_file(download(u), u.split("?")[0], amc)
    return to_frame(out).drop_duplicates(["scheme", "isin"])


def download(url: str) -> bytes:
    import requests
    r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    return r.content


def latest_mcap_list() -> tuple[str, pd.DataFrame]:
    """AMFI publishes the list every January and July, for the half-year ending 31 Dec / 30 Jun."""
    today = pd.Timestamp.today()
    tries = []
    for y in range(today.year, today.year - 3, -1):
        tries += [f"31Dec{y}", f"30Jun{y}"]
    for label in tries:
        url = f"https://portal.amfiindia.com/spages/AverageMarketCapitalization{label}.xlsx"
        try:
            data = download(url)
        except Exception:  # noqa: BLE001
            continue
        return label, parse_mcap_list(data, url)
    raise RuntimeError("could not download AMFI's market cap list")


# ---------------------------------------------------------------- monthly run

def _load_month(d: Path, month: str) -> pd.DataFrame:
    f = d / f"{month}.csv.gz"
    return pd.read_csv(f, dtype={"isin": str}) if f.exists() else pd.DataFrame()


def _prev_month(month: str) -> str:
    return (pd.Period(month, "M") - 1).strftime("%Y-%m")


def run(state: str = "state", dry_run: bool = False, only: list[str] | None = None,
        send_email: bool = True) -> str:
    from .mfsources import SOURCES
    root = Path(state) / "mfwatch"
    root.mkdir(parents=True, exist_ok=True)
    try:
        label, mcap = latest_mcap_list()
        mcap.to_csv(root / "mcap.csv", index=False)
    except Exception as e:  # noqa: BLE001
        print("AMFI market cap list:", e, "- using the saved copy")
        label, mcap = "saved", pd.read_csv(root / "mcap.csv", dtype={"isin": str})
    print(f"Small cap list: AMFI {label}, {int((mcap['category'].str.lower().str.startswith('small')).sum())} stocks")

    found, read, failed, months = [], [], [], set()
    for src in SOURCES:
        amc = src["amc"]
        if only and amc.lower() not in only:
            continue
        d = root / "holdings" / re.sub(r"\W+", "_", amc.lower())
        try:
            files = monthly_files(src)
            if not files:
                raise RuntimeError("no monthly portfolio files found on the page")
            latest = files[0][0]
            if (d / f"{latest}.csv.gz").exists():
                print(f"{amc}: {latest} already processed")
                continue
            # Make sure we also have the month before, so the first run can compare.
            todo = [(m, u) for m, u in files[:2] if not (d / f"{m}.csv.gz").exists()]
            if src.get("multi"):
                # Per-scheme pages often list only some of last month's files, which would make
                # every stock in the missing schemes look new. Compare only with a month we saved earlier.
                todo = todo[:1]
                files = files[:1]
            for m, urls in reversed(todo):
                h = read_month(urls, amc)
                if h.empty:
                    raise RuntimeError(f"no stock holdings read from {urls[0]}")
                print(f"{amc}: {m} {len(h)} stock rows in {h['scheme'].nunique()} schemes from "
                      f"{len(urls)} file(s), e.g. {urls[0]}")
                if not dry_run:
                    d.mkdir(parents=True, exist_ok=True)
                    h.to_csv(d / f"{m}.csv.gz", index=False)
                if m == latest:
                    prev = _load_month(d, _prev_month(m))
                    if prev.empty and len(files) > 1 and files[1][0] == _prev_month(m):
                        prev = read_month(files[1][1], amc)
                    new = new_positions(prev, h, mcap)
                    print(f"{amc}: {len(new)} new small cap positions in {m}")
                    found.append(new.assign(month=m))
                    months.add(m)
            read.append(amc)
        except Exception as e:  # noqa: BLE001
            print(f"{amc}: FAILED {type(e).__name__}: {e}")
            failed.append(amc)

    new = pd.concat(found, ignore_index=True) if found else pd.DataFrame()
    if not new.empty:
        new["n_amcs"] = new.groupby("isin")["amc"].transform("nunique")
        new = new.sort_values(["n_amcs", "value_lakh"], ascending=False)
        if not dry_run:
            log = root / "alerts_log.csv"
            new.assign(found_on=pd.Timestamp.today().strftime("%Y-%m-%d")).to_csv(
                log, mode="a", header=not log.exists(), index=False)
    month_label = ", ".join(pd.Period(m, "M").strftime("%b %Y") for m in sorted(months)) or "no new month"
    summary = f"{len(read)} fund houses read, {len(failed)} failed, {new['isin'].nunique() if not new.empty else 0} new small cap stocks"
    if not new.empty and send_email and not dry_run:
        from . import emailer
        s, t, h = render_email(new, month_label, read, failed)
        emailer.send(s, t, h)
        summary += " (emailed)"
    elif not new.empty:
        print(render_email(new, month_label, read, failed)[1])
    return summary
