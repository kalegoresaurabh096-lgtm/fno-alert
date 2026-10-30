"""Automatic paper trading of the emailed trade ideas, on real Upstox prices.

Same rules as the backtest (strategy.simulate):
* The strikes and expiry are fixed when the alert is sent.
* Entry at the **open of the next session** (the day's open from the Upstox quote).
* Exit at the **close of the 5th session after the alert** (the price at the
  end-of-day scan, which runs after 15:30), or earlier if expiry is that close.
* Costs from strategy.side_cost, the same as in the backtest.

One trade per stock at a time, so the intraday and end-of-day emails for the
same stock do not double count. The book lives in state/paper/trades.csv and
is copied to state/dash/paper.csv for the dashboard.
"""
from __future__ import annotations

import datetime as dt
import math
import os

import numpy as np
import pandas as pd

from . import strategy

COLS = ["alert_date", "alert_time", "symbol", "action", "p_big", "expiry", "lot",
        "ce_strike", "pe_strike", "ce_key", "pe_key", "alert_ce", "alert_pe",
        "status", "entry_date", "entry_ce", "entry_pe", "exit_date", "exit_ce", "exit_pe",
        "sessions", "pnl_per_lot", "ret", "note"]
OPEN, PENDING, CLOSED, SKIPPED = "open", "pending", "closed", "skipped"
TEXT = ["alert_date", "alert_time", "symbol", "action", "expiry", "ce_key", "pe_key", "status",
        "entry_date", "exit_date", "note"]


def _typed(book: pd.DataFrame) -> pd.DataFrame:
    book = book.copy()
    for c in COLS:
        if c not in book:
            book[c] = np.nan
    for c in TEXT:
        book[c] = book[c].astype(object)
    for c in COLS:
        if c not in TEXT:
            book[c] = pd.to_numeric(book[c], errors="coerce").astype(float)
    return book[COLS]


def _path(root: str) -> str:
    return os.path.join(root, "paper", "trades.csv")


def load(root: str) -> pd.DataFrame:
    p = _path(root)
    if not os.path.exists(p):
        return _typed(pd.DataFrame(columns=COLS))
    return _typed(pd.read_csv(p, dtype={c: str for c in TEXT}))


def save(root: str, book: pd.DataFrame):
    os.makedirs(os.path.join(root, "paper"), exist_ok=True)
    book = book[COLS]
    book.to_csv(_path(root), index=False)
    os.makedirs(os.path.join(root, "dash"), exist_ok=True)
    book.to_csv(os.path.join(root, "dash", "paper.csv"), index=False)


def option_key(instr: pd.DataFrame, symbol: str, expiry: dt.date, strike: float, opt: str) -> str | None:
    """Upstox instrument key of one option contract, from the instruments file."""
    if instr is None or instr.empty or "strike_price" not in instr:
        return None
    if "_exp" not in instr:
        instr["_exp"] = pd.to_datetime(instr["expiry"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    m = instr[(instr["segment"] == "NSE_FO") & (instr["underlying_symbol"] == symbol) &
              (instr["instrument_type"] == opt) & (instr["_exp"] == expiry) &
              (np.isclose(instr["strike_price"].astype(float), strike))]
    return str(m["instrument_key"].iloc[0]) if len(m) else None


def add_alerts(book: pd.DataFrame, alerts: list[dict], instr: pd.DataFrame, now: dt.datetime) -> pd.DataFrame:
    """Record a pending paper trade for each new alert with a BUY idea."""
    active = set(book.loc[book["status"].isin([OPEN, PENDING]), "symbol"])
    rows = []
    for a in alerts:
        legs = a.get("legs") or []
        if not legs or a["symbol"] in active or any(not leg[0].startswith("BUY") for leg in legs):
            continue
        rec = {c: np.nan for c in COLS}
        rec.update({"alert_date": now.date().isoformat(), "alert_time": now.strftime("%Y-%m-%d %H:%M"),
                    "symbol": a["symbol"], "action": a["action"], "p_big": round(float(a.get("p_big", np.nan)), 4),
                    "lot": a.get("lot"), "status": PENDING, "sessions": 0, "note": ""})
        for leg, K, exp, ltp in legs:
            o = leg[-2:].lower()
            rec["expiry"] = exp.isoformat()
            rec[f"{o}_strike"] = K
            rec[f"{o}_key"] = option_key(instr, a["symbol"], exp, K, leg[-2:])
            rec[f"alert_{o}"] = ltp
        rows.append(rec)
        active.add(a["symbol"])
    if rows:
        new = _typed(pd.DataFrame(rows, columns=COLS))
        book = pd.concat([book, new], ignore_index=True) if len(book) else new
    return _typed(book)


def resolve_keys(book: pd.DataFrame, instr: pd.DataFrame) -> pd.DataFrame:
    """Fill missing option keys (rows added by hand only need symbol, expiry and strikes)."""
    book = _typed(book)
    for i, r in book[book["status"] == PENDING].iterrows():
        exp = dt.date.fromisoformat(str(r["expiry"])[:10])
        for o in ("ce", "pe"):
            if r[f"{o}_strike"] == r[f"{o}_strike"] and not (isinstance(r[f"{o}_key"], str) and r[f"{o}_key"]):
                book.at[i, f"{o}_key"] = option_key(instr, r["symbol"], exp, float(r[f"{o}_strike"]), o.upper())
    return book


def _legs(r) -> list[str]:
    """Legs of the trade, from the strikes (a leg without a key cannot be priced)."""
    return [o for o in ("ce", "pe") if r[f"{o}_strike"] == r[f"{o}_strike"]]


def _has_keys(r) -> bool:
    return all(isinstance(r[f"{o}_key"], str) and r[f"{o}_key"] and r[f"{o}_key"] != "nan" for o in _legs(r))


def _quote_price(q: dict | None, field: str) -> float:
    if not q:
        return math.nan
    if field == "open":
        v = (q.get("ohlc") or {}).get("open")
    else:
        v = q.get("last_price")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v if v > 0 else math.nan


def update(book: pd.DataFrame, today: dt.date, sessions: list[dt.date], after_close: bool,
           quote_fn, candle_fn, rules: strategy.TradeRules = strategy.TradeRules()) -> pd.DataFrame:
    """Fill entries and exits.

    sessions: trading days known so far, including today if the market is open.
    quote_fn(keys) -> {key: full quote}. candle_fn(key, day) -> (open, close) for a past day.
    """
    book = _typed(book)
    sess = sorted(set(sessions))
    keys = [k for _, r in book[book["status"].isin([OPEN, PENDING])].iterrows() for k in (r["ce_key"], r["pe_key"])
            if isinstance(k, str) and k]
    quotes = quote_fn(sorted(set(keys))) if keys else {}

    def price(key, day, field):
        if day == today:
            v = _quote_price(quotes.get(key), field)
            if v == v:
                return v
        o, c = candle_fn(key, day)
        return o if field == "open" else c

    for i, r in book.iterrows():
        if r["status"] not in (OPEN, PENDING):
            continue
        a_day = dt.date.fromisoformat(str(r["alert_date"])[:10])
        after = [d for d in sess if d > a_day]
        book.at[i, "sessions"] = len(after)
        legs = _legs(r)
        if not _has_keys(r):
            book.at[i, "status"] = SKIPPED
            book.at[i, "note"] = "option contract not found"
            continue
        if r["status"] == PENDING:
            if not after:
                continue
            d_in = after[0]
            px = {o: price(r[f"{o}_key"], d_in, "open") for o in legs}
            if any(not (v == v) for v in px.values()):
                if d_in < today:  # the entry day passed without a usable open price
                    book.at[i, "status"] = SKIPPED
                    book.at[i, "note"] = "no traded open price on entry day"
                continue
            book.at[i, "entry_date"] = d_in.isoformat()
            for o, v in px.items():
                book.at[i, f"entry_{o}"] = v
            book.at[i, "status"] = OPEN
            r = book.loc[i]
        exp = dt.date.fromisoformat(str(r["expiry"])[:10])
        due = len(after) >= rules.hold_days or (exp - today).days <= 1
        if not due:
            continue
        d_exit = after[rules.hold_days - 1] if len(after) >= rules.hold_days else today
        if d_exit == today and not after_close:
            continue
        px = {o: price(r[f"{o}_key"], d_exit, "close") for o in legs}
        if any(not (v == v) for v in px.values()):
            continue
        lot = float(r["lot"]) if r["lot"] == r["lot"] else 1.0
        pnl, paid = 0.0, 0.0
        for o in legs:
            e, x = float(r[f"entry_{o}"]), px[o]
            pnl += (x - strategy.side_cost(x, lot, False, rules)) - (e + strategy.side_cost(e, lot, True, rules))
            paid += e
            book.at[i, f"exit_{o}"] = x
        book.at[i, "exit_date"] = d_exit.isoformat()
        book.at[i, "pnl_per_lot"] = round(pnl * lot, 2)
        book.at[i, "ret"] = round(pnl / paid, 4) if paid else np.nan
        book.at[i, "status"] = CLOSED
    return book


def mark_to_market(book: pd.DataFrame, quotes: dict, rules: strategy.TradeRules = strategy.TradeRules()) -> pd.DataFrame:
    """For open trades, the P&L if closed at the last price (shown on the dashboard)."""
    book = _typed(book)
    for i, r in book[book["status"] == OPEN].iterrows():
        lot = float(r["lot"]) if r["lot"] == r["lot"] else 1.0
        pnl, paid = 0.0, 0.0
        for o in _legs(r):
            x = _quote_price(quotes.get(r[f"{o}_key"]), "last")
            if not (x == x):
                break
            e = float(r[f"entry_{o}"])
            pnl += (x - strategy.side_cost(x, lot, False, rules)) - (e + strategy.side_cost(e, lot, True, rules))
            paid += e
        else:
            book.at[i, "pnl_per_lot"] = round(pnl * lot, 2)
            book.at[i, "ret"] = round(pnl / paid, 4) if paid else np.nan
    return book
