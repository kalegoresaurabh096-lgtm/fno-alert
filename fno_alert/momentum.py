"""Monthly 12-1 momentum picks among the F&O stocks, with a paper portfolio.

The rule (research/momentum_cash.py, the only stock strategy that held up on
unseen data): on the first trading day of each month, rank the F&O stocks by
their return from 12 months ago to 1 month ago, and hold the top N equally.
Stocks that drop out are sold, new ones bought; stocks still in the top N
are kept as they are. Orders are emailed after the close and paper-filled at
the next session's open, with delivery costs (STT 0.1% + fees + slippage).
"""
from __future__ import annotations

import datetime as dt
import html
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import prices


@dataclass(frozen=True)
class Rules:
    top_n: int = 30
    lookback: int = 252          # trading days
    skip: int = 21               # leave out the latest month (short-term reversal)
    min_turnover_cr: float = 10.0
    capital: float = 500_000.0
    cost: float = 0.0015         # per side: STT 0.1% + fees + slippage

    @classmethod
    def from_cfg(cls, cfg: dict | None):
        c = (cfg or {}).get("momentum", {}) or {}
        kw = {k: type(getattr(cls(), k))(c[k]) for k in cls().__dict__ if k in c}
        return cls(**kw)


HOLD_COLS = ["symbol", "qty", "entry_date", "entry_price", "cost_paid"]
ORDER_COLS = ["date", "symbol", "side", "qty", "status", "fill_date", "fill_price"]


def scores(closes: dict[str, pd.Series], turnover_cr: dict[str, float], r: Rules = Rules()) -> pd.DataFrame:
    """12-1 momentum for each stock with enough history and turnover, best first."""
    rows = []
    for sym, c in closes.items():
        c = c.dropna()
        if len(c) < r.lookback + 1 or turnover_cr.get(sym, 0) < r.min_turnover_cr:
            continue
        a, b = float(c.iloc[-1 - r.lookback]), float(c.iloc[-1 - r.skip])
        if a > 0 and b > 0:
            rows.append({"symbol": sym, "mom": b / a - 1, "close": float(c.iloc[-1])})
    df = pd.DataFrame(rows, columns=["symbol", "mom", "close"])
    df = df.sort_values("mom", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


class Book:
    """Paper portfolio in state/momentum/."""

    def __init__(self, root: str, r: Rules = Rules()):
        self.dir = os.path.join(root, "momentum")
        self.dash = os.path.join(root, "dash")
        self.r = r
        os.makedirs(self.dir, exist_ok=True)
        self.hold = self._read("holdings.csv", HOLD_COLS)
        self.orders = self._read("orders.csv", ORDER_COLS)
        self.equity = self._read("equity.csv", ["date", "equity", "cash", "invested", "nifty"])
        st = self._read("state.csv", ["key", "value"])
        self.state = dict(zip(st["key"], st["value"]))
        self.cash = float(self.state.get("cash", r.capital))

    def _read(self, name, cols):
        p = os.path.join(self.dir, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=cols)

    def save(self):
        self.state["cash"] = round(self.cash, 2)
        self.hold[HOLD_COLS].to_csv(os.path.join(self.dir, "holdings.csv"), index=False)
        self.orders[ORDER_COLS].to_csv(os.path.join(self.dir, "orders.csv"), index=False)
        self.equity.to_csv(os.path.join(self.dir, "equity.csv"), index=False)
        pd.DataFrame({"key": list(self.state), "value": list(self.state.values())}).to_csv(
            os.path.join(self.dir, "state.csv"), index=False)

    def rebalanced_month(self) -> str:
        return str(self.state.get("last_rebalance", ""))

    def plan(self, ranked: pd.DataFrame, today: dt.date) -> pd.DataFrame:
        """Orders for today's rebalance: sell drop-outs, buy new names (qty set at fill time)."""
        top = list(ranked["symbol"].head(self.r.top_n))
        held = set(self.hold["symbol"])
        pending = self.orders[self.orders["status"] == "pending"]
        if len(pending):  # an unfilled plan is replaced, not stacked
            self.orders = self.orders[self.orders["status"] != "pending"]
        new = [{"date": today.isoformat(), "symbol": s, "side": "SELL", "qty": float(self.hold.loc[self.hold["symbol"] == s, "qty"].sum()),
                "status": "pending", "fill_date": "", "fill_price": np.nan} for s in sorted(held - set(top))]
        new += [{"date": today.isoformat(), "symbol": s, "side": "BUY", "qty": np.nan, "status": "pending",
                 "fill_date": "", "fill_price": np.nan} for s in top if s not in held]
        o = pd.DataFrame(new, columns=ORDER_COLS)
        self.orders = pd.concat([self.orders, o], ignore_index=True) if len(self.orders) else o
        self.state["last_rebalance"] = today.strftime("%Y-%m")
        return o

    def fill(self, today: dt.date, opens: dict[str, float]):
        """Fill pending orders from an earlier day at today's open: sells first, then equal-weight buys."""
        p = self.orders[(self.orders["status"] == "pending") & (self.orders["date"] < today.isoformat())]
        if p.empty:
            return
        for i, o in p[p["side"] == "SELL"].iterrows():
            px = opens.get(o["symbol"])
            h = self.hold[self.hold["symbol"] == o["symbol"]]
            if not px or h.empty:
                if h.empty:
                    self.orders.loc[i, "status"] = "cancelled"
                continue
            qty = float(h["qty"].sum())
            self.cash += qty * px * (1 - self.r.cost)
            self.hold = self.hold[self.hold["symbol"] != o["symbol"]]
            self.orders.loc[i, ["status", "fill_date", "fill_price", "qty"]] = ["filled", today.isoformat(), px, qty]
        buys = p[p["side"] == "BUY"]
        n_slots = max(self.r.top_n - len(self.hold), 0)
        budget = self.cash / max(min(n_slots, len(buys)), 1)
        for i, o in buys.iterrows():
            px = opens.get(o["symbol"])
            if not px:
                continue
            qty = math.floor(budget / (px * (1 + self.r.cost)))
            if qty < 1:
                self.orders.loc[i, "status"] = "skipped"
                continue
            paid = qty * px * (1 + self.r.cost)
            self.cash -= paid
            row = pd.DataFrame([{"symbol": o["symbol"], "qty": qty, "entry_date": today.isoformat(),
                                 "entry_price": px, "cost_paid": round(paid, 2)}])
            self.hold = pd.concat([self.hold, row], ignore_index=True) if len(self.hold) else row
            self.orders.loc[i, ["status", "fill_date", "fill_price", "qty"]] = ["filled", today.isoformat(), px, qty]

    def mark(self, today: dt.date, closes: dict[str, float], nifty: float | None):
        h = self.hold.copy()
        h["last"] = h["symbol"].map(closes).astype(float)
        h["last"] = h["last"].fillna(h["entry_price"].astype(float))
        h["value"] = h["qty"].astype(float) * h["last"]
        h["pnl"] = h["value"] - h["cost_paid"].astype(float)
        h["ret"] = h["pnl"] / h["cost_paid"].astype(float)
        invested = float(h["value"].sum()) if len(h) else 0.0
        eq = self.cash + invested
        row = {"date": today.isoformat(), "equity": round(eq, 2), "cash": round(self.cash, 2),
               "invested": round(invested, 2), "nifty": nifty}
        e = self.equity[self.equity["date"] != today.isoformat()] if len(self.equity) else self.equity
        self.equity = pd.concat([e, pd.DataFrame([row])], ignore_index=True) if len(e) else pd.DataFrame([row])
        os.makedirs(self.dash, exist_ok=True)
        h.round(4).to_csv(os.path.join(self.dash, "momentum.csv"), index=False)
        self.equity.to_csv(os.path.join(self.dash, "momentum_equity.csv"), index=False)
        return eq


def email(ranked: pd.DataFrame, orders: pd.DataFrame, book: Book, today: dt.date) -> tuple[str, str, str]:
    r = book.r
    top = ranked.head(r.top_n)
    buys = list(orders.loc[orders["side"] == "BUY", "symbol"])
    sells = list(orders.loc[orders["side"] == "SELL", "symbol"])
    month = today.strftime("%B %Y")
    subject = f"Momentum picks {month}: {len(buys)} buy, {len(sells)} sell"
    per = r.capital / r.top_n
    lines = [f"Monthly momentum rebalance ({month}). Place these at tomorrow's open.", "",
             "SELL: " + (", ".join(sells) if sells else "nothing"),
             "BUY:  " + (", ".join(buys) if buys else "nothing"), "",
             f"Top {r.top_n} (12-1 month return), about Rs {per:,.0f} each on Rs {r.capital:,.0f}:"]
    rows = []
    for t in top.itertuples():
        act = "BUY" if t.symbol in buys else "HOLD"
        qty = math.floor(per / t.close) if t.close > 0 else 0
        lines.append(f"{t.rank:>2}. {t.symbol:<12} {t.mom*100:+6.1f}%  {act:<4}  ~{qty} shares @ Rs {t.close:,.2f}")
        color = "#0a7f3f" if act == "BUY" else "#555"
        rows.append(f"<tr><td>{t.rank}</td><td><b>{html.escape(t.symbol)}</b></td><td style='text-align:right'>{t.mom*100:+.1f}%</td>"
                    f"<td style='color:{color}'><b>{act}</b></td><td style='text-align:right'>~{qty}</td>"
                    f"<td style='text-align:right'>{t.close:,.2f}</td></tr>")
    note = ("Backtest 2024 to Oct 2026 (today's F&O list, delivery costs): +48% vs NIFTY +4%, worst fall -24%. "
            "It lagged NIFTY in 2025. Past results can fail; this is a statistical rule, not investment advice.")
    lines += ["", note]
    body = (f"<div style='font-family:Arial,sans-serif;max-width:640px'>"
            f"<h2 style='margin:0 0 8px'>Momentum picks, {month}</h2>"
            f"<p>Place these at tomorrow's open (delivery, not F&amp;O).</p>"
            f"<p><b style='color:#b3261e'>SELL:</b> {html.escape(', '.join(sells) or 'nothing')}<br>"
            f"<b style='color:#0a7f3f'>BUY:</b> {html.escape(', '.join(buys) or 'nothing')}</p>"
            f"<table cellpadding='4' style='border-collapse:collapse;font-size:14px'>"
            f"<tr><th>#</th><th>Stock</th><th>12-1m</th><th>Action</th><th>Qty</th><th>Close</th></tr>{''.join(rows)}</table>"
            f"<p style='color:#666;font-size:12px'>About Rs {per:,.0f} per stock on Rs {r.capital:,.0f}. {html.escape(note)}</p></div>")
    return subject, "\n".join(lines), body


def run(st_root: str, uni_symbols: list[str], load_px, today_bars: dict[str, dict], lots_hist: pd.DataFrame,
        today: dt.date, mode: str, nifty: float | None, cfg: dict | None, send_email) -> str:
    """Called by the scan after alerts. Fills yesterday's orders, marks the book,
    and on the first end-of-day run of a month ranks stocks and emails the picks."""
    r = Rules.from_cfg(cfg)
    book = Book(st_root, r)
    opens = {s: float(b["open"]) for s, b in today_bars.items() if b.get("open")}
    closes_now = {s: float(b["close"]) for s, b in today_bars.items() if b.get("close")}
    book.fill(today, opens)
    msg = ""
    if mode == "eod" and book.rebalanced_month() != today.strftime("%Y-%m"):
        closes, turnover = {}, {}
        for s in uni_symbols:
            px = load_px(s)
            if px is None or px.empty:
                continue
            b = today_bars.get(s)
            if b and b.get("close"):
                px = px[px.index < pd.Timestamp(today)]
                px = pd.concat([px, pd.DataFrame([b], index=[pd.Timestamp(today)])])
            lots = lots_hist[lots_hist["symbol"] == s].set_index("date")["lot"] if len(lots_hist) else None
            pxa = prices.adjust_splits(px.astype(float), lots)
            closes[s] = pxa["close"]
            turnover[s] = float((pxa["close"] * pxa["volume"]).tail(20).mean() / 1e7)
        ranked = scores(closes, turnover, r)
        if len(ranked) >= r.top_n:
            orders = book.plan(ranked, today)
            subject, text, body = email(ranked, orders, book, today)
            send_email(subject, text, body)
            ranked.to_csv(os.path.join(book.dir, f"ranking_{today:%Y-%m}.csv"), index=False)
            msg = f"momentum rebalance: {int((orders['side'] == 'BUY').sum())} buy, {int((orders['side'] == 'SELL').sum())} sell"
        else:
            msg = f"momentum: only {len(ranked)} stocks have 13 months of history, no rebalance"
    eq = book.mark(today, closes_now, nifty)
    book.save()
    return (msg + "; " if msg else "") + f"momentum paper equity Rs {eq:,.0f}"
