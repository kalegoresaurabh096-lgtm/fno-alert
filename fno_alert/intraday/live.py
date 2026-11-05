"""Run the intraday rules on live Upstox prices: paper trading, or (switched OFF)
live trading through the Upstox order API.

    python -m fno_alert.intraday.live --mode paper            # the 30-day paper run
    python -m fno_alert.intraday.live --mode live             # refuses unless every lock is open

The loop runs from start-up until a few minutes after the rules' exit time, checking
prices every `poll_sec` seconds. Decisions come from engine.py, the same code the
backtest uses. State is saved after every change under state/intraday/<mode>/, so a
restarted job picks up an open position.

Live trading locks (all three must be open, see broker.py):
  1. config.toml [intraday_live] enabled = true
  2. environment LIVE_TRADING=I_ACCEPT_REAL_MONEY_RISK
  3. environment UPSTOX_TRADING_TOKEN set (a daily login token; the Analytics Token cannot trade)
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import time
import urllib.parse

import pandas as pd

from .. import emailer
from ..upstox import IST, Upstox, UpstoxError
from . import engine
from .rules import Rules, hm, load_events

KEYS = {"NIFTY": "NSE_INDEX|Nifty 50", "SENSEX": "BSE_INDEX|SENSEX"}
VIX = "NSE_INDEX|India VIX"


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST).replace(tzinfo=None)


class Feed:
    """Live market data for the engine, from the read-only Analytics Token."""

    def __init__(self, ux: Upstox):
        self.ux = ux
        self._contracts = {}

    def contracts(self, underlying: str) -> pd.DataFrame:
        if underlying not in self._contracts:
            data = self.ux.get("/v2/option/contract", {"instrument_key": KEYS[underlying]}) or []
            df = pd.DataFrame(data)
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
            self._contracts[underlying] = df
        return self._contracts[underlying]

    def nearest_expiry(self, underlying: str, today: dt.date) -> tuple[dt.date, float]:
        c = self.contracts(underlying)
        c = c[c["expiry"] >= today]
        e = c["expiry"].min()
        return e, float(c[c["expiry"] == e]["lot_size"].iloc[0])

    def minutes_today(self, key: str) -> pd.DataFrame:
        data = self.ux.get(f"/v3/historical-candle/intraday/{urllib.parse.quote(key, safe='')}/minutes/1") or {}
        rows = data.get("candles", [])
        df = pd.DataFrame([r[:5] for r in rows], columns=["ts", "open", "high", "low", "close"])
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(IST).dt.tz_localize(None)
        return df.sort_values("ts").reset_index(drop=True)

    def prev_close(self, key: str, today: dt.date) -> float:
        path = (f"/v3/historical-candle/{urllib.parse.quote(key, safe='')}/days/1/"
                f"{today - dt.timedelta(days=1)}/{today - dt.timedelta(days=10)}")
        c = (self.ux.get(path) or {}).get("candles", [])
        return float(c[0][4]) if c else math.nan

    def ltp(self, keys: list[str]) -> dict:
        data = self.ux.get("/v2/market-quote/ltp", {"instrument_key": ",".join(keys)}) or {}
        return {v.get("instrument_token"): float(v.get("last_price") or math.nan) for v in data.values()}

    def chain(self, underlying: str, expiry: dt.date):
        """(strike, opt) -> (price, instrument key), price = mid of bid/ask when both are quoted."""
        out = {}
        for item in self.ux.option_chain(KEYS[underlying], expiry):
            k = float(item["strike_price"])
            for side, opt in (("call_options", "CE"), ("put_options", "PE")):
                leg = item.get(side) or {}
                md = leg.get("market_data") or {}
                bid, ask, ltp = (float(md.get(x) or 0) for x in ("bid_price", "ask_price", "ltp"))
                px = (bid + ask) / 2 if bid > 0 and ask > 0 and ask >= bid else ltp
                if px > 0:
                    out[(k, opt)] = (px, leg.get("instrument_key"), bid, ask)
        return out

    def snapshot(self, underlying: str, r: Rules, events: dict):
        t = now_ist()
        today = t.date()
        key = KEYS[underlying]
        m = self.minutes_today(key)
        if len(m) < r.or_minutes:
            return None, {}, "market not open yet or closed today"
        e, lot = self.nearest_expiry(underlying, today)
        dte = int(pd.bdate_range(today + dt.timedelta(days=1), e).size) if e > today else 0
        vix = self.ltp([VIX]).get(VIX, math.nan)
        ch = self.chain(underlying, e)
        o = m.iloc[: r.or_minutes]
        snap = engine.Snapshot(underlying=underlying, ts=t, spot=float(m["close"].iloc[-1]),
                               day_open=float(m["open"].iloc[0]), prev_close=self.prev_close(key, today),
                               or_high=float(o["high"].max()), or_low=float(o["low"].min()), vix=vix,
                               expiry=e, dte=dte, lot=lot, chain={k: v[0] for k, v in ch.items()},
                               event=events.get(today, ""))
        return snap, ch, ""


# ------------------------------------------------------------------ state
class Book:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.open_path = os.path.join(root, "open.json")
        self.trades_path = os.path.join(root, "trades.csv")
        self.days_path = os.path.join(root, "days.csv")

    def open_positions(self) -> dict:
        if not os.path.exists(self.open_path):
            return {}
        raw = json.load(open(self.open_path))
        out = {}
        for u, p in raw.items():
            p["legs"] = [engine.Leg(**l) for l in p["legs"]]
            for f in ("entry_ts", "exit_ts"):
                p[f] = dt.datetime.fromisoformat(p[f]) if p.get(f) else None
            p["expiry"] = dt.date.fromisoformat(p["expiry"])
            out[u] = engine.Position(**p)
        return out

    def save_open(self, pos: dict):
        def enc(p):
            d = dataclasses.asdict(p)
            d["expiry"] = p.expiry.isoformat()
            d["entry_ts"] = p.entry_ts.isoformat()
            d["exit_ts"] = p.exit_ts.isoformat() if p.exit_ts else None
            return d
        json.dump({u: enc(p) for u, p in pos.items()}, open(self.open_path, "w"), indent=1, default=str)

    def _append(self, path, row: dict):
        df = pd.DataFrame([row])
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    def add_trade(self, row: dict):
        self._append(self.trades_path, row)

    def add_day(self, row: dict):
        self._append(self.days_path, row)

    def trades(self) -> pd.DataFrame:
        return pd.read_csv(self.trades_path) if os.path.exists(self.trades_path) else pd.DataFrame()

    def days(self) -> pd.DataFrame:
        return pd.read_csv(self.days_path) if os.path.exists(self.days_path) else pd.DataFrame()

    def equity(self, capital: float) -> float:
        t = self.trades()
        return capital + (float(t["pnl"].sum()) if len(t) else 0.0)

    def today(self, day: dt.date):
        t, d = self.trades(), self.days()
        n = int((t["date"].astype(str) == str(day)).sum()) if len(t) else 0
        pnl = float(t.loc[t["date"].astype(str) == str(day), "pnl"].sum()) if len(t) else 0.0
        # one look per underlying per day: a skipped setup is not retried (rule 5)
        decided = set(d.loc[(d["date"].astype(str) == str(day)) & (d["action"] == "skip"), "underlying"]) \
            if len(d) else set()
        return n, pnl, decided


def notify(subject: str, lines: list[str], dry: bool):
    text = "\n".join(lines + ["", "Paper/automated trading log. Not investment advice; past results do not "
                                  "guarantee future results. Option selling can lose more than planned."])
    print(f"[email] {subject}\n{text}", flush=True)
    if dry or not os.environ.get("SMTP_USER"):
        return
    try:
        body = "<pre style='font:14px/1.4 monospace'>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
        emailer.send(subject, text, body)
    except Exception as e:  # never let email break trading
        print(f"email failed: {e}", flush=True)


def describe(p: engine.Position) -> list[str]:
    legs = ", ".join(f"{'SELL' if l.side < 0 else 'BUY'} {p.underlying} {l.strike:g} {l.opt} @ {l.entry:.2f}"
                     for l in p.legs)
    return [f"{p.structure.upper()} {p.underlying} expiry {p.expiry} ({p.lots} lot(s), {p.qty:g} qty)",
            f"Legs: {legs}",
            f"Credit {p.credit:.2f} per unit = Rs {p.credit * p.qty:,.0f}; planned max loss Rs {p.max_loss:,.0f}",
            f"Profit target and stop: exit at {p.credit * 0.5:.2f} or loss of 1x credit; everything closed by exit time"]


def run(mode: str, cfg: dict, poll_sec: int = 20, until: str | None = None, dry: bool = False,
        underlyings: tuple = ("NIFTY", "SENSEX")):
    from .broker import Broker  # noqa: local import keeps paper mode free of order code paths
    r = Rules.from_config(cfg.get("intraday"))
    events = load_events()
    book = Book(os.path.join("state", "intraday", mode))
    broker = Broker(mode, cfg.get("intraday_live") or {})
    feed = Feed(Upstox())
    stop_at = hm(until) if until else (dt.datetime.combine(dt.date.today(), hm(r.exit_time))
                                       + dt.timedelta(minutes=5)).time()
    t_entry, t_last = hm(r.entry_time), hm(r.last_entry_time)
    positions = book.open_positions()
    keys = {}  # (underlying, strike, opt) -> instrument key
    for u, p in positions.items():
        for l in p.legs:
            keys[(u, l.strike, l.opt)] = p.notes.get("keys", {}).get(f"{l.strike:g}{l.opt}")
    last_entry_try = {}
    print(f"{mode} run started {now_ist():%Y-%m-%d %H:%M}, open positions: {list(positions)}", flush=True)
    while True:
        t = now_ist()
        if t.weekday() >= 5:
            print("Weekend, nothing to do.")
            break
        if t.time() >= stop_at and not positions:
            break
        n_today, pnl_today, decided = book.today(t.date())
        equity = book.equity(r.capital)
        # ---- manage open positions
        for u, p in list(positions.items()):
            leg_keys = [keys.get((u, l.strike, l.opt)) for l in p.legs]
            try:
                q = feed.ltp([k for k in leg_keys if k])
            except UpstoxError as e:
                print(f"quote error: {e}", flush=True)
                continue
            px = {(l.strike, l.opt): q[k] for l, k in zip(p.legs, leg_keys) if k in q and q[k] == q[k]}
            if len(px) < len(p.legs):
                print(f"{u}: missing leg prices {px}", flush=True)
                continue
            ex = engine.check_exit(p, t, px, px, px, px, r, force_time=t.time() >= hm(r.exit_time))
            if ex:
                broker.close(p, keys)
                engine.close_position(p, t, ex[0], ex[1], px, r)
                pnl = p.pnl()
                row = trade_row(p, pnl, equity + pnl)
                book.add_trade(row)
                del positions[u]
                book.save_open(positions)
                notify(f"[{mode}] EXIT {u} {p.structure}: {ex[0]}, P&L Rs {pnl:,.0f}",
                       describe(p) + [f"Exit {t:%H:%M} at {p.exit_value:.2f}: {ex[0]}",
                                      f"P&L after costs Rs {pnl:,.0f}. Paper equity Rs {equity + pnl:,.0f}"], dry)
        # ---- look for new entries
        n_today, pnl_today, decided = book.today(t.date())
        n_today += len(positions)
        equity = book.equity(r.capital)
        if t_entry <= t.time() <= t_last:
            for u in underlyings:
                if u in positions or u in decided:
                    continue
                if n_today >= r.max_trades_per_day or pnl_today <= -r.daily_loss_pct * equity:
                    continue
                if last_entry_try.get(u) and (t - last_entry_try[u]).total_seconds() < 60:
                    continue
                last_entry_try[u] = t
                try:
                    snap, ch, why = feed.snapshot(u, r, events)
                except (UpstoxError, KeyError, ValueError) as e:
                    print(f"{u}: data error {e}", flush=True)
                    continue
                if snap is None:
                    print(f"{u}: {why}", flush=True)
                    continue
                pos, why = engine.plan_entry(snap, r, capital=equity)
                day_row = {"date": t.date(), "underlying": u, "dte": snap.dte, "time": f"{t:%H:%M}",
                           "spot": round(snap.spot, 1), "vix": round(snap.vix, 2)}
                if pos is None:
                    book.add_day({**day_row, "action": "skip", "reason": why})
                    if why not in ("expiry too far",):
                        notify(f"[{mode}] {u}: no trade today ({why})",
                               [f"{u} spot {snap.spot:,.1f}, VIX {snap.vix:.1f}, expiry {snap.expiry} "
                                f"({snap.dte} sessions away). Reason: {why}"], dry)
                    continue
                pos.notes["keys"] = {f"{l.strike:g}{l.opt}": ch[(l.strike, l.opt)][1] for l in pos.legs}
                for l in pos.legs:
                    keys[(u, l.strike, l.opt)] = ch[(l.strike, l.opt)][1]
                    bid, ask = ch[(l.strike, l.opt)][2:4]
                    # paper fills at the real bid (sells) / ask (buys) when that is worse than the model
                    if l.side < 0 and bid > 0:
                        l.entry = min(l.entry, bid)
                    if l.side > 0 and ask > 0:
                        l.entry = max(l.entry, ask)
                pos.credit = sum(-l.side * l.entry for l in pos.legs)
                if not broker.open(pos, keys):
                    book.add_day({**day_row, "action": "skip", "reason": "broker refused the order"})
                    continue
                positions[u] = pos
                book.save_open(positions)
                book.add_day({**day_row, "action": "trade", "reason": pos.structure})
                notify(f"[{mode}] ENTRY {u} {pos.structure}, credit Rs {pos.credit * pos.qty:,.0f}",
                       describe(pos) + [f"Spot {snap.spot:,.1f}, VIX {snap.vix:.1f}, "
                                        f"{snap.dte} session(s) to expiry"], dry)
                n_today += 1
        if t.time() >= stop_at and not positions:
            break
        time.sleep(poll_sec)
    print(f"{mode} run finished {now_ist():%H:%M}", flush=True)


def trade_row(p: engine.Position, pnl: float, equity: float) -> dict:
    from .backtest import trade_row as tr
    return tr(p, pnl, equity)


def main():
    from ..cli import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--until", default=None, help="HH:MM IST to stop (default: exit time + 5 min)")
    ap.add_argument("--dry-run", action="store_true", help="no emails")
    a = ap.parse_args()
    run(a.mode, load_config(a.config), a.poll_sec, a.until, a.dry_run)


if __name__ == "__main__":
    main()
