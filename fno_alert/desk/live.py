"""Run the desk on live Upstox prices: every strategy on one paper account under the
shared risk engine.

    python -m fno_alert.desk.live --mode paper              # the daily paper run
    python -m fno_alert.desk.live --mode paper --until 14:30
    python -m fno_alert.desk.live --resume                  # clear a monthly or strategy pause by hand

Live mode uses the same three locks as the intraday runner (intraday/broker.py) and is OFF.
State lives in state/desk/<mode>/:
  open.json     open positions (the condor is held overnight)
  trades.csv    closed trades, one row each, with the strategy name
  log.csv       every decision: entries, skips, refusals by the risk engine, risk actions
  equity.csv    one row per day after the close: account equity and each strategy's P&L
  status.json   the latest marks, risk state and limits, for the dashboard
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import time

import pandas as pd

from ..intraday import engine
from ..intraday.backtest import trade_row
from ..intraday.live import KEYS, VIX, Feed, now_ist
from ..intraday.rules import hm, load_events
from ..upstox import Upstox, UpstoxError
from .backtest import build
from .risk import RiskEngine, RiskState
from .strategies import fit_margin

U = "NIFTY"


class Ledger:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.p = {n: os.path.join(root, n) for n in ("open.json", "trades.csv", "log.csv", "equity.csv",
                                                       "status.json", "risk.json")}

    # ---- open positions
    def load_open(self) -> dict:
        if not os.path.exists(self.p["open.json"]):
            return {}
        out = {}
        for name, d in json.load(open(self.p["open.json"])).items():
            d["legs"] = [engine.Leg(**l) for l in d["legs"]]
            d["entry_ts"] = dt.datetime.fromisoformat(d["entry_ts"])
            d["exit_ts"] = None
            d["expiry"] = dt.date.fromisoformat(d["expiry"])
            out[name] = engine.Position(**d)
        return out

    def save_open(self, pos: dict):
        def enc(p):
            d = dataclasses.asdict(p)
            d.update(expiry=p.expiry.isoformat(), entry_ts=p.entry_ts.isoformat(), exit_ts=None)
            return d
        json.dump({n: enc(p) for n, p in pos.items()}, open(self.p["open.json"], "w"), indent=1, default=str)

    # ---- csv logs
    def _append(self, key, row):
        pd.DataFrame([row]).to_csv(self.p[key], mode="a", header=not os.path.exists(self.p[key]), index=False)

    def add_trade(self, row):
        self._append("trades.csv", row)

    def log(self, ts, strategy, action, detail):
        self._append("log.csv", {"time": f"{ts:%Y-%m-%d %H:%M}", "strategy": strategy, "action": action,
                                 "detail": detail})

    def trades(self) -> pd.DataFrame:
        return pd.read_csv(self.p["trades.csv"]) if os.path.exists(self.p["trades.csv"]) else pd.DataFrame()

    def booked(self) -> dict:
        t = self.trades()
        return t.groupby("strategy")["pnl"].sum().to_dict() if len(t) else {}

    def looked_today(self, day, strategy) -> bool:
        if not os.path.exists(self.p["log.csv"]):
            return False
        lg = pd.read_csv(self.p["log.csv"])
        return bool(((lg["time"].str[:10] == str(day)) & (lg["strategy"] == strategy)
                     & lg["action"].isin(["entry", "skip", "refused"])).any())

    def traded_expiries(self) -> set:
        t = self.trades()
        out = set(t.loc[t["strategy"] == "condor", "expiry"].astype(str)) if len(t) else set()
        if os.path.exists(self.p["log.csv"]):
            lg = pd.read_csv(self.p["log.csv"])
            out |= set(lg.loc[(lg["strategy"] == "condor") & (lg["action"] == "refused"), "detail"]
                       .str.extract(r"expiry (\d{4}-\d\d-\d\d)")[0].dropna())
        return out

    def save_day(self, row: dict):
        old = pd.read_csv(self.p["equity.csv"]) if os.path.exists(self.p["equity.csv"]) else pd.DataFrame()
        if len(old):
            old = old[old["date"].astype(str) != str(row["date"])]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(self.p["equity.csv"], index=False)

    # ---- risk state
    def load_risk(self) -> RiskState:
        return RiskState.from_dict(json.load(open(self.p["risk.json"])) if os.path.exists(self.p["risk.json"]) else None)

    def save_risk(self, s: RiskState):
        json.dump(s.to_dict(), open(self.p["risk.json"], "w"), indent=1)


class Market:
    """NIFTY prices, expiries and sessions from the read-only Analytics Token."""

    def __init__(self, feed: Feed):
        self.f = feed
        self._holidays = None

    def expiries(self, today) -> list:
        c = self.f.contracts(U)
        return sorted(e for e in c["expiry"].unique() if e >= today)

    def lot(self, e) -> float:
        c = self.f.contracts(U)
        return float(c[c["expiry"] == e]["lot_size"].iloc[0])

    def holidays(self) -> set:
        if self._holidays is None:
            try:
                data = self.f.ux.get("/v2/market/holidays") or []
                self._holidays = {dt.date.fromisoformat(h["date"]) for h in data
                                  if "NFO" in (h.get("closed_exchanges") or []) or "NSE" in (h.get("closed_exchanges") or [])}
            except Exception as e:  # weekends are still known; a missed holiday only shifts the condor exit check
                print(f"holiday list unavailable: {e}", flush=True)
                self._holidays = set()
        return self._holidays

    def sessions_to(self, today, e) -> int:
        if e <= today:
            return 0
        days = pd.bdate_range(today + dt.timedelta(days=1), e)
        return int(sum(1 for d in days if d.date() not in self.holidays()))

    def base(self, today, events) -> dict | None:
        m = self.f.minutes_today(KEYS[U])
        if len(m) < 30:
            return None
        o = m.iloc[:30]
        return dict(underlying=U, spot=float(m["close"].iloc[-1]), day_open=float(m["open"].iloc[0]),
                    prev_close=self.f.prev_close(KEYS[U], today), or_high=float(o["high"].max()),
                    or_low=float(o["low"].min()), vix=self.f.ltp([VIX]).get(VIX, math.nan),
                    event=events.get(today, ""))


def fill_like_market(pos, ch):
    """Paper fills at the real bid (sells) or ask (buys) when that is worse than the model price."""
    for l in pos.legs:
        bid, ask = ch[(l.strike, l.opt)][2:4]
        if l.side < 0 and bid > 0:
            l.entry = min(l.entry, bid)
        if l.side > 0 and ask > 0:
            l.entry = max(l.entry, ask)
    pos.credit = sum(-l.side * l.entry for l in pos.legs)
    pos.notes["keys"] = {f"{l.strike:g}{l.opt}": ch[(l.strike, l.opt)][1] for l in pos.legs}


def mtm(pos, px) -> float:
    v = engine.value(pos, px)
    return (pos.credit - v) * pos.qty - pos.entry_cost if v == v else -pos.entry_cost


def run(mode: str, cfg: dict, poll_sec: int = 20, until: str | None = None, resume: bool = False):
    from ..intraday.broker import Broker
    r, limits, cp, strats = build(cfg)
    by_name = {s.name: s for s in strats}
    events = load_events()
    led = Ledger(os.path.join("state", "desk", mode))
    lim_path = os.path.join("reports", "desk", "limits.json")
    strategy_dd = json.load(open(lim_path)) if os.path.exists(lim_path) else {}
    risk = RiskEngine(limits, led.load_risk(), events=events, strategy_dd=strategy_dd)
    if resume:
        risk.resume(now_ist())
        led.save_risk(risk.s)
        print("Desk resumed.")
        return
    broker = Broker(mode, cfg.get("intraday_live") or {})
    mk = Market(Feed(Upstox()))
    open_ = led.load_open()
    marks = {n: {} for n in open_}
    stop_at = hm(until) if until else dt.time(15, 31)
    print(f"desk {mode} run started {now_ist():%Y-%m-%d %H:%M}, open: {list(open_)}", flush=True)

    def keys_of(p):
        return [p.notes.get("keys", {}).get(f"{l.strike:g}{l.opt}") for l in p.legs]

    def equity():
        b = led.booked()
        return limits.capital + sum(b.values()) + sum(mtm(p, marks.get(n, {})) for n, p in open_.items())

    def close(name, p, t, reason, fill, px):
        broker.close(p, {(p.underlying, l.strike, l.opt): k for l, k in zip(p.legs, keys_of(p))})
        by_name[name].close(p, t, reason, fill, px)
        pnl = p.pnl()
        row = trade_row(p, pnl, equity() - mtm(p, px) + pnl)
        row.update(strategy=name, entry_date=p.entry_ts.date(), exit_date=t.date())
        led.add_trade(row)
        del open_[name]
        marks.pop(name, None)
        led.save_open(open_)
        led.log(t, name, "exit", f"{p.structure} {row['sold']} / {row['bought']}: {reason}, P&L Rs {pnl:,.0f}")
        print(f"EXIT {name}: {reason} P&L {pnl:,.0f}", flush=True)

    started_day = False
    while True:
        t = now_ist()
        today = t.date()
        if t.weekday() >= 5 or today in mk.holidays():
            print("Market closed today.")
            break
        if t.time() >= stop_at:
            break
        # ---- marks for open positions
        for name, p in open_.items():
            ks = keys_of(p)
            try:
                q = mk.f.ltp([k for k in ks if k])
            except UpstoxError as e:
                print(f"quote error: {e}", flush=True)
                continue
            px = {(l.strike, l.opt): q[k] for l, k in zip(p.legs, ks) if k in q and q[k] == q[k]}
            marks[name] = {**marks.get(name, {}), **px}
        if t.time() < dt.time(9, 16):
            time.sleep(poll_sec)
            continue
        if not started_day:
            risk.start_day(today, equity())
            started_day = True
        # ---- exits
        for name, p in list(open_.items()):
            px = marks.get(name, {})
            if len(px) < len(p.legs):
                continue
            ex = by_name[name].check_exit(p, t, px, px, px, px, mk.sessions_to(today, p.expiry),
                                          name == "strangle" and t.time() >= hm(r.exit_time))
            if ex:
                close(name, p, t, ex[0], ex[1], px)
        # ---- desk loss limits
        if open_:
            why = risk.check(t, equity())
            if why:
                for name, p in list(open_.items()):
                    if not risk.closes(why, p.hedged):
                        continue
                    px = marks.get(name, {})
                    close(name, p, t, "desk loss limit", engine.value(p, px), px)
                led.log(t, "desk", "risk", why)
        # ---- entries
        for s in strats:
            if s.name in open_:
                continue
            e = s.target_expiry(today, mk.expiries(today))
            if e is None:
                continue
            n = mk.sessions_to(today, e)
            due = (s.entry_due(t, e, n, led.looked_today(today, s.name), False) if s.name == "strangle"
                   else s.entry_due(t, e, n, led.traded_expiries(), False))
            if not due:
                continue
            try:
                base = mk.base(today, events)
                ch = mk.f.chain(U, e)
            except (UpstoxError, KeyError, ValueError) as err:
                print(f"{s.name}: data error {err}", flush=True)
                continue
            if base is None or not ch:
                continue
            snap = engine.Snapshot(ts=t, expiry=e, dte=n, lot=mk.lot(e), chain={k: v[0] for k, v in ch.items()},
                                   **base)
            pos, why = s.plan(snap, equity())
            if pos is None:
                if s.name == "strangle":
                    led.log(t, s.name, "skip", why)
                continue
            fill_like_market(pos, ch)
            om = sum(by_name[x].margin(p) for x, p in open_.items())
            fit_margin(s, pos, limits.max_margin_frac * limits.capital - om)
            why = risk.allow_entry(s.name, pos.max_loss, s.margin(pos), om, today)
            if why:
                led.log(t, s.name, "refused", f"expiry {e}: {why}")
                continue
            keys = {(U, l.strike, l.opt): ch[(l.strike, l.opt)][1] for l in pos.legs}
            if not broker.open(pos, keys):
                led.log(t, s.name, "refused", f"expiry {e}: broker refused the order")
                continue
            open_[s.name] = pos
            marks[s.name] = {(l.strike, l.opt): ch[(l.strike, l.opt)][0] for l in pos.legs}
            led.save_open(open_)
            legs = ", ".join(f"{'SELL' if l.side < 0 else 'BUY'} {l.strike:g} {l.opt} @ {l.entry:.2f}" for l in pos.legs)
            led.log(t, s.name, "entry", f"{pos.structure} expiry {e}, {pos.lots} lot(s): {legs}; "
                                        f"credit Rs {pos.credit * pos.qty:,.0f}, planned max loss Rs {pos.max_loss:,.0f}")
            print(f"ENTRY {s.name}: {legs}", flush=True)
        for name, p in open_.items():
            risk.update_strategy(t, name, led.booked().get(name, 0.0) + mtm(p, marks.get(name, {})))
        write_status(led, risk, open_, marks, equity(), t, limits)
        led.save_risk(risk.s)
        time.sleep(poll_sec)
    t = now_ist()
    if started_day:
        b = led.booked()
        row = {"date": str(t.date()), "equity": round(equity()), "day_pnl": round(equity() - risk.s.day_start_equity),
               "halted": risk.s.halted_today, "paused": risk.s.paused}
        for s in strats:
            row[f"{s.name}_pnl"] = round(b.get(s.name, 0.0) + (mtm(open_[s.name], marks.get(s.name, {}))
                                                               if s.name in open_ else 0.0))
        if t.time() >= dt.time(15, 25):
            led.save_day(row)
        write_status(led, risk, open_, marks, equity(), t, limits)
        led.save_risk(risk.s)
    print(f"desk {mode} run finished {t:%H:%M}", flush=True)


def write_status(led, risk, open_, marks, equity, t, limits):
    st = {"time": f"{t:%Y-%m-%d %H:%M}", "equity": round(equity), "capital": limits.capital,
          "day_pnl": round(equity - risk.s.day_start_equity) if risk.s.day_start_equity else 0,
          "month_pnl": round(equity - risk.s.month_start_equity) if risk.s.month_start_equity else 0,
          "daily_limit": round(limits.daily_loss_pct * limits.capital),
          "monthly_limit": round(limits.monthly_loss_pct * (risk.s.month_start_equity or limits.capital)),
          "kill_switch": risk.killed(), "halted_today": risk.s.halted_today, "paused": risk.s.paused,
          "paused_strategies": risk.s.paused_strategies,
          "open": [{"strategy": n, "structure": p.structure, "expiry": str(p.expiry), "lots": p.lots,
                    "legs": " / ".join(f"{'S' if l.side < 0 else 'B'} {l.strike:g}{l.opt}" for l in p.legs),
                    "credit_rs": round(p.credit * p.qty), "open_pnl": round(mtm(p, marks.get(n, {}))),
                    "planned_max_loss": round(p.max_loss)} for n, p in open_.items()]}
    json.dump(st, open(led.p["status.json"], "w"), indent=1)


def main():
    from ..cli import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--until", default=None, help="HH:MM IST to stop (default 15:31); open positions are kept")
    ap.add_argument("--resume", action="store_true", help="clear a monthly or strategy pause and exit")
    a = ap.parse_args()
    run(a.mode, load_config(a.config), a.poll_sec, a.until, a.resume)


if __name__ == "__main__":
    main()
