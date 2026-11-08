"""Backtest the desk: both strategies on one account, minute by minute, under the
shared risk engine.

    python -m fno_alert.desk.backtest --data data/intraday --out reports/desk

Data: 1-minute NIFTY index and option history (scripts/intraday_backfill.py,
branch intraday-data). Fills, slippage and costs follow fno_alert/intraday/backtest.py.
Prices of an option that did not trade are carried forward up to 5 minutes; an
open position whose leg has no price uses that leg's last known price.

Variants written to the report:
  strangle alone, condor alone        each with its own account, no desk limits
  desk without limits                 both on one account
  desk                                both on one account under [desk.risk] (the setup that is paper traded)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

from ..intraday import engine
from ..intraday.backtest import FFILL_LIMIT, History  # noqa: F401
from ..intraday.backtest import trade_row
from ..intraday.rules import Rules, load_events
from .risk import RiskEngine, RiskLimits
from .strategies import Condor, CondorParams, Strangle, fit_margin

SPLIT = dt.date(2025, 11, 1)   # the strangle was chosen on data before this date; after it is its holdout


class Frames:
    """Option minute frames of one day, per expiry, with fast access to a few legs."""

    def __init__(self, h: History, day: dt.date):
        self.h, self.day, self._f = h, day, {}

    def get(self, e: dt.date):
        if e not in self._f:
            self._f[e] = self.h.day_frames(self.day, e) if e in self.h.files else None
        return self._f[e]

    def chain(self, e, i) -> dict:
        fr = self.get(e)
        if fr is None:
            return {}
        s = fr[0]["close"].iloc[i]
        return {(float(k), o): float(v) for (k, o), v in s.items() if v == v}

    def legs(self, e, i, keys, col="close") -> dict:
        fr = self.get(e)
        out = {}
        if fr is None:
            return out
        f = fr[0][col]
        for key in keys:
            if key in f.columns:
                v = f[key].iat[i]
                if v == v:
                    out[key] = float(v)
        return out


def mtm(pos: engine.Position, px: dict) -> float:
    """Open profit in rupees if closed at these prices (entry costs counted, exit costs not)."""
    v = engine.value(pos, px)
    return (pos.credit - v) * pos.qty - pos.entry_cost if v == v else -pos.entry_cost


def run(h: History, strategies: list, limits: RiskLimits, events: dict, start=None, end=None,
        use_limits: bool = True):
    capital = limits.capital
    risk = RiskEngine(limits.__class__(**{**limits.__dict__, "monthly_auto_resume": True}), events=events,
                      kill_file="")
    by_name = {s.name: s for s in strategies}
    realized = {s.name: 0.0 for s in strategies}
    open_: dict[str, engine.Position] = {}
    last_px: dict[str, dict] = {}
    traded: set = set()   # condor expiries already traded
    trades, days, refusals = [], [], []
    all_days = [d for d in h.days if (not start or d >= start) and (not end or d <= end)]

    def equity():
        return capital + sum(realized.values()) + sum(mtm(p, last_px.get(n, {})) for n, p in open_.items())

    def book(name, pos, ts, reason, fill, px):
        s = by_name[name]
        s.close(pos, ts, reason, fill, px)
        pnl = pos.pnl()
        realized[name] += pnl
        row = trade_row(pos, pnl, capital + sum(realized.values()))
        row.update(strategy=name, exit_date=ts.date(), entry_date=pos.entry_ts.date())
        trades.append(row)
        del open_[name]
        last_px.pop(name, None)

    for day in all_days:
        ix = h.idx.loc[str(day)]
        if len(ix) < 60:
            continue
        risk.start_day(day, equity())
        fr = Frames(h, day)
        prev = h.days[h.days.index(day) - 1] if h.days.index(day) > 0 else None
        prev_close = float(h.idx.loc[str(prev), "close"].iloc[-1]) if prev else math.nan
        or_ = ix.iloc[:30]
        base = dict(underlying=h.underlying, day_open=float(ix["open"].iloc[0]), prev_close=prev_close,
                    or_high=float(or_["high"].max()), or_low=float(or_["low"].min()), event=events.get(day, ""),
                    rv=float(h.rv.get(day, math.nan)))
        targets = {s.name: s.target_expiry(day, h.expiries) for s in strategies}
        dte = {n: (h.dte(day, e) if e else 99) for n, e in targets.items()}
        looked = False
        times = list(ix.index)
        for i, t in enumerate(times):
            ts = t.to_pydatetime()
            last = i == len(times) - 1
            # ---- marks and exits
            for name, pos in list(open_.items()):
                keys = [(l.strike, l.opt) for l in pos.legs]
                close = {**last_px.get(name, {}), **fr.legs(pos.expiry, i, keys)}
                last_px[name] = close
                if len(close) < len(keys):
                    continue
                s = by_name[name]
                ex = s.check_exit(pos, ts, close, {**close, **fr.legs(pos.expiry, i, keys, "open")},
                                  {**close, **fr.legs(pos.expiry, i, keys, "high")},
                                  {**close, **fr.legs(pos.expiry, i, keys, "low")},
                                  h.dte(day, pos.expiry), last and name == "strangle")
                if ex:
                    book(name, pos, ts, ex[0], ex[1], close)
            # ---- desk loss limits
            if use_limits and open_:
                why = risk.check(ts, equity())
                if why:
                    for name, pos in list(open_.items()):
                        if not risk.closes(why, pos.hedged):
                            continue
                        px = last_px.get(name, {})
                        book(name, pos, ts, "desk loss limit", engine.value(pos, px), px)
            # ---- entries
            for s in strategies:
                if s.name in open_:
                    continue
                e = targets[s.name]
                if e is None:
                    continue
                if s.name == "strangle":
                    if not s.entry_due(ts, e, dte[s.name], looked, False):
                        continue
                    looked = True
                else:
                    if not s.entry_due(ts, e, dte[s.name], traded, False):
                        continue
                ch = fr.chain(e, i)
                if not ch:
                    continue
                snap = engine.Snapshot(ts=ts, spot=float(ix["close"].iat[i]), vix=float(ix["vix"].iat[i]),
                                       expiry=e, dte=dte[s.name], lot=fr.get(e)[1], chain=ch, **base)
                pos, why = s.plan(snap, equity())
                if pos is None:
                    if s.name == "strangle":
                        refusals.append({"date": day, "strategy": s.name, "reason": why})
                    continue
                if use_limits:
                    om = sum(by_name[n].margin(p) for n, p in open_.items())
                    fit_margin(s, pos, limits.max_margin_frac * limits.capital - om)
                    why = risk.allow_entry(s.name, pos.max_loss, s.margin(pos), om, day)
                    if why:
                        refusals.append({"date": day, "strategy": s.name, "reason": why})
                        if s.name == "condor":
                            traded.add(str(e))   # one look per expiry once the desk refuses it
                        continue
                open_[s.name] = pos
                last_px[s.name] = {(l.strike, l.opt): ch[(l.strike, l.opt)] for l in pos.legs}
                if s.name == "condor":
                    traded.add(str(e))
        eq = equity()
        row = {"date": day, "equity": round(eq), "day_pnl": round(eq - risk.s.day_start_equity),
               "halted": risk.s.halted_today, "paused": risk.s.paused}
        for n in by_name:
            row[f"{n}_pnl"] = round(realized[n] + (mtm(open_[n], last_px.get(n, {})) if n in open_ else 0.0))
        days.append(row)
    return pd.DataFrame(trades), pd.DataFrame(days), pd.DataFrame(refusals), risk.s.events


def stats(eq: pd.DataFrame, tr: pd.DataFrame, capital: float, col: str = "equity") -> dict:
    if eq.empty:
        return {}
    e = eq[col].astype(float).values
    peak = np.maximum.accumulate(np.r_[capital, e])[1:]
    dd = e - peak
    daily = np.diff(np.r_[capital, e])
    years = max((pd.Timestamp(eq["date"].iloc[-1]) - pd.Timestamp(eq["date"].iloc[0])).days / 365.25, 0.25)
    net = float(e[-1] - capital)
    sd = daily.std(ddof=1) if len(daily) > 2 else math.nan
    out = {"from": str(eq["date"].iloc[0]), "to": str(eq["date"].iloc[-1]), "net_pnl": round(net),
           "per_year_pct": round(net / capital * 100 / years, 1), "max_drawdown": round(float(dd.min())),
           "max_drawdown_pct": round(float((dd / peak).min()) * 100, 1),
           "worst_day": round(float(daily.min())), "best_day": round(float(daily.max())),
           "sharpe": round(float(daily.mean() / sd * math.sqrt(252)), 2) if sd and sd == sd and sd > 0 else None}
    if len(tr):
        w, l = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
        out.update(trades=int(len(tr)), win_rate=round(len(w) / len(tr) * 100, 1),
                   worst_trade=round(float(tr["pnl"].min())), costs=round(float(tr["costs"].sum())),
                   profit_factor=round(float(w["pnl"].sum() / -l["pnl"].sum()), 2) if l["pnl"].sum() < 0 else None)
    return out


def halves(eq, tr, capital, col="equity"):
    out = {}
    for name, m_eq, m_tr in (("before_nov_2025", eq["date"] < SPLIT, None), ("from_nov_2025", eq["date"] >= SPLIT, None)):
        sub = eq[m_eq].copy()
        if sub.empty:
            continue
        start = capital if name.startswith("before") else float(eq[eq["date"] < SPLIT][col].iloc[-1]) \
            if (eq["date"] < SPLIT).any() else capital
        sub[col] = sub[col] - start + capital
        t = tr[(pd.to_datetime(tr["exit_date"]).dt.date < SPLIT) if name.startswith("before")
               else (pd.to_datetime(tr["exit_date"]).dt.date >= SPLIT)] if len(tr) else tr
        out[name] = stats(sub, t, capital, col)
    return out


def build(cfg: dict):
    r = Rules.from_config(cfg.get("intraday"))
    desk = cfg.get("desk", {})
    limits = RiskLimits.from_config(desk.get("risk"))
    cp = CondorParams.from_config(desk.get("condor"))
    on = desk.get("strategies", ["strangle", "condor"])
    strats = []
    if "strangle" in on:
        strats.append(Strangle(r))
    if "condor" in on:
        strats.append(Condor(cp, r, limits.capital))
    return r, limits, cp, strats


def main():
    from ..cli import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="reports/desk")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--quick", action="store_true", help="only the desk variant")
    a = ap.parse_args()
    cfg = load_config(a.config)
    r, limits, cp, strats = build(cfg)
    h = History(a.data, "NIFTY")
    ev = load_events()
    os.makedirs(a.out, exist_ok=True)
    variants = {"desk": (strats, True)}
    if not a.quick:
        variants.update({"desk_no_limits": (strats, False),
                         "strangle_alone": ([s for s in strats if s.name == "strangle"], False),
                         "condor_alone": ([s for s in strats if s.name == "condor"], False)})
    summary = {}
    for name, (ss, lim) in variants.items():
        tr, eq, ref, rev = run(h, ss, limits, ev, use_limits=lim)
        summary[name] = {"all": stats(eq, tr, limits.capital), **halves(eq, tr, limits.capital)}
        if name == "desk":
            tr.to_csv(os.path.join(a.out, "trades.csv"), index=False)
            eq.to_csv(os.path.join(a.out, "equity.csv"), index=False)
            ref.to_csv(os.path.join(a.out, "refusals.csv"), index=False)
            pd.DataFrame(rev).to_csv(os.path.join(a.out, "risk_events.csv"), index=False)
            per = {}
            for s in ss:
                col = f"{s.name}_pnl"
                e2 = eq[["date", col]].rename(columns={col: "equity"})
                e2["equity"] = e2["equity"] + limits.capital
                per[s.name] = stats(e2, tr[tr["strategy"] == s.name], limits.capital)
            summary["desk_by_strategy"] = per
            if len(eq) > 2:
                d = eq[[f"{s.name}_pnl" for s in ss]].diff().dropna()
                if d.shape[1] == 2:
                    summary["daily_pnl_correlation"] = round(float(d.corr().iloc[0, 1]), 2)
            # worst drawdown of each strategy, used to auto-pause it in paper and live trading
            json.dump({s: abs(v.get("max_drawdown", 0)) for s, v in per.items()},
                      open(os.path.join(a.out, "limits.json"), "w"), indent=1)
        print(name, json.dumps(summary[name]["all"], default=str), flush=True)
    summary["settings"] = {"risk": limits.__dict__, "condor": cp.__dict__,
                           "strangle": {k: getattr(r, k) for k in ("entry_time", "short_delta", "take_profit",
                                                                   "stop_mult", "exit_time", "risk_pct", "hedged")}}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
