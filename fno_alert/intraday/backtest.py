"""Backtest the intraday selling rules on 1-minute Upstox history.

    python -m fno_alert.intraday.backtest --data data/intraday --out reports/intraday

Data comes from scripts/intraday_backfill.py (branch intraday-data).
Execution model:
* Entry at the close of the entry minute, each leg filled `slippage` worse than the last price.
* Every following minute: stop-loss checked against each leg's worst price in that minute
  (shorts at the minute's high, hedges at its low) and filled as a stop-market order:
  at the stop, or at the minute's open if it opened past the stop, plus extra slippage.
  Profit target and time exit use the minute's close.
* Prices of a contract that did not trade are carried forward up to 5 minutes; after
  that the contract counts as unquoted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from . import engine
from .rules import Rules, hm, load_events

FFILL_LIMIT = 5


class History:
    def __init__(self, root: str, underlying: str):
        d = os.path.join(root, underlying)
        self.underlying = underlying
        idx = pd.read_parquet(os.path.join(d, "index_1m.parquet"))
        idx["ts"] = pd.to_datetime(idx["ts"])
        idx = idx[(idx["ts"].dt.time >= dt.time(9, 15)) & (idx["ts"].dt.time <= dt.time(15, 30))]
        self.idx = idx.set_index("ts").sort_index()
        self.idx["vix"] = self.idx["vix"].ffill()
        self.days = sorted(set(self.idx.index.date))
        self.dayset = set(self.days)
        closes = self.idx["close"].groupby(self.idx.index.date).last()
        rv = np.log(closes).diff().rolling(10).std() * np.sqrt(252)
        self.rv = rv.shift(1).to_dict()  # known before the day starts
        self.files = {dt.date.fromisoformat(os.path.basename(f)[:10]): f
                      for f in glob.glob(os.path.join(d, "opt", "*.parquet"))}
        self.expiries = sorted(self.files)
        self._cache = {}

    def nearest_expiry(self, day: dt.date):
        for e in self.expiries:
            if e >= day:
                return e
        return None

    def dte(self, day: dt.date, expiry: dt.date) -> int:
        return sum(1 for x in self.days if day < x <= expiry)

    def options(self, expiry: dt.date) -> pd.DataFrame:
        if expiry not in self._cache:
            if len(self._cache) > 3:
                self._cache.pop(next(iter(self._cache)))
            o = pd.read_parquet(self.files[expiry])
            if len(o):
                o["ts"] = pd.to_datetime(o["ts"])
                o["day"] = o["ts"].dt.date
            self._cache[expiry] = o
        return self._cache[expiry]

    def day_frames(self, day: dt.date, expiry: dt.date):
        o = self.options(expiry)
        if not len(o):
            return None
        o = o[o["day"] == day]
        if not len(o):
            return None
        minutes = self.idx.loc[str(day)].index
        out = {}
        for col in ("open", "high", "low", "close"):
            p = o.pivot_table(index="ts", columns=["strike", "opt"], values=col, aggfunc="last")
            out[col] = p.reindex(minutes)
        traded = out["close"].notna()
        close = out["close"].ffill(limit=FFILL_LIMIT)
        for col in ("open", "high", "low"):
            out[col] = out[col].where(traded, close)
        out["close"] = close
        lot = float(o["lot"].dropna().iloc[-1]) if o["lot"].notna().any() else math.nan
        return out, lot


def row_dict(frame: pd.DataFrame, i: int) -> dict:
    s = frame.iloc[i]
    return {(float(k), o): float(v) for (k, o), v in s.items() if v == v}


def run(hists: list[History], r: Rules, events: dict, start=None, end=None, log_days: bool = True):
    """Simulate day by day. Returns (trades DataFrame, day log DataFrame)."""
    equity = r.capital
    trades, daylog = [], []
    days = sorted(set().union(*[set(h.days) for h in hists]))
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    t_entry, t_last, t_exit = hm(r.entry_time), hm(r.last_entry_time), hm(r.exit_time)
    for day in days:
        day_pnl, n_today = 0.0, 0
        for h in hists:
            if day not in h.dayset:
                continue
            e = h.nearest_expiry(day)
            if e is None:
                continue
            ix = h.idx.loc[str(day)]
            if len(ix) < 60:
                continue
            prev = [x for x in h.days if x < day]
            prev_close = float(h.idx.loc[str(prev[-1]), "close"].iloc[-1]) if prev else math.nan
            or_ = ix.iloc[: r.or_minutes]
            base = dict(underlying=h.underlying, day_open=float(ix["open"].iloc[0]), prev_close=prev_close,
                        or_high=float(or_["high"].max()), or_low=float(or_["low"].min()),
                        expiry=e, dte=h.dte(day, e), event=events.get(day, ""),
                        rv=float(h.rv.get(day, math.nan)))
            pre = engine.Snapshot(ts=dt.datetime.combine(day, t_entry), spot=float(ix["close"].iloc[r.or_minutes - 1]),
                                  vix=float(ix["vix"].iloc[r.or_minutes - 1]), lot=0, chain={}, **base)
            why = engine.skip_reason(pre, r)
            if why:
                if log_days:
                    daylog.append({"date": day, "underlying": h.underlying, "dte": base["dte"], "action": "skip",
                                   "reason": why})
                continue
            fr = h.day_frames(day, e)
            if fr is None:
                daylog.append({"date": day, "underlying": h.underlying, "dte": base["dte"], "action": "skip",
                               "reason": "no option data"})
                continue
            fr, lot = fr
            times = list(fr["close"].index)
            i = next((j for j, t in enumerate(times) if t.time() >= t_entry), None)
            pos, reason_logged = None, False
            while i is not None and i < len(times):
                t = times[i]
                if pos is None:
                    if (n_today >= r.max_trades_per_day or t.time() > t_last
                            or day_pnl <= -r.daily_loss_pct * equity):
                        break
                    snap = engine.Snapshot(ts=t.to_pydatetime(), spot=float(ix["close"].loc[t]),
                                           vix=float(ix["vix"].loc[t]), lot=lot, chain=row_dict(fr["close"], i), **base)
                    pos, why = engine.plan_entry(snap, r, capital=equity)
                    if pos is None:
                        if not reason_logged:
                            daylog.append({"date": day, "underlying": h.underlying, "dte": base["dte"],
                                           "action": "skip", "reason": why})
                            reason_logged = True
                        if n_today == 0:
                            break
                        i += 1
                        continue
                    n_today += 1
                    i += 1
                    continue
                px = row_dict(fr["close"], i)
                ex = engine.check_exit(pos, t.to_pydatetime(), px, row_dict(fr["open"], i), row_dict(fr["high"], i),
                                       row_dict(fr["low"], i), r, force_time=(i == len(times) - 1))
                if ex:
                    engine.close_position(pos, t.to_pydatetime(), ex[0], ex[1], px, r)
                    pnl = pos.pnl()
                    day_pnl += pnl
                    equity += pnl
                    trades.append(trade_row(pos, pnl, equity))
                    daylog.append({"date": day, "underlying": h.underlying, "dte": base["dte"], "action": "trade",
                                   "reason": f"{pos.structure}, {pos.exit_reason}", "pnl": round(pnl)})
                    pos = None
                    i += r.reentry_gap_min
                    continue
                i += 1
            if pos is not None:  # data ran out: close at last known prices
                px = row_dict(fr["close"], len(times) - 1)
                v = engine.value(pos, px)
                engine.close_position(pos, times[-1].to_pydatetime(), "end of data", v if v == v else pos.credit, px, r)
                pnl = pos.pnl()
                day_pnl += pnl
                equity += pnl
                trades.append(trade_row(pos, pnl, equity))
    return pd.DataFrame(trades), pd.DataFrame(daylog)


def trade_row(p: engine.Position, pnl: float, equity: float) -> dict:
    shorts = [l for l in p.legs if l.side < 0]
    longs = [l for l in p.legs if l.side > 0]
    return {
        "date": p.entry_ts.date(), "underlying": p.underlying, "expiry": p.expiry, "dte": p.notes.get("dte"),
        "structure": p.structure, "entry": p.entry_ts.strftime("%H:%M"), "exit": p.exit_ts.strftime("%H:%M"),
        "spot": round(p.notes.get("spot", math.nan), 1), "vix": round(p.notes.get("vix", math.nan), 1),
        "sold": " / ".join(f"{l.strike:g} {l.opt}" for l in shorts),
        "bought": " / ".join(f"{l.strike:g} {l.opt}" for l in longs),
        "lots": p.lots, "qty": p.qty, "credit": round(p.credit, 2), "exit_value": round(p.exit_value, 2),
        "planned_max_loss": round(p.max_loss), "costs": round(p.entry_cost + p.exit_cost),
        "pnl": round(pnl), "reason": p.exit_reason, "equity": round(equity),
        "loss_vs_plan": round(-pnl / p.max_loss, 2) if pnl < 0 and p.max_loss else 0.0,
    }


def summary(tr: pd.DataFrame, r: Rules, days: int | None = None) -> dict:
    if tr.empty:
        return {"trades": 0}
    eq = r.capital + tr["pnl"].cumsum()
    peak = np.maximum.accumulate(np.r_[r.capital, eq.values])[1:]
    dd = (eq.values - peak) / peak
    wins, losses = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
    first, last = pd.Timestamp(tr["date"].min()), pd.Timestamp(tr["date"].max())
    years = max((last - first).days / 365.25, 0.25)
    worst = tr.nsmallest(1, "pnl").iloc[0]
    return {
        "trades": int(len(tr)), "from": str(first.date()), "to": str(last.date()),
        "net_pnl": round(float(tr["pnl"].sum())), "return_pct": round(float(tr["pnl"].sum()) / r.capital * 100, 1),
        "per_year_pct": round(float(tr["pnl"].sum()) / r.capital * 100 / years, 1),
        "win_rate": round(len(wins) / len(tr) * 100, 1),
        "avg_win": round(float(wins["pnl"].mean())) if len(wins) else 0,
        "avg_loss": round(float(losses["pnl"].mean())) if len(losses) else 0,
        "profit_factor": round(float(wins["pnl"].sum() / -losses["pnl"].sum()), 2) if losses["pnl"].sum() < 0 else None,
        "max_drawdown_pct": round(float(dd.min()) * 100, 1),
        "worst_trade": round(float(worst["pnl"])), "worst_trade_date": str(worst["date"]),
        "worst_vs_plan": round(float(tr["loss_vs_plan"].max()), 2),
        "losses_over_plan": int((tr["loss_vs_plan"] > 1.0).sum()),
        "costs": round(float(tr["costs"].sum())),
        "exit_reasons": tr["reason"].value_counts().to_dict(),
    }


def main():
    from ..cli import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intraday")
    ap.add_argument("--out", default="reports/intraday")
    ap.add_argument("--underlying", default="NIFTY,SENSEX")
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args()
    r = Rules.from_config(load_config(a.config).get("intraday"))
    hists = [History(a.data, u) for u in a.underlying.split(",")]
    tr, log = run(hists, r, load_events())
    os.makedirs(a.out, exist_ok=True)
    tr.to_csv(os.path.join(a.out, "trades.csv"), index=False)
    log.to_csv(os.path.join(a.out, "days.csv"), index=False)
    print(json.dumps(summary(tr, r), indent=1, default=str))


if __name__ == "__main__":
    main()
