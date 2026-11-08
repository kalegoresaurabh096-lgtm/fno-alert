"""The desk's strategies. Each one decides entries and exits from engine.Snapshot
prices, so the backtest and the paper run share every decision.

  strangle  Expiry-day NIFTY short strangle: 11:30 entry on expiry day, 0.20 delta,
            no hedge, stop when the loss equals the premium, take profit at 50%,
            closed by 15:15 (config [intraday], the setup in its 30-day paper run).
  condor    Weekly NIFTY iron condor: about 7 calendar days before an expiry (in
            practice at 15:20 on the previous expiry day), sell the 0.10-delta call
            and put, buy wings 500 points further out, close at 15:20 on the last
            session before expiry. Optional loss stop (stop_loss_pct of capital).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from ..intraday import engine
from ..intraday.rules import Rules, hm


@dataclass(frozen=True)
class CondorParams:
    dte_days: int = 7               # enter on the first session at most this many calendar days before expiry
    min_sessions: int = 2           # never enter with fewer sessions than this left
    short_delta: float = 0.10
    wing: float = 500
    lots: int = 1
    entry_time: str = "15:20"
    last_entry_time: str = "15:28"
    exit_time: str = "15:20"        # on the last session before expiry
    stop_loss_pct: float = 0.0      # close when the loss reaches this share of capital (0 = hold to the hedge)
    slippage: float = 0.01          # per leg per side, share of price (far options are thin)
    min_slippage: float = 0.25

    @classmethod
    def from_config(cls, cfg: dict | None, **over) -> "CondorParams":
        names = set(cls.__dataclass_fields__)
        vals = {k: v for k, v in (cfg or {}).items() if k in names}
        vals.update(over)
        return cls(**vals)


class Strangle:
    name = "strangle"
    label = "Expiry-day strangle"

    def __init__(self, rules: Rules):
        self.r = rules

    def target_expiry(self, day: dt.date, expiries: list) -> dt.date | None:
        return next((e for e in sorted(expiries) if e >= day), None)

    def entry_due(self, ts: dt.datetime, expiry: dt.date, dte: int, looked: bool, open_now: bool) -> bool:
        """One look per day, at the first check inside the entry window."""
        return (not looked and not open_now and dte <= self.r.max_dte
                and hm(self.r.entry_time) <= ts.time() <= hm(self.r.last_entry_time))

    def plan(self, snap: engine.Snapshot, capital: float):
        return engine.plan_entry(snap, self.r, capital=capital)

    def margin(self, pos: engine.Position) -> float:
        n_short = sum(1 for l in pos.legs if l.side < 0)
        spot = pos.notes.get("spot", 0.0)
        if pos.hedged:
            return self.r.wing.get(pos.underlying, 100) * pos.qty * 1.1
        return self.r.naked_margin_frac * spot * pos.qty * (1 if n_short == 1 else 1.15)

    def check_exit(self, pos, ts, close, open_, hi, lo, dte: int, last_minute: bool):
        return engine.check_exit(pos, ts, close, open_, hi, lo, self.r, force_time=last_minute)

    def close(self, pos, ts, reason, fill, px):
        return engine.close_position(pos, ts, reason, fill, px, self.r)


class Condor:
    name = "condor"
    label = "Weekly condor"

    def __init__(self, p: CondorParams, cost_rules: Rules, capital: float):
        self.p = p
        # same brokerage, taxes and fees as the strangle, wider slippage for far options
        self.r = cost_rules.with_(slippage=p.slippage, min_slippage=p.min_slippage, hedged=True,
                                  stop_mult=0.0, take_profit=0.0, wing={"NIFTY": p.wing, "SENSEX": p.wing})
        self.capital = capital

    def target_expiry(self, day: dt.date, expiries: list) -> dt.date | None:
        """The first expiry after today (on an expiry day that is next week's)."""
        return next((e for e in sorted(expiries) if e > day), None)

    def entry_due(self, ts: dt.datetime, expiry: dt.date, dte: int, traded: set, open_now: bool) -> bool:
        return (not open_now and str(expiry) not in traded and dte >= self.p.min_sessions
                and ts.date() >= expiry - dt.timedelta(days=self.p.dte_days)
                and hm(self.p.entry_time) <= ts.time() <= hm(self.p.last_entry_time))

    def plan(self, snap: engine.Snapshot, capital: float):
        deltas = engine.otm_deltas(snap)
        legs = []
        for opt in ("CE", "PE"):
            cands = [(k, d) for (k, o), d in deltas.items() if o == opt]
            if not cands:
                return None, "no option prices"
            k, _ = min(cands, key=lambda kd: abs(kd[1] - self.p.short_delta))
            hk = k + self.p.wing if opt == "CE" else k - self.p.wing
            if (hk, opt) not in snap.chain:
                return None, "wing strike not quoted"
            sp, hp = snap.chain[(k, opt)], snap.chain[(hk, opt)]
            legs += [engine.Leg(k, opt, -1, sp - engine.slip(sp, self.r)),
                     engine.Leg(hk, opt, +1, hp + engine.slip(hp, self.r))]
        credit = sum(-l.side * l.entry for l in legs)
        if credit <= 0:
            return None, "no premium after costs"
        qty = self.p.lots * snap.lot
        cost = sum(engine.side_cost(l.entry, qty, l.side > 0, self.r, snap.underlying) for l in legs)
        hard = (self.p.wing - credit) * qty + 2 * cost
        planned = min(hard, self.p.stop_loss_pct * capital) if self.p.stop_loss_pct > 0 else hard
        pos = engine.Position(snap.underlying, snap.expiry, "iron condor", legs, qty, self.p.lots, credit,
                              planned, snap.ts, cost, True,
                              notes={"spot": snap.spot, "vix": snap.vix, "dte": snap.dte, "hard_max_loss": hard,
                                     "deltas": {f"{l.strike:g}{l.opt}": round(deltas.get((l.strike, l.opt), math.nan), 3)
                                                for l in legs}})
        return pos, ""

    def margin(self, pos: engine.Position) -> float:
        return self.p.wing * pos.qty * 1.1

    def check_exit(self, pos, ts, close, open_, hi, lo, dte: int, last_minute: bool):
        v = engine.value(pos, close)
        if v != v:
            return None
        if self.p.stop_loss_pct > 0 and (v - pos.credit) * pos.qty + pos.entry_cost >= self.p.stop_loss_pct * self.capital:
            return "stop-loss", min(v, self.p.wing)
        if dte <= 0 or (dte == 1 and ts.time() >= hm(self.p.exit_time)):
            return "time exit", v
        return None

    def close(self, pos, ts, reason, fill, px):
        return engine.close_position(pos, ts, reason, fill, px, self.r)


def fit_margin(strategy, pos: engine.Position, room: float) -> bool:
    """Cut a position's lots so its margin fits in `room`. False if not even one lot fits."""
    need = strategy.margin(pos)
    if need <= room:
        return True
    per_lot = need / pos.lots
    lots = int(room // per_lot)
    if lots < 1:
        return False
    k = lots / pos.lots
    pos.lots, pos.qty = lots, pos.qty * k
    pos.max_loss *= k
    pos.entry_cost = pos.entry_cost * k  # close enough: brokerage per order does not shrink, fees do
    return True
