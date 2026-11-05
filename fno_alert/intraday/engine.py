"""Decision logic shared by the backtest, the paper run and live trading.

The engine sees the market only through a Snapshot (the index, VIX and the option
chain of the nearest weekly expiry at one minute), so the same code runs on history
and on live Upstox quotes.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from .rules import Rules

YEAR_MIN = 375 * 252  # trading minutes in a year


@dataclass
class Snapshot:
    underlying: str
    ts: dt.datetime
    spot: float
    day_open: float
    prev_close: float
    or_high: float          # opening-range high and low (first or_minutes)
    or_low: float
    vix: float
    expiry: dt.date
    dte: int                # sessions to expiry (0 on expiry day)
    lot: float
    chain: dict             # (strike, "CE"/"PE") -> last price
    event: str = ""         # name of a scheduled event today, if any


@dataclass
class Leg:
    strike: float
    opt: str                # CE / PE
    side: int               # -1 sold, +1 bought
    entry: float            # fill price per unit (after slippage)


@dataclass
class Position:
    underlying: str
    expiry: dt.date
    structure: str          # put spread / call spread / iron condor / short put / ...
    legs: list
    qty: float              # units = lots x lot size
    lots: int
    credit: float           # net premium received per unit, after entry slippage
    max_loss: float         # planned worst case in rupees (hedged: hard cap; unhedged: at the stop)
    entry_ts: dt.datetime
    entry_cost: float       # brokerage, taxes and fees at entry, rupees
    hedged: bool
    exit_ts: dt.datetime | None = None
    exit_value: float = math.nan
    exit_reason: str = ""
    exit_cost: float = 0.0
    notes: dict = field(default_factory=dict)

    def pnl(self) -> float:
        return (self.credit - self.exit_value) * self.qty - self.entry_cost - self.exit_cost


def years_left(ts: dt.datetime, dte: int) -> float:
    close = ts.replace(hour=15, minute=30, second=0, microsecond=0)
    mins = max((close - ts).total_seconds() / 60, 1.0) + 375 * dte
    return mins / YEAR_MIN


def _d1(F, K, T, s):
    return (np.log(F / K) + 0.5 * s * s * T) / (s * np.sqrt(T))


def bs_price(F, K, T, s, call):
    d1 = _d1(F, K, T, s)
    d2 = d1 - s * np.sqrt(T)
    return np.where(call, F * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - F * ndtr(-d1))


def implied_vol(px, F, K, T, call, iters=50):
    px, K = np.asarray(px, float), np.asarray(K, float)
    call = np.asarray(call, bool)
    lo, hi = np.full(px.shape, 0.01), np.full(px.shape, 3.0)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        high = bs_price(F, K, T, m, call) > px
        hi, lo = np.where(high, m, hi), np.where(high, lo, m)
    iv = 0.5 * (lo + hi)
    intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    return np.where((px > intrinsic + 0.01) & (iv < 2.95) & (iv > 0.011), iv, np.nan)


def forward(chain: dict, spot: float) -> float:
    ks = sorted({k for k, o in chain if (k, "CE") in chain and (k, "PE") in chain})
    if not ks:
        return spot
    k = min(ks, key=lambda x: abs(x - spot))
    return k + chain[(k, "CE")] - chain[(k, "PE")]


def otm_deltas(snap: Snapshot) -> dict:
    """|delta| of every out-of-the-money option in the chain."""
    T = years_left(snap.ts, snap.dte)
    F = forward(snap.chain, snap.spot)
    out = {}
    for opt in ("CE", "PE"):
        ks = np.array(sorted(k for k, o in snap.chain if o == opt and ((k > F) if opt == "CE" else (k < F))))
        if not len(ks):
            continue
        px = np.array([snap.chain[(k, opt)] for k in ks])
        iv = implied_vol(px, F, ks, T, opt == "CE")
        d = ndtr(_d1(F, ks, T, np.nan_to_num(iv, nan=1e-6)))
        d = d if opt == "CE" else 1 - d
        for k, x, v in zip(ks, d, iv):
            if v == v:
                out[(float(k), opt)] = float(x)
    return out


def side_cost(price: float, qty: float, buy: bool, r: Rules, underlying: str) -> float:
    """Brokerage, taxes and fees in rupees for one order (slippage is in the fill price)."""
    turnover = price * qty
    fees = r.brokerage + (r.exch_fee.get(underlying, 0.00035) + r.sebi_fee) * turnover
    fees *= 1 + r.gst
    return fees + (r.stamp_buy * turnover if buy else r.stt_sell * turnover)


def slip(price: float, r: Rules, stop: bool = False) -> float:
    return max(price * (r.slippage + (r.stop_slippage if stop else 0.0)), r.min_slippage)


def skip_reason(snap: Snapshot, r: Rules) -> str:
    """Rule 5 / rule 9: reasons not to trade today ('' = OK to look for a trade)."""
    if snap.event:
        return f"event: {snap.event}"
    if snap.dte > r.max_dte:
        return "expiry too far"
    if snap.prev_close and abs(snap.day_open / snap.prev_close - 1) * 100 > r.gap_max_pct:
        return "big opening gap"
    if snap.vix == snap.vix and snap.vix > r.vix_max:
        return "VIX too high"
    if snap.day_open and (snap.or_high - snap.or_low) / snap.day_open * 100 > r.or_max_pct:
        return "wild first half hour"
    return ""


def trend_sides(snap: Snapshot, r: Rules) -> tuple[str, ...]:
    """Rule 6: which sides to sell. Up-trend: puts only; down-trend: calls only."""
    if r.trend == "none" or not snap.day_open:
        return ("PE", "CE")
    move = (snap.spot / snap.day_open - 1) * 100
    if move > r.trend_pct:
        return ("PE",)
    if move < -r.trend_pct:
        return ("CE",)
    return ("PE", "CE")


def plan_entry(snap: Snapshot, r: Rules, capital: float | None = None) -> tuple[Position | None, str]:
    """Pick strikes and size for a new position, or say why not."""
    why = skip_reason(snap, r)
    if why:
        return None, why
    capital = capital or r.capital
    deltas = otm_deltas(snap)
    wing = r.wing.get(snap.underlying, 100)
    legs, credit_mid = [], 0.0
    for opt in trend_sides(snap, r):
        cands = [(k, d) for (k, o), d in deltas.items() if o == opt]
        if not cands:
            return None, "no option prices"
        k, _ = min(cands, key=lambda kd: abs(kd[1] - r.short_delta))
        p = snap.chain[(k, opt)]
        legs.append(Leg(k, opt, -1, p - slip(p, r)))
        credit_mid += p
        if r.hedged:
            hk = k + wing if opt == "CE" else k - wing
            if (hk, opt) not in snap.chain:
                return None, "hedge strike not quoted"
            hp = snap.chain[(hk, opt)]
            legs.append(Leg(hk, opt, +1, hp + slip(hp, r)))
            credit_mid -= hp
    credit = sum(-l.side * l.entry for l in legs)
    if credit < r.min_credit_pts:
        return None, "premium too small"
    if r.hedged and credit < r.min_credit_frac * wing:
        return None, "premium too small for the hedge"
    n_short = sum(1 for l in legs if l.side < 0)
    if r.hedged:
        loss_per_unit = wing - credit           # true worst case of a credit spread / condor
    else:
        loss_per_unit = r.stop_mult * credit    # planned only: the stop can fill worse
    est_cost_unit = 2 * sum(slip(l.entry, r) for l in legs)  # exit slippage allowance
    per_lot = (loss_per_unit + est_cost_unit) * snap.lot + 2 * len(legs) * r.brokerage * 1.2
    lots = int(r.risk_rupees() * capital / r.capital // per_lot) if per_lot > 0 else 0
    if r.hedged:
        margin_lot = (wing * snap.lot) * 1.1
    else:
        margin_lot = r.naked_margin_frac * snap.spot * snap.lot * (1 if n_short == 1 else 1.15)
    lots = min(lots, int(r.max_margin_frac * capital // margin_lot))
    if lots < 1:
        return None, "1 lot would risk more than the limit"
    qty = lots * snap.lot
    cost = sum(side_cost(l.entry, qty, l.side > 0, r, snap.underlying) for l in legs)
    name = {("PE",): "put spread", ("CE",): "call spread", ("PE", "CE"): "iron condor"}[
        tuple(dict.fromkeys(l.opt for l in legs if l.side < 0))]
    if not r.hedged:
        name = {"put spread": "short put", "call spread": "short call", "iron condor": "short strangle"}[name]
    pos = Position(snap.underlying, snap.expiry, name, legs, qty, lots, credit,
                   per_lot * lots, snap.ts, cost, r.hedged,
                   notes={"spot": snap.spot, "vix": snap.vix, "dte": snap.dte,
                          "deltas": {f"{l.strike:g}{l.opt}": round(deltas.get((l.strike, l.opt), math.nan), 3)
                                     for l in legs}})
    return pos, ""


def value(pos: Position, px: dict, worst: bool = False, hi: dict | None = None, lo: dict | None = None) -> float:
    """Cost per unit to close the position now. worst=True uses each leg's adverse
    extreme in the minute (shorts at the high, longs at the low)."""
    v = 0.0
    for l in pos.legs:
        key = (l.strike, l.opt)
        if worst:
            p = (hi if l.side < 0 else lo).get(key, px.get(key))
        else:
            p = px.get(key)
        if p is None or p != p:
            return math.nan
        v += -l.side * p
    return v


def check_exit(pos: Position, ts: dt.datetime, close: dict, open_: dict, hi: dict, lo: dict,
               r: Rules, force_time: bool = False) -> tuple[str, float] | None:
    """Rules 7 and 8 and the stop-loss, one minute bar at a time.
    Returns (reason, fill value per unit before slippage) or None to hold."""
    stop_v = pos.credit * (1 + r.stop_mult)
    tp_v = pos.credit * (1 - r.take_profit)
    v_open = value(pos, open_)
    v_close = value(pos, close)
    if v_close != v_close:
        return None
    v_worst = value(pos, close, worst=True, hi=hi, lo=lo) if r.stop_trigger == "high" else v_close
    if r.stop_mult > 0 and v_worst == v_worst and v_worst >= stop_v:
        # stop-market order: fills at the stop, or worse if the price jumped past it
        if r.stop_trigger == "high":
            fill = max(stop_v, v_open) if v_open == v_open else max(stop_v, v_close)
        else:
            fill = max(stop_v, v_close)
        if pos.hedged:
            fill = min(fill, r.wing.get(pos.underlying, 100))
        return "stop-loss", fill
    if v_close <= tp_v:
        return "profit target", v_close
    if force_time or ts.time() >= dt.time(*map(int, r.exit_time.split(":"))):
        return "time exit", v_close
    return None


def close_position(pos: Position, ts: dt.datetime, reason: str, fill_value: float, px: dict, r: Rules):
    stop = reason == "stop-loss"
    slip_total = sum(slip(px.get((l.strike, l.opt), 0.0) or 0.0, r, stop) for l in pos.legs)
    pos.exit_ts = ts
    pos.exit_reason = reason
    pos.exit_value = fill_value + slip_total
    pos.exit_cost = sum(side_cost(px.get((l.strike, l.opt), 0.0) or 0.0, pos.qty, l.side < 0, r, pos.underlying)
                        for l in pos.legs)
    return pos
