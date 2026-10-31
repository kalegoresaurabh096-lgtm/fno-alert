"""Hedged premium selling (short iron condor) on single stocks, chosen by the
big-move model's walk-forward (out-of-sample) score.

Monthly cycle: signal on an expiry day, enter at the next session's close
(far OTM legs rarely trade at the open), exit at the close of the last session
before the next expiry (variant: 5 sessions later). Short strikes about
k x expected move away (atm_iv), long wings the next listed strike >= w% further.
All four legs must have traded on entry day. A leg with no exit price is
valued at intrinsic from the stock's close. Costs: strategy.side_cost per leg
per side (2% / Rs 0.15 slippage, brokerage, STT, fees).
"""
from __future__ import annotations
import sys, math
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from fno_alert import strategy
import os
SRC = os.environ.get("SRC", "oos")          # oos: model-scored months; panel: all 2 years, no model
COSTS = os.environ.get("COSTS", "1")
R = strategy.TradeRules() if COSTS == "1" else strategy.TradeRules(slippage=0.01, min_slippage=0.05) if COSTS == "low" else strategy.TradeRules(slippage=0, min_slippage=0, brokerage=0, exch_fee=0,
                                                          sebi_fee=0, stamp_buy=0, stt_sell=0)

if SRC == "oos":
    oos = pd.read_parquet("research/oos_big.parquet")
else:
    oos = pd.read_parquet("research/panel_min.parquet")
    oos["p_big"] = np.nan
oos["date"] = pd.to_datetime(oos["date"])
oos["rank"] = oos.groupby("date")["p_big"].rank(pct=True)
td = np.sort(oos["date"].unique())
c = pd.read_parquet("data/processed/fo_contracts.parquet")
c = c[c["date"] >= td[0]]
c["px"] = np.where(np.isfinite(c["close"]), c["close"], c["settle"])
c = c.set_index(["symbol", "expiry", "date", "opt", "strike"]).sort_index()
spot = oos.set_index(["symbol", "date"])["close"]

def leg(sym, exp, d, opt, k, need_trade=False):
    try:
        r = c.loc[(sym, exp, d, opt, k)]
    except KeyError:
        return math.nan
    if need_trade and not (r["volume"] > 0):
        return math.nan
    return float(r["px"]) if r["px"] == r["px"] else math.nan

def trade(r, d_in, d_out, exp, k_mult, wing_pct):
    sym = r.symbol
    F = r.F if r.F == r.F else r.fut_close
    if not (F > 0 and r.atm_iv == r.atm_iv):
        return None
    try:
        ks = np.sort(c.loc[(sym, exp, d_in)].index.get_level_values("strike").unique())
    except KeyError:
        return None
    T = (exp - d_in) / np.timedelta64(365, "D")
    em = F * r.atm_iv * math.sqrt(max(T, 1 / 365))
    sc = ks[ks >= F + k_mult * em]; sp = ks[ks <= F - k_mult * em]
    if not len(sc) or not len(sp):
        return None
    sc, sp = sc[0], sp[-1]
    lc = ks[ks >= sc + wing_pct * F]; lp = ks[ks <= sp - wing_pct * F]
    if not len(lc) or not len(lp):
        return None
    lc, lp = lc[0], lp[-1]
    legs = [("CE", sc, -1), ("PE", sp, -1), ("CE", lc, +1), ("PE", lp, +1)]
    e = [leg(sym, exp, d_in, o, k, True) for o, k, _ in legs]
    if any(not (v == v and v > 0) for v in e):
        return None
    S = spot.get((sym, d_out), math.nan)
    x = []
    for (o, k, _), ev in zip(legs, e):
        v = leg(sym, exp, d_out, o, k)
        if not v == v:
            if not S == S:
                return None
            v = max(S - k, 0) if o == "CE" else max(k - S, 0)
        x.append(v)
    lot = r.lot if r.lot == r.lot else 1
    pnl = 0.0
    for (o, k, s), ev, xv in zip(legs, e, x):
        if s < 0:   # sold: receive entry, pay exit
            pnl += (ev - strategy.side_cost(ev, lot, False, R)) - (xv + strategy.side_cost(xv, lot, True, R))
        else:
            pnl += (xv - strategy.side_cost(xv, lot, False, R)) - (ev + strategy.side_cost(ev, lot, True, R))
    credit = e[0] + e[1] - e[2] - e[3]
    width = max(lc - sc, sp - lp)
    max_loss = width - credit
    if not (credit > 0 and max_loss > 0):
        return None
    return dict(symbol=sym, signal=r.date, d_in=pd.Timestamp(d_in), d_out=pd.Timestamp(d_out), p_big=r.p_big, rank=r.rank,
                credit=credit, max_loss=max_loss, pnl=pnl, ror=pnl / max_loss, lot=lot,
                pnl_lot=pnl * lot, risk_lot=max_loss * lot, short_dist=(sc - F) / F)

exps = np.sort(c.index.get_level_values("expiry").unique())
rows = []
for k_mult in (0.5, 1.0, 1.5):
    for wing in (0.03, 0.06):
        for hold in ("expiry",):
            for i in range(len(exps) - 1):
                sig = exps[i]
                if sig not in td:
                    j = np.searchsorted(td, sig)
                    if j >= len(td) or (td[j] - sig) / np.timedelta64(1, "D") > 4:
                        continue
                    sig = td[j]
                j = np.searchsorted(td, sig)
                if j + 1 >= len(td):
                    continue
                d_in = td[j + 1]
                nxt = exps[i + 1]
                if (nxt - d_in) / np.timedelta64(1, "D") < 12:
                    continue
                before = td[td < nxt]
                if hold == "expiry":
                    if not len(before) or before[-1] <= d_in or nxt > td[-1]:
                        continue
                    d_out = before[-1]
                else:
                    if j + 6 >= len(td):
                        continue
                    d_out = min(td[j + 6], before[-1])
                for r in oos[oos.date == sig].itertuples(index=False):
                    t = trade(r, d_in, d_out, nxt, k_mult, wing)
                    if t:
                        t.update(k=k_mult, wing=wing, hold=hold)
                        rows.append(t)
            print("done", k_mult, wing, hold, len(rows), flush=True)
T = pd.DataFrame(rows)
T.to_parquet(f"research/condor_{SRC}_{COSTS}.parquet")
