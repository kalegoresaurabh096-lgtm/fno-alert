"""NIFTY iron-condor selling backtest, with the rules of the options seller Saurabh follows.

Data: data/nifty (scripts/nifty_backfill.py, branch nifty-data).

Trade:
* Enter at the close `dte` calendar days (or the nearest session after) before an expiry.
* Sell the CE and PE whose delta is closest to `short_delta` (Black-76 on the
  put-call-parity forward), buy wings `wing` points further out.
* Exit at the close of the last session before expiry (no expiry-day risk), or
  earlier under the stop rule:
    - "delta": close the whole condor at the close when a short leg's delta >= 0.30
    - "loss2x": close when the loss reaches 2x the credit
    - "none": hold.
* One condor at a time; next one opens at the first entry date after the last closed.
* Costs per leg per side: 1% slippage (min Rs 0.25), Rs 20 brokerage, STT 0.1% on sells,
  exchange and SEBI fees, GST, stamp duty (strategy.side_cost).
* Capital Rs 5 lakh, 1 lot (lot size from the data). Max loss per trade = wing x lot.
"""
from __future__ import annotations

import glob
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from fno_alert import black76, strategy  # noqa: E402

D = os.environ.get("NIFTY_DIR", "data/nifty")
R = strategy.TradeRules(slippage=0.01, min_slippage=0.25)
CAPITAL = 500_000
pd.set_option("display.width", 220)


def load():
    idx = pd.read_csv(f"{D}/index.csv", parse_dates=["date"]).set_index("date").sort_index()
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/opt/*.parquet"))]
    o = pd.concat([p for p in parts if len(p)], ignore_index=True)
    o["date"] = pd.to_datetime(o["date"]).dt.normalize()
    o["expiry"] = pd.to_datetime(o["expiry"])
    o = o[o["close"] > 0]
    return idx, o


def chain_on(o_exp: pd.DataFrame, d) -> pd.DataFrame:
    c = o_exp[o_exp["date"] == d]
    return c.pivot_table(index="strike", columns="opt", values="close") if len(c) else pd.DataFrame()


def forward(ch: pd.DataFrame, spot: float, T: float) -> float:
    if not {"CE", "PE"} <= set(ch.columns):
        return spot
    both = ch.dropna(subset=["CE", "PE"])
    if both.empty:
        return spot
    k = both.index[np.argmin(np.abs(both.index - spot))]
    return float(k + (both.at[k, "CE"] - both.at[k, "PE"]) * math.exp(0.065 * T))


def deltas(ch: pd.DataFrame, F: float, T: float) -> pd.DataFrame:
    out = []
    for opt in ("CE", "PE"):
        if opt not in ch:
            continue
        s = ch[opt].dropna()
        iv = black76.implied_vol(s.values, np.full(len(s), F), s.index.values, np.full(len(s), T), opt == "CE")
        d1 = (np.log(F / s.index.values) + 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
        from scipy.special import ndtr
        dl = ndtr(d1) if opt == "CE" else ndtr(d1) - 1
        out.append(pd.DataFrame({"strike": s.index.values, "opt": opt, "px": s.values, "iv": iv, "delta": np.abs(dl)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run(idx, o, dte=7, short_delta=0.15, wing=300, stop="delta", vix_max=None, vix_min=None):
    days = idx.index
    exps = sorted(o["expiry"].unique())
    by_exp = {e: g for e, g in o.groupby("expiry")}
    trades, busy_until = [], pd.Timestamp(0)
    for e in exps:
        e = pd.Timestamp(e)
        target = e - pd.Timedelta(days=dte)
        cand = days[(days >= target) & (days < e)]
        if not len(cand) or cand[0] <= busy_until:
            continue
        d0 = cand[0]
        exit_day = days[days < e][-1] if len(days[days < e]) else None
        if exit_day is None or exit_day <= d0:
            continue
        vix = idx.at[d0, "vix"] if "vix" in idx else np.nan
        if (vix_max and vix > vix_max) or (vix_min and vix < vix_min):
            continue
        g = by_exp[e]
        T0 = max((e - d0).days, 0.5) / 365
        ch = chain_on(g, d0)
        if ch.empty:
            continue
        spot = float(idx.at[d0, "close"])
        F = forward(ch, spot, T0)
        dl = deltas(ch, F, T0).dropna()
        ce = dl[(dl.opt == "CE") & (dl.strike > F)]
        pe = dl[(dl.opt == "PE") & (dl.strike < F)]
        if ce.empty or pe.empty:
            continue
        sc = ce.iloc[np.argmin(np.abs(ce.delta - short_delta))]
        sp = pe.iloc[np.argmin(np.abs(pe.delta - short_delta))]
        lc, lp = sc.strike + wing, sp.strike - wing
        legs = [("CE", sc.strike, -1), ("PE", sp.strike, -1), ("CE", lc, 1), ("PE", lp, 1)]
        try:
            e_px = [float(ch.at[k, opt]) for opt, k, _ in legs]
        except KeyError:
            continue
        if any(not (p == p and p > 0) for p in e_px):
            continue
        lot = float(g["lot"].dropna().iloc[-1]) if "lot" in g and g["lot"].notna().any() else 75.0
        credit = e_px[0] + e_px[1] - e_px[2] - e_px[3]
        if credit <= 0:
            continue
        # walk forward day by day
        path = days[(days > d0) & (days <= exit_day)]
        x_px, x_day, reason = None, None, "expiry-1"
        for d in path:
            chd = chain_on(g, d)
            px = []
            for opt, k, _ in legs:
                v = chd.at[k, opt] if (len(chd) and k in chd.index and opt in chd) else np.nan
                if not v == v:  # no trade that day: intrinsic from the index close
                    S = float(idx.at[d, "close"])
                    v = max(S - k, 0) if opt == "CE" else max(k - S, 0)
                px.append(float(v))
            value = px[0] + px[1] - px[2] - px[3]
            hit = False
            if stop == "loss2x" and value - credit >= 2 * credit:
                hit = True
            elif stop == "delta" and len(chd):
                T = max((e - d).days, 0.5) / 365
                Fd = forward(chd, float(idx.at[d, "close"]), T)
                dd = deltas(chd.loc[[k for k in (sc.strike, sp.strike) if k in chd.index]], Fd, T)
                if len(dd):
                    sd = dd[((dd.opt == "CE") & (dd.strike == sc.strike)) | ((dd.opt == "PE") & (dd.strike == sp.strike))]
                    if (sd.delta >= 0.30).any():
                        hit = True
            x_px, x_day = px, d
            if hit:
                reason = "stop"
                break
        if x_px is None:
            continue
        pnl = 0.0
        for (opt, k, s), ev, xv in zip(legs, e_px, x_px):
            if s < 0:
                pnl += (ev - strategy.side_cost(ev, lot, False, R)) - (xv + strategy.side_cost(xv, lot, True, R))
            else:
                pnl += (xv - strategy.side_cost(xv, lot, False, R)) - (ev + strategy.side_cost(ev, lot, True, R))
        trades.append(dict(entry=d0, exit=x_day, expiry=e, spot=spot, vix=vix, short_ce=sc.strike, short_pe=sp.strike,
                           credit=credit, credit_rs=credit * lot, max_loss_rs=(wing - credit) * lot, lot=lot,
                           pnl_rs=pnl * lot, reason=reason))
        busy_until = x_day
    return pd.DataFrame(trades)


def summary(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    eq = CAPITAL + t["pnl_rs"].cumsum()
    years = max((t["exit"].max() - t["entry"].min()).days / 365, 0.1)
    return dict(trades=len(t), win=round((t.pnl_rs > 0).mean(), 3), avg_credit=round(t.credit_rs.mean()),
                avg_pnl=round(t.pnl_rs.mean()), worst=round(t.pnl_rs.min()), best=round(t.pnl_rs.max()),
                total=round(t.pnl_rs.sum()), per_year_pct=round(t.pnl_rs.sum() / CAPITAL / years * 100, 1),
                max_dd=round((eq - eq.cummax()).min()), stops=int((t.reason == "stop").sum()))


if __name__ == "__main__":
    idx, o = load()
    print("NIFTY options:", o["expiry"].nunique(), "expiries,", o["date"].min().date(), "to", o["date"].max().date(),
          "lots:", sorted(o["lot"].dropna().unique()))
    rows = []
    for dte, sd, w, stop in itertools.product([7, 14, 28], [0.10, 0.15, 0.20], [200, 300, 500], ["none", "delta", "loss2x"]):
        t = run(idx, o, dte, sd, w, stop)
        s = summary(t)
        if s:
            rows.append(dict(dte=dte, delta=sd, wing=w, stop=stop, **s))
            t.to_parquet(f"research/nifty_trades_{dte}_{sd}_{w}_{stop}.parquet")
        print(dte, sd, w, stop, s, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("research/nifty_condor_grid.csv", index=False)
    print(res.sort_values("total", ascending=False).to_string(index=False))
