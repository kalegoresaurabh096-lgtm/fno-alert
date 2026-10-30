"""Do a stop-loss, a profit target, a longer hold or a VIX filter improve the
alert straddle (ATM CE + PE, entry next open, costs as in strategy.simulate)?
Uses daily closes only, so stops and targets trigger at the close (no intraday peeking).
Walk-forward out-of-sample alerts from research/oos_big.parquet."""
from __future__ import annotations
import sys, math, itertools
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from fno_alert import strategy
R = strategy.TradeRules()
TH = 0.3011

oos = pd.read_parquet("research/oos_big.parquet")
oos["date"] = pd.to_datetime(oos["date"])
td = np.sort(oos["date"].unique()).astype("datetime64[ns]")
al = oos[oos.p_big >= TH].copy()
syms = set(al.symbol)
c = pd.read_parquet("data/processed/fo_contracts.parquet", filters=[("symbol", "in", list(syms))])
c["date"] = pd.to_datetime(c["date"]).values.astype("datetime64[ns]")
c["expiry"] = pd.to_datetime(c["expiry"]).values.astype("datetime64[ns]")
c["px"] = np.where(np.isfinite(c["close"]), c["close"], c["settle"])
g = {k: v.set_index("date")[["open", "px"]] for k, v in c.groupby(["symbol", "expiry", "opt", "strike"])}
exps = c.groupby(["symbol", "date"])["expiry"].unique()
strikes = c.groupby(["symbol", "date", "expiry"])["strike"].unique()

paths = []  # per alert: entry cost, lot, list of (ce_close+pe_close) for days 1..10, vix
for r in al.itertuples(index=False):
    d0 = np.datetime64(r.date, "ns"); i0 = np.searchsorted(td, d0)
    if i0 + 1 >= len(td): continue
    try: ex = sorted(exps.loc[(r.symbol, d0)])
    except KeyError: continue
    exp = next((e for e in ex if (e - d0) / np.timedelta64(1, "D") >= R.min_trade_dte), None)
    F = r.F if r.F == r.F else r.fut_close
    if exp is None or not F == F: continue
    ks = strikes.loc[(r.symbol, d0, exp)]; k = float(ks[np.argmin(np.abs(ks - F))])
    ce, pe = g.get((r.symbol, exp, "CE", k)), g.get((r.symbol, exp, "PE", k))
    if ce is None or pe is None: continue
    d_in = td[i0 + 1]
    try: e = ce.at[d_in, "open"] + pe.at[d_in, "open"]; e1, e2 = ce.at[d_in, "open"], pe.at[d_in, "open"]
    except KeyError: continue
    if not (e1 > 0 and e2 > 0): continue
    vals = []
    for j in range(1, 11):
        if i0 + j >= len(td) or td[i0 + j] >= exp: break
        d = td[i0 + j]
        try: vals.append((ce.at[d, "px"], pe.at[d, "px"]))
        except KeyError: vals.append((np.nan, np.nan))
    if len(vals) < 5: continue
    paths.append(dict(date=r.date, symbol=r.symbol, lot=r.lot, e1=e1, e2=e2, vals=vals, vix=r.vix, vix_rank=r.vix_rank, big=r.big))
P = pd.DataFrame(paths); print("alerts with full paths:", len(P), "of", len(al))

def ret(p, x1, x2):
    lot = p.lot if p.lot == p.lot else 1
    pnl = sum((x - strategy.side_cost(x, lot, False, R)) - (e + strategy.side_cost(e, lot, True, R)) for e, x in ((p.e1, x1), (p.e2, x2)))
    return pnl / (p.e1 + p.e2)

def run(hold, tgt, stop):
    out = []
    for p in P.itertuples(index=False):
        last = None; res = None
        for j, (a, b) in enumerate(p.vals[:hold], 1):
            if not (a == a and b == b): continue
            last = (a, b); rr = (a + b) / (p.e1 + p.e2) - 1
            if (tgt and rr >= tgt) or (stop and rr <= -stop): res = ret(p, a, b); break
        if res is None and last: res = ret(p, *last)
        out.append(res)
    return pd.Series(out, index=P.index)

rows = []
for hold, tgt, stop in itertools.product([3, 5, 10], [None, .2, .3, .5, 1.0], [None, .2, .3, .5]):
    s = run(hold, tgt, stop).dropna()
    h1 = s[P.loc[s.index, "date"] < P.date.median()]; h2 = s[P.loc[s.index, "date"] >= P.date.median()]
    rows.append(dict(hold=hold, target=tgt, stop=stop, n=len(s), mean=s.mean(), median=s.median(), win=(s > 0).mean(),
                     first_half=h1.mean(), second_half=h2.mean()))
T = pd.DataFrame(rows).sort_values("mean", ascending=False)
pd.set_option("display.width", 200)
print(T.round(3).to_string(index=False))
base = run(5, None, None)
P["r5"] = base
print("\nVIX filter (plain 5-day exit):")
P["vixq"] = pd.qcut(P.vix, 4, labels=["low", "mid-low", "mid-high", "high"])
print(P.groupby("vixq", observed=True)["r5"].agg(["count", "mean", "median", lambda x: (x > 0).mean()]).round(3))
P["vrq"] = pd.qcut(P.vix_rank, 4, labels=["low", "mid-low", "mid-high", "high"], duplicates="drop")
print(P.groupby("vrq", observed=True)["r5"].agg(["count", "mean", "median", lambda x: (x > 0).mean()]).round(3))
P.drop(columns="vals").to_parquet("research/exit_paths.parquet")
