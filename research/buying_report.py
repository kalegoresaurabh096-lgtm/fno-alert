"""Analytics for buying stock options (5-day hold, entry next open, exit close, costs as strategy.simulate)."""
import pandas as pd, numpy as np
pd.set_option("display.width", 220)
L = pd.read_parquet("research/trade_labels.parquet"); L["date"] = pd.to_datetime(L["date"])
oos = pd.read_parquet("research/oos_big.parquet")[["date", "symbol", "p_big", "big", "lot", "F"]]; oos["date"] = pd.to_datetime(oos["date"])
print("all stock-days", len(L), L.date.min().date(), L.date.max().date())
def st(x):
    x = x.dropna(); return pd.Series(dict(n=len(x), win=(x > 0).mean(), avg=x.mean(), median=x.median(), p5=x.quantile(.05), p95=x.quantile(.95), big_win=(x > 1).mean()))
print("\n== All stock-days, 2 years (return on premium paid)")
print(pd.DataFrame({k: st(L[k]) for k in ("ce_ret", "pe_ret", "strad_ret")}).T.round(3).to_string())
M = L.merge(oos, on=["date", "symbol"])
M["dec"] = pd.qcut(M.p_big, 10, labels=False) + 1
print("\n== By model score decile (last 11 months, out-of-sample)")
print(M.groupby("dec").agg(n=("strad_ret", "count"), big_rate=("big", "mean"), strad=("strad_ret", "mean"), ce=("ce_ret", "mean"), pe=("pe_ret", "mean"), strad_win=("strad_ret", lambda x: (x.dropna() > 0).mean())).round(3).to_string())
A = M[M.p_big >= 0.3011]
print("\n== Alerts only (p_big >= 0.30)"); print(pd.DataFrame({k: st(A[k]) for k in ("ce_ret", "pe_ret", "strad_ret")}).T.round(3).to_string())
print("big-move rate on alerts", round(A.big.mean(), 3), "vs all", round(M.big.mean(), 3))
# perfect direction hindsight: best of CE/PE
print("\n== If you knew the direction in advance (hindsight, alerts): avg", round(A[["ce_ret", "pe_ret"]].max(axis=1).mean(), 3))
L["m"] = L.date.dt.to_period("M")
mm = L.groupby("m").agg(n=("strad_ret", "count"), strad=("strad_ret", "mean"), ce=("ce_ret", "mean"), pe=("pe_ret", "mean"), strad_win=("strad_ret", lambda x: (x.dropna() > 0).mean()))
print("\n== Month by month, all stocks"); print(mm.round(3).to_string())
A = A.dropna(subset=["strad_ret", "strad_cost_pct"]).copy(); A["m"] = A.date.dt.to_period("M")
print("\n== Alerts month by month"); print(A.groupby("m").agg(n=("strad_ret", "size"), avg=("strad_ret", "mean"), win=("strad_ret", lambda x: (x > 0).mean())).round(3).to_string())
# account sim on alerts: Rs 5L, straddle cost per lot = strad_cost_pct*F*lot, take if <= 50k and open capital <= 2.5L, one per stock, sequential
A = A.sort_values("date"); cap = 500000.0; eq = cap; open_ = []; taken = 0; curve = []
dates = sorted(M.date.unique())
idx = {d: i for i, d in enumerate(dates)}
for r in A.itertuples():
    i = idx[r.date]
    still = []
    for (j, cost, pnl) in open_:
        if i >= j + 6: eq += pnl
        else: still.append((j, cost, pnl))
    open_ = still
    cost = r.strad_cost_pct * r.F * r.lot
    if cost > 0.10 * cap or sum(c for _, c, _ in open_) + cost > 0.5 * cap: continue
    open_.append((i, cost, cost * r.strad_ret)); taken += 1; curve.append((r.date, eq))
eq += sum(p for _, _, p in open_)
c = pd.DataFrame(curve, columns=["date", "eq"])
print(f"\n== Rs 5 lakh account on alerts (11 months): trades {taken}, end equity Rs {eq:,.0f} ({eq/cap-1:+.1%}), lowest Rs {min(c["eq"].min(), eq):,.0f}")
# all-stock random account sim (like the selling one): Rs 5L, up to 10 trades a month, 1 lot, cost <= Rs 50k
P = pd.read_parquet("research/panel_min.parquet"); P["date"] = pd.to_datetime(P["date"])
X = L.merge(P[["symbol", "date", "lot", "F"]], on=["symbol", "date"])
rng = np.random.default_rng(0)
print("\n== Rs 5 lakh account, all stocks, 10 random trades a month, 1 lot each (cost <= Rs 50k), 500 draws, 2 years")
for kind in ("strad", "ce", "pe"):
    col = f"{kind}_ret"
    Y = X.dropna(subset=[col, "strad_cost_pct"]).copy()
    prem = Y.strad_cost_pct * Y.F * Y.lot
    Y["cost"] = prem if kind == "strad" else prem / 2
    Y = Y[Y.cost <= 50000]; Y["pnl"] = Y.cost * Y[col]
    groups = [g for _, g in Y.groupby("m")]
    tot, worst = [], []
    for _ in range(300):
        mp = [g.sample(min(10, len(g)), random_state=int(rng.integers(1e9))).pnl.sum() for g in groups]
        tot.append(sum(mp)); worst.append(min(mp))
    tot, worst = np.array(tot), np.array(worst)
    print(f"{kind}: {len(groups)} months, total median Rs {np.median(tot):,.0f} (10-90%: {np.percentile(tot,10):,.0f} to {np.percentile(tot,90):,.0f}), worst month median Rs {np.median(worst):,.0f}, P(profit)={np.mean(tot>0):.2f}")
