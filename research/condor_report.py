import pandas as pd, numpy as np
pd.set_option("display.width", 220)
def summ(T):
    return T.groupby(["k", "wing"]).agg(trades=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()),
        avg_pnl_lot=("pnl_lot", "mean"), avg_risk_lot=("risk_lot", "mean"),
        ret_on_risk=("pnl_lot", lambda x: x.sum()), worst_lot=("pnl_lot", "min"), best_lot=("pnl_lot", "max")).assign(
        ret_on_risk=lambda d: d.ret_on_risk / T.groupby(["k", "wing"])["risk_lot"].sum()).round(3)
for src in ("panel", "oos"):
    for cost in (0, 1):
        T = pd.read_parquet(f"research/condor_{src}_{cost}.parquet")
        print(f"\n=== {src} costs={cost} months={T.d_in.dt.to_period('M').nunique()} {T.d_in.min().date()}..{T.d_out.max().date()}")
        print(summ(T).to_string())
T = pd.read_parquet("research/condor_oos_1.parquet")
T["grp"] = pd.cut(T["rank"], [0, .2, .5, .8, 1], labels=["calmest20", "20-50", "50-80", "riskiest20"])
print("\n=== model filter (oos, costs)")
print(T.groupby(["k", "wing", "grp"], observed=True).agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()),
      avg_lot=("pnl_lot", "mean"), ror=("pnl_lot", "sum"), risk=("risk_lot", "sum")).assign(ror=lambda d: d.ror / d.risk).drop(columns="risk").round(3).to_string())
P = pd.read_parquet("research/condor_panel_1.parquet")
print("\n=== by month (panel, costs, k=1.0 wing 3%) return on risk")
m = P[(P.k == 1.0) & (P.wing == 0.03)].groupby(P.d_in.dt.to_period("M")).agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()), pnl=("pnl_lot", "sum"), risk=("risk_lot", "sum"))
m["ror"] = m.pnl / m.risk
print(m.round(3).to_string())

print("\n=== CLEANED: drop impossible P&L (beyond max loss/credit, i.e. split/bad-price rows)")
def clean(T):
    lot = T.lot
    ok = (T.pnl_lot >= -(T.max_loss + 1.0 + 0.1 * T.max_loss) * lot) & (T.pnl_lot <= (T.credit + 0.5) * lot)
    return T[ok], (~ok).sum()
for src in ("panel", "oos"):
    for cost in (0, 1):
        T, bad = clean(pd.read_parquet(f"research/condor_{src}_{cost}.parquet"))
        print(f"\n{src} costs={cost} dropped={bad}")
        print(summ(T).to_string())
P, _ = clean(pd.read_parquet("research/condor_panel_1.parquet"))
rng = np.random.default_rng(0)
print("\n=== Rs 5 lakh account, 1 lot per trade, risk/trade <= Rs 25k, up to 10 trades/month (total risk <= 2.5L), 500 random draws")
for (k, w), g in P.groupby(["k", "wing"]):
    g = g[g.risk_lot <= 25000]
    months = sorted(g.d_in.dt.to_period("M").unique())
    tot, worst = [], []
    for _ in range(500):
        mp = []
        for m in months:
            gm = g[g.d_in.dt.to_period("M") == m]
            pick = gm.sample(min(10, len(gm)), random_state=int(rng.integers(1e9)))
            mp.append(pick.pnl_lot.sum())
        tot.append(sum(mp)); worst.append(min(mp))
    tot, worst = np.array(tot), np.array(worst)
    print(f"k={k} wing={w}: months={len(months)} total P&L median Rs {np.median(tot):,.0f} (10%..90%: {np.percentile(tot,10):,.0f}..{np.percentile(tot,90):,.0f}); worst month median Rs {np.median(worst):,.0f}, worst-case {worst.min():,.0f}; P(profit)={np.mean(tot>0):.2f}")
T, _ = clean(pd.read_parquet("research/condor_oos_1.parquet"))
T["grp"] = pd.cut(T["rank"], [0, .2, .5, .8, 1], labels=["calmest20", "20-50", "50-80", "riskiest20"])
print("\n=== model filter cleaned (oos, costs)")
print(T.groupby(["k", "wing", "grp"], observed=True).agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()),
      avg_lot=("pnl_lot", "mean"), ror=("pnl_lot", "sum"), risk=("risk_lot", "sum")).assign(ror=lambda d: d.ror / d.risk).drop(columns="risk").round(3).to_string())
m = P[(P.k == 1.0) & (P.wing == 0.03)].groupby(P.d_in.dt.to_period("M")).agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()), pnl=("pnl_lot", "sum"), risk=("risk_lot", "sum"))
m["ror"] = m.pnl / m.risk
print("\n=== by month cleaned (k=1.0 wing 3%)"); print(m.round(3).to_string())
