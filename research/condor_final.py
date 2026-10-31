"""Final condor numbers. Rows whose cost-free P&L is impossible for a condor
(beyond max loss or credit: split-adjusted or bad prices) are dropped from both runs."""
import pandas as pd, numpy as np
pd.set_option("display.width", 220)
K = ["symbol", "d_in", "k", "wing"]
def load(src):
    g = pd.read_parquet(f"research/condor_{src}_0.parquet"); n = pd.read_parquet(f"research/condor_{src}_1.parquet")
    ok = (g.pnl >= -g.max_loss - 0.05 * g.max_loss - 0.05) & (g.pnl <= g.credit + 0.05)
    good = g.loc[ok, K]
    return n.merge(good, on=K), g.merge(good, on=K), (~ok).sum(), len(g)
def summ(T):
    a = T.groupby(["k", "wing"]).agg(trades=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()),
        avg_credit_lot=("credit", lambda x: (x * T.loc[x.index, "lot"]).mean()), avg_risk_lot=("risk_lot", "mean"),
        avg_pnl_lot=("pnl_lot", "mean"), pnl=("pnl_lot", "sum"), risk=("risk_lot", "sum"), worst_lot=("pnl_lot", "min"),
        p5_lot=("pnl_lot", lambda x: x.quantile(.05)))
    a["ret_on_risk"] = a.pnl / a.risk
    return a.drop(columns=["pnl", "risk"]).round(3)
for src in ("panel", "oos"):
    n, g, bad, tot = load(src)
    print(f"\n=== {src}: dropped {bad} of {tot} bad-data trades")
    print("-- before costs"); print(summ(g).to_string())
    print("-- after costs"); print(summ(n).to_string())
n, _, _, _ = load("oos")
n["grp"] = pd.cut(n["rank"], [0, .2, .5, .8, 1], labels=["calmest20", "20-50", "50-80", "riskiest20"])
print("\n=== model filter, after costs (last 11 months)")
f = n.groupby(["k", "wing", "grp"], observed=True).agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()), avg_lot=("pnl_lot", "mean"), pnl=("pnl_lot", "sum"), risk=("risk_lot", "sum"))
f["ror"] = f.pnl / f.risk; print(f.drop(columns=["pnl", "risk"]).round(3).to_string())
P, _, _, _ = load("panel")
for k, w in ((1.0, 0.03), (0.5, 0.06)):
    m = P[(P.k == k) & (P.wing == w)].groupby(P.d_in.dt.to_period("M")).agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()), pnl=("pnl_lot", "sum"), risk=("risk_lot", "sum"))
    m["ror"] = m.pnl / m.risk; print(f"\n=== by month k={k} wing={w}, after costs"); print(m.round(3).to_string())
rng = np.random.default_rng(0)
print("\n=== Rs 5 lakh account: 1 lot each, risk/trade <= Rs 25k, up to 10 trades a month, 500 random draws, after costs")
for (k, w), gg in P.groupby(["k", "wing"]):
    gg = gg[gg.risk_lot <= 25000]; gg = gg.assign(m=gg.d_in.dt.to_period("M"))
    months = sorted(gg.m.unique())
    if len(months) < 10: continue
    tot, worst = [], []
    for _ in range(500):
        mp = [gm.sample(min(10, len(gm)), random_state=int(rng.integers(1e9))).pnl_lot.sum() for _, gm in gg.groupby("m")]
        tot.append(sum(mp)); worst.append(min(mp))
    tot, worst = np.array(tot), np.array(worst)
    print(f"k={k} wing={w}: {len(months)} months, total median Rs {np.median(tot):,.0f} (10-90%: {np.percentile(tot,10):,.0f} to {np.percentile(tot,90):,.0f}), worst month median Rs {np.median(worst):,.0f}, P(profit)={np.mean(tot>0):.2f}")
