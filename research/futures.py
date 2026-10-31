"""Stock-futures strategies on the F&O stocks, daily closes, signals lagged 1 day.
Costs: 0.06% per side of traded notional (slippage + brokerage + STT + fees, conservative for futures).
Development: 2023-01 .. 2025-10. Unseen holdout: 2025-11 .. 2026-10."""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
o = pd.read_parquet("data/processed/ohlcv.parquet")
px = o.pivot_table(index="date", columns="symbol", values="close").sort_index()
vol = o.pivot_table(index="date", columns="symbol", values="volume").sort_index()
hi = o.pivot_table(index="date", columns="symbol", values="high").sort_index()
lo = o.pivot_table(index="date", columns="symbol", values="low").sort_index()
ret = px.pct_change(fill_method=None).clip(-0.35, 0.35)
nifty = pd.read_parquet("data/processed/market.parquet")["nifty"].reindex(px.index).pct_change(fill_method=None)
COST = 0.0006
HOLD = pd.Timestamp("2025-11-01")

def run(w, name):
    """w: target weights (sum |w| = 1 gross), decided at close t, earn from t+1."""
    w = w.fillna(0)
    pos = w.shift(1).fillna(0)
    gross = (pos * ret).sum(axis=1)
    turn = (w - w.shift(1)).abs().sum(axis=1).shift(1).fillna(0)
    net = gross - turn * COST
    out = {}
    for lab, s in (("dev", net[net.index < HOLD]), ("holdout", net[net.index >= HOLD])):
        s = s[s.index >= w.dropna(how="all").index[0]]
        s = s.iloc[1:] if len(s) else s
        if len(s) < 20: continue
        ann = s.mean() * 252; vol_ = s.std() * np.sqrt(252)
        eq = (1 + s).cumprod(); dd = (eq / eq.cummax() - 1).min()
        out[lab] = f"ann {ann:+.1%} vol {vol_:.1%} sharpe {ann/vol_ if vol_ else 0:+.2f} maxDD {dd:.1%} turnover/yr {turn.loc[s.index].sum()/len(s)*252:.0f}x"
    print(f"{name:42s} | " + " | ".join(f"{k}: {v}" for k, v in out.items()))
    return net

def xs(score, top=0.1, long_only=False, rebal=5):
    """long top decile, short bottom decile, equal weight, rebalance every `rebal` days."""
    r = score.rank(axis=1, pct=True)
    n = score.notna().sum(axis=1)
    L = (r >= 1 - top).astype(float); S = (r <= top).astype(float)
    L = L.div(L.sum(axis=1), axis=0); S = S.div(S.sum(axis=1), axis=0)
    w = (L if long_only else 0.5 * L - 0.5 * S)
    w[n < 50] = np.nan
    keep = np.arange(len(w)) % rebal == 0
    w = w.where(pd.Series(keep, index=w.index), np.nan).ffill()
    return w

def bench():
    s = nifty[nifty.index >= px.index[260]]
    for lab, x in (("dev", s[s.index < HOLD]), ("holdout", s[s.index >= HOLD])):
        print(f"NIFTY buy & hold {lab}: ann {x.mean()*252:+.1%} vol {x.std()*np.sqrt(252):.1%}")
bench()
lr = np.log(px)
mom12 = lr.shift(21) - lr.shift(252); mom6 = lr.shift(21) - lr.shift(126); mom3 = lr.shift(5) - lr.shift(63)
rev5 = -(lr - lr.shift(5)); rev1 = -(lr - lr.shift(1))
vol60 = ret.rolling(60).std()
print("\n-- cross-sectional long/short (top vs bottom 10%)")
run(xs(mom12, rebal=21), "momentum 12-1m, monthly")
run(xs(mom6, rebal=21), "momentum 6-1m, monthly")
run(xs(mom3, rebal=5), "momentum 3m, weekly")
run(xs(rev5, rebal=5), "1-week reversal, weekly")
run(xs(rev1, rebal=1), "1-day reversal, daily")
run(xs(-vol60, rebal=21), "low volatility, monthly")
print("\n-- long-only top 10% (vs NIFTY)")
run(xs(mom12, long_only=True, rebal=21), "momentum 12-1m long-only")
run(xs(mom6, long_only=True, rebal=21), "momentum 6-1m long-only")
run(xs(-vol60, long_only=True, rebal=21), "low vol long-only")
run(xs(rev5, long_only=True, rebal=5), "1-week losers long-only")
print("\n-- time-series trend, each stock long or short, equal gross")
ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
t = np.sign(ma50 - ma200); w = t.div(t.abs().sum(axis=1), axis=0)
w = w.where(pd.Series(np.arange(len(w)) % 5 == 0, index=w.index)).ffill(); run(w, "50/200 MA trend L/S, weekly")
t = np.sign(px - ma200); w = t.div(t.abs().sum(axis=1), axis=0)
w = w.where(pd.Series(np.arange(len(w)) % 5 == 0, index=w.index)).ffill(); run(w, "price vs 200MA L/S, weekly")
print("\n-- breakout: close above 55-day high on 2x volume -> long 10 days; below 55-day low -> short")
h55 = hi.rolling(55).max().shift(1); l55 = lo.rolling(55).min().shift(1); rv = vol / vol.rolling(20).mean().shift(1)
up = ((px > h55) & (rv > 2)).astype(float); dn = ((px < l55) & (rv > 2)).astype(float)
sig = up.rolling(10).max().fillna(0) - dn.rolling(10).max().fillna(0)
w = sig.div(sig.abs().sum(axis=1).replace(0, np.nan), axis=0); run(w, "55d breakout L/S (both sides)")
w = up.rolling(10).max().fillna(0); w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0); run(w, "55d breakout long only")
