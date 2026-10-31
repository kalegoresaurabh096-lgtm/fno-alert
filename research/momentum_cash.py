exec(open("research/futures.py").read().split("\nbench()")[0])
COST = 0.0015
lr = np.log(px)
for top in (0.1, 0.15, 0.2):
    sc = lr.shift(21) - lr.shift(252)
    net = run(xs(sc, top=top, long_only=True, rebal=21), f"cash long-only 12-1 top{int(top*100)}%")
    s = net[net.index >= px.index[273]]; n = nifty.reindex(s.index)
    yr = pd.DataFrame({"strategy": s.groupby(s.index.year).apply(lambda x: (1+x).prod()-1), "nifty": n.groupby(n.index.year).apply(lambda x: (1+x).prod()-1)})
    print(yr.round(3).T.to_string())
    eq = (1+s).cumprod(); ne = (1+n.fillna(0)).cumprod()
    print(f"  total {eq.iloc[-1]-1:+.1%} vs nifty {ne.iloc[-1]-1:+.1%}, maxDD {(eq/eq.cummax()-1).min():.1%} vs {(ne/ne.cummax()-1).min():.1%}, months beating nifty {( (s.groupby(s.index.to_period('M')).sum()-n.groupby(n.index.to_period('M')).sum())>0).mean():.0%}")
    print("  holdings per side:", int((xs(sc, top=top, long_only=True)>0).sum(axis=1).iloc[-1]))
