exec(open("research/futures.py").read().split("\nbench()")[0])
lr = np.log(px)
import itertools
for lb, top, reb in itertools.product([126, 189, 252], [0.1, 0.2, 0.3], [21]):
    sc = lr.shift(21) - lr.shift(lb)
    net = run(xs(sc, top=top, rebal=reb), f"mom {lb}d-1m top{int(top*100)}%")
sc = lr.shift(21) - lr.shift(252); net = run(xs(sc, top=0.2, rebal=21), "12-1 top20")
net = net[net.index >= px.index[273]]
print(net.groupby(net.index.year).apply(lambda s: f"{s.sum():+.1%}"))
print(net.groupby(net.index.to_period('Q')).sum().round(3).to_string())
