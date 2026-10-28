import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from fno_alert import model, panel as pn
p = pn.build_panel("data/processed", 10.0); p["date"] = pd.to_datetime(p["date"])
lab = pd.read_parquet("research/trade_labels.parquet"); lab["date"] = pd.to_datetime(lab["date"])
for c in ("ce_ret", "pe_ret", "strad_ret"): lab[c] = lab[c].clip(-1, 3)
p = p.dropna(subset=["big"]).merge(lab, on=["symbol", "date"], how="left")
feats = model.feature_list(p)
oos = model.walk_forward(p, feats, "hgb", verbose=False).merge(p, on=["date", "symbol"])
oos.to_parquet("research/oos_big.parquet", index=False)
ud = np.sort(oos.date.unique()); split = ud[int(len(ud) * .6)]
for name, d in (("val", oos[oos.date < split]), ("hold", oos[oos.date >= split])):
    d = d.dropna(subset=["strad_ret"]).copy()
    d["dec"] = d.groupby("date")["p_big"].rank(pct=True).mul(10).clip(upper=9.999).astype(int)
    print(name)
    print(d.groupby("dec").agg(n=("strad_ret", "size"), big=("big", "mean"), buy_win=("strad_ret", lambda s: (s > 0).mean()),
                               buy_avg=("strad_ret", "mean"), buy_med=("strad_ret", "median"), cost=("strad_cost_pct", "mean")).round(3).to_string())
