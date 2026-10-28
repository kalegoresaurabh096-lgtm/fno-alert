"""Walk-forward test of models that predict whether buying options pays,
instead of whether the stock moves a lot."""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from fno_alert import model, panel as pn  # noqa: E402
from fno_alert.labels import HORIZON  # noqa: E402

p = pn.build_panel("data/processed", 10.0)
p["date"] = pd.to_datetime(p["date"])
lab = pd.read_parquet("research/trade_labels.parquet")
lab["date"] = pd.to_datetime(lab["date"])
for _c in ("ce_ret", "pe_ret", "strad_ret"):
    lab[_c] = lab[_c].clip(-1, 3)
p = p.merge(lab, on=["symbol", "date"], how="left")
base_feats = model.feature_list(p)
extra = ["term_slope"]
p["iv_vs_strad"] = p["straddle_pct"] / (p["hv20"] * np.sqrt(30 / 252)) if "hv20" in p else np.nan
extra.append("iv_vs_strad")
feats = base_feats + [f for f in extra if f not in base_feats]
print(f"{len(p):,} rows, {len(feats)} features; straddle label available on {p['strad_ret'].notna().mean():.0%}")

TARGETS = {"strad": "strad_ret", "ce": "ce_ret", "pe": "pe_ret"}


def fit_pred(train, test, col, kind="hgb"):
    tr = train.dropna(subset=[col])
    X = model._finite(tr[feats].astype(float))
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    X = X.clip(lower=lo, upper=hi, axis=1)
    y = (tr[col] > 0).astype(int)
    m = model.make_hgb(0) if kind == "hgb" else model.make_logit()
    m.fit(X, y)
    Xt = model._clip(test[feats].astype(float), lo, hi)
    return m.predict_proba(Xt)[:, 1]


dates = np.sort(p["date"].unique())
preds = []
start = 250
while start < len(dates):
    end = min(start + 63, len(dates))
    train = p[p["date"] <= dates[start - HORIZON - 1]]
    test = p[(p["date"] >= dates[start]) & (p["date"] <= dates[end - 1])].copy()
    for t, col in TARGETS.items():
        test[f"q_{t}"] = fit_pred(train, test, col)
    preds.append(test)
    print(f"fold {pd.Timestamp(dates[start]).date()}..{pd.Timestamp(dates[end-1]).date()} train={len(train):,}", flush=True)
    start = end
oos = pd.concat(preds)
oos.to_parquet("research/oos_v2.parquet", index=False)

ud = np.sort(oos["date"].unique())
split = ud[int(len(ud) * 0.6)]
val = oos[oos["date"] < ud[int(len(ud) * 0.6) - HORIZON]]
hold = oos[oos["date"] >= split]
print("validation", val.date.min().date(), val.date.max().date(), " holdout", hold.date.min().date(), hold.date.max().date())

for t, col in TARGETS.items():
    for name, d in (("val", val), ("hold", hold)):
        d = d.dropna(subset=[col])
        auc = roc_auc_score(d[col] > 0, d[f"q_{t}"])
        print(f"{t:5s} {name:4s} AUC {auc:.3f}  base win {(d[col] > 0).mean():.1%}  base avg {d[col].mean():+.1%}")


def topk_stats(d, q, col, per_day=1):
    d = d.dropna(subset=[col])
    top = d.sort_values(q, ascending=False).groupby("date").head(per_day)
    return len(top), (top[col] > 0).mean(), top[col].mean(), top[col].median()


def thr_stats(d, q, col, th):
    d = d.dropna(subset=[col])
    s = d[d[q] >= th]
    return len(s), (s[col] > 0).mean() if len(s) else np.nan, s[col].mean() if len(s) else np.nan


print("\nTop-1 per day (straddle / best single leg):")
for name, d in (("val", val), ("hold", hold)):
    print(name, "strad", topk_stats(d, "q_strad", "strad_ret"))
    d2 = d.copy()
    d2["best_q"] = d2[["q_ce", "q_pe"]].max(axis=1)
    d2["best_ret"] = np.where(d2["q_ce"] >= d2["q_pe"], d2["ce_ret"], d2["pe_ret"])
    print(name, "leg  ", topk_stats(d2, "best_q", "best_ret"))

print("\nThresholds on straddle model:")
for th in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
    v = thr_stats(val, "q_strad", "strad_ret", th)
    h = thr_stats(hold, "q_strad", "strad_ret", th)
    print(f"th {th:.2f}  val n={v[0]:5d} win={v[1]:.1%} avg={v[2]:+.1%}   hold n={h[0]:5d} win={h[1]:.1%} avg={h[2]:+.1%}")
