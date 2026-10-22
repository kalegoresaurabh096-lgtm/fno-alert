"""Walk-forward backtest, threshold selection and report.

Protocol (designed to avoid fooling ourselves):
1. Every prediction is out-of-sample: models are retrained every quarter on
   past data only, with a 5-day embargo so no training label overlaps the
   test period.
2. The out-of-sample period is split in time. The alert threshold and the
   trade rules are chosen on the first part (validation) only.
3. The second part (holdout) is touched once, to report the final numbers.
4. Confidence intervals come from a block bootstrap over calendar months,
   because alerts on the same day are not independent.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from . import explain, model, strategy


def _safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    m = ~np.isnan(p) & ~np.isnan(y)
    if m.sum() < 50 or len(np.unique(y[m])) < 2:
        return np.nan
    return roc_auc_score(y[m], p[m])


def block_bootstrap(df: pd.DataFrame, stat, n: int = 500, seed: int = 0):
    """95% interval of stat(df) resampling whole calendar months."""
    rng = np.random.default_rng(seed)
    months = df["date"].dt.to_period("M")
    groups = {m: g for m, g in df.groupby(months)}
    keys = list(groups)
    if len(keys) < 3:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        s = pd.concat([groups[keys[i]] for i in pick])
        v = stat(s)
        if v == v:
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def alert_table(oos: pd.DataFrame, score_col: str, thresholds, max_per_day: int | None = None) -> pd.DataFrame:
    base = oos["big"].mean()
    n_days = oos["date"].nunique()
    rows = []
    for th in thresholds:
        a = oos[oos[score_col] >= th]
        if max_per_day:
            a = a.sort_values(score_col, ascending=False).groupby("date").head(max_per_day)
        if len(a) == 0:
            rows.append({"threshold": th, "alerts": 0})
            continue
        prec = a["big"].mean()
        rows.append({
            "threshold": th, "alerts": len(a), "alerts_per_day": len(a) / n_days,
            "days_with_alert": a["date"].nunique() / n_days,
            "hit_rate": prec, "false_alarm_rate": 1 - prec, "lift": prec / base if base else np.nan,
            "recall": a["big"].sum() / oos["big"].sum() if oos["big"].sum() else np.nan,
            "avg_max_move": np.maximum(a["max_up"], -a["max_dn"]).mean(),
        })
    return pd.DataFrame(rows)


def direction_accuracy(a: pd.DataFrame, conf: float) -> dict:
    tot = a["p_up"] + a["p_dn"]
    c = a["p_up"] / tot.replace(0, np.nan)
    called = a[(c >= conf) | (c <= 1 - conf)].copy()
    if called.empty:
        return {"called": 0}
    called["call"] = np.where(called["p_up"] / (called["p_up"] + called["p_dn"]) >= conf, 1, -1)
    big = called[called["big"] == 1]
    return {
        "called": len(called), "share_called": len(called) / len(a),
        "correct_when_big_move": float((big["call"] == big["dir"]).mean()) if len(big) else np.nan,
        "correct_5d_close": float((np.sign(called["end_ret"]) == called["call"]).mean()),
    }


def calibration(oos: pd.DataFrame, col: str = "p_big", bins=(0, .05, .1, .15, .2, .3, .4, .5, .7, 1.0)):
    cut = pd.cut(oos[col], bins, include_lowest=True)
    g = oos.groupby(cut, observed=True)
    return pd.DataFrame({"predicted": g[col].mean(), "actual": g["big"].mean(), "n": g.size()})


def choose_threshold(val: pd.DataFrame, col: str, min_alerts_per_day: float, target_lift: float,
                     max_per_day: int | None, max_alerts_per_day: float = 5.0):
    """Lowest threshold on the validation period whose lift >= target, giving
    between min and max alerts per day on average."""
    ths = np.round(np.arange(0.03, 0.90, 0.01), 2)
    t = alert_table(val, col, ths, max_per_day)
    ok = t[(t["alerts"] >= 30) & (t["lift"] >= target_lift) & (t["alerts_per_day"] >= min_alerts_per_day)
           & (t["alerts_per_day"] <= max_alerts_per_day)]
    if ok.empty:
        ok = t[t["alerts"] >= 30].sort_values("lift", ascending=False).head(1)
    return float(ok["threshold"].iloc[0]), t


def run(panel: pd.DataFrame, contracts: pd.DataFrame | None, out_dir: str = "reports",
        holdout_frac: float = 0.4, target_lift: float = 2.5, min_alerts_per_day: float = 0.3,
        max_per_day: int = 10, kinds=("hgb", "logit"), max_alerts_per_day: float = 5.0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    panel = panel.dropna(subset=["big"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    feats = model.feature_list(panel)
    print(f"Panel: {len(panel):,} stock-days, {panel['symbol'].nunique()} stocks, "
          f"{panel['date'].min().date()}..{panel['date'].max().date()}, {len(feats)} features", flush=True)
    res = {"features": feats, "n_rows": len(panel), "n_stocks": int(panel["symbol"].nunique()),
           "start": str(panel["date"].min().date()), "end": str(panel["date"].max().date())}

    oos_by_kind = {}
    for kind in kinds:
        print(f"Walk-forward ({kind}) ...", flush=True)
        p = model.walk_forward(panel, feats, kind)
        oos_by_kind[kind] = p
    oos = panel.merge(oos_by_kind["hgb"], on=["date", "symbol"], how="inner")
    if "logit" in oos_by_kind:
        lg = oos_by_kind["logit"].rename(columns={"p_big": "lg_big", "p_up": "lg_up", "p_dn": "lg_dn"})
        oos = oos.merge(lg, on=["date", "symbol"], how="left")
    oos["rule_score"] = explain.rule_scores(oos) / 100.0

    dates = np.sort(oos["date"].unique())
    split = dates[int(len(dates) * (1 - holdout_frac))]
    val, hold = oos[oos["date"] < split], oos[oos["date"] >= split]
    res["validation"] = f"{pd.Timestamp(dates[0]).date()}..{(pd.Timestamp(split) - pd.Timedelta(days=1)).date()}"
    res["holdout"] = f"{pd.Timestamp(split).date()}..{pd.Timestamp(dates[-1]).date()}"
    res["base_rate_val"] = float(val["big"].mean())
    res["base_rate_holdout"] = float(hold["big"].mean())

    res["auc"] = {}
    for name, col in (("model", "p_big"), ("logistic", "lg_big"), ("rule_points", "rule_score"),
                      ("iv_rank_only", "iv_rank"), ("bb_squeeze_only", "bb_width_rank")):
        if col in oos:
            sign = -1 if col == "bb_width_rank" else 1
            res["auc"][name] = {"validation": _safe_auc(val["big"], sign * val[col]),
                                "holdout": _safe_auc(hold["big"], sign * hold[col])}
    res["auc_by_year"] = {str(y): _safe_auc(g["big"], g["p_big"]) for y, g in oos.groupby(oos["date"].dt.year)}

    th, val_table = choose_threshold(val, "p_big", min_alerts_per_day, target_lift, max_per_day, max_alerts_per_day)
    res["threshold"] = th
    hold_table = alert_table(hold, "p_big", sorted(set([th] + list(np.round(np.arange(0.05, 0.71, 0.05), 2)))), max_per_day)
    res["holdout_at_threshold"] = hold_table[hold_table["threshold"] == th].iloc[0].to_dict()
    rule_th, _ = choose_threshold(val, "rule_score", min_alerts_per_day, target_lift, max_per_day, max_alerts_per_day)
    res["rule_baseline_holdout"] = alert_table(hold, "rule_score", [rule_th], max_per_day).iloc[0].to_dict()

    alerts_h = hold[hold["p_big"] >= th].sort_values("p_big", ascending=False).groupby("date").head(max_per_day)
    alerts_v = val[val["p_big"] >= th].sort_values("p_big", ascending=False).groupby("date").head(max_per_day)
    res["hit_rate_ci"] = block_bootstrap(alerts_h, lambda d: d["big"].mean())
    res["base_rate_ci"] = block_bootstrap(hold, lambda d: d["big"].mean())

    # Direction confidence chosen on validation, reported on holdout.
    best = None
    for conf in (0.55, 0.6, 0.62, 0.65, 0.7, 0.75):
        da = direction_accuracy(alerts_v, conf)
        if da.get("called", 0) >= 30 and (best is None or da["correct_when_big_move"] > best[1]["correct_when_big_move"] + 0.01):
            best = (conf, da)
    dir_conf = best[0] if best else 0.62
    res["direction_conf"] = dir_conf
    res["direction_holdout"] = direction_accuracy(alerts_h, dir_conf)
    res["calibration_holdout"] = calibration(hold).reset_index(drop=True).to_dict("records")

    # Options trade ideas on real option prices.
    if contracts is not None and len(contracts):
        book = strategy.ContractBook(contracts)
        tdays = np.sort(panel["date"].unique())
        trade_res = {}
        for name, al in (("validation", alerts_v), ("holdout", alerts_h)):
            rules = strategy.TradeRules(dir_conf=dir_conf)
            al = al.copy()
            al[["action", "dir_label", "why"]] = [list(strategy.suggest(r.p_up, r.p_dn, r.iv_rank, r.iv_hv, rules))
                                                  for r in al.itertuples()]
            tr = strategy.simulate(al, book, tdays, rules)
            tr.to_csv(os.path.join(out_dir, f"trades_{name}.csv"), index=False)
            trade_res[name] = strategy.summarise_trades(tr).reset_index().to_dict("records")
            # Same trades on random non-alert days of the same period, as a baseline.
            pool = (val if name == "validation" else hold)
            pool = pool[pool["p_big"] < th].sample(min(len(al) * 3, len(pool)), random_state=1).copy() if len(al) else pool.iloc[:0]
            if len(pool):
                pool[["action", "dir_label", "why"]] = [list(strategy.suggest(r.p_up, r.p_dn, r.iv_rank, r.iv_hv, rules))
                                                        for r in pool.itertuples()]
                tb = strategy.simulate(pool, book, tdays, rules)
                trade_res[name + "_random_days"] = strategy.summarise_trades(tb).reset_index().to_dict("records")
        res["trades"] = trade_res

    # Feature importance: which inputs move the model (permutation on holdout sample).
    try:
        from sklearn.inspection import permutation_importance
        fitted = model.fit(panel[panel["date"] < split], feats)
        s = hold.sample(min(20000, len(hold)), random_state=0)
        X = model._clip(s[feats].astype(float), fitted.lo, fitted.hi)
        pi = permutation_importance(fitted.models["big"], X, s["big"].astype(int), scoring="roc_auc",
                                    n_repeats=3, random_state=0)
        imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
        res["top_features"] = imp.head(20).round(4).to_dict()
    except Exception as e:  # importance is informative only
        res["top_features_error"] = str(e)

    val_table.to_csv(os.path.join(out_dir, "threshold_table_validation.csv"), index=False)
    hold_table.to_csv(os.path.join(out_dir, "threshold_table_holdout.csv"), index=False)
    oos.to_parquet(os.path.join(out_dir, "oos_predictions.parquet"), index=False)
    with open(os.path.join(out_dir, "backtest_summary.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=lambda x: x if not isinstance(x, (np.floating, np.integer)) else x.item())
    return res
