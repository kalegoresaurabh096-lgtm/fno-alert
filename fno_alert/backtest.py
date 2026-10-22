"""Walk-forward backtest, threshold selection and report.

Protocol (designed to avoid fooling ourselves):
1. Every prediction is out-of-sample: models are retrained every quarter on
   past data only, with a 5-day embargo so no training label overlaps the
   test period.
2. The out-of-sample period is split in time. The alert threshold, the
   direction rule and the trade rules are chosen on the first part
   (validation) only; the last 5 sessions of validation are dropped so its
   labels never overlap the holdout.
3. The holdout is used once, to report the final numbers.
4. Alerts are counted the way the scanner emails them: a stock is not
   re-alerted within 5 sessions unless its probability rises by 0.10. This
   stops one volatile episode from being counted as many "hits".
5. Labels start at the next day's open (you can't trade the close you
   used to compute the signal), and trades enter at the next day's open.
6. Confidence intervals come from a block bootstrap over calendar months.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from . import alerting, explain, model, strategy
from .labels import HORIZON


def _safe_auc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = ~np.isnan(p) & ~np.isnan(y)
    if m.sum() < 50 or len(np.unique(y[m])) < 2:
        return np.nan
    return roc_auc_score(y[m], p[m])


def block_bootstrap(df: pd.DataFrame, stat, n: int = 500, seed: int = 0):
    """95% interval of stat(df), resampling whole calendar months."""
    if df.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    months = pd.to_datetime(df["date"]).dt.to_period("M")
    groups = [g for _, g in df.groupby(months)]
    if len(groups) < 3:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = stat(s)
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def alert_stats(alerts: pd.DataFrame, universe: pd.DataFrame) -> dict:
    base = universe["big"].mean()
    n_days = universe["date"].nunique()
    if alerts.empty:
        return {"alerts": 0, "alerts_per_day": 0.0}
    prec = alerts["big"].mean()
    return {"alerts": len(alerts), "alerts_per_day": len(alerts) / n_days,
            "days_with_alert": alerts["date"].nunique() / n_days,
            "hit_rate": prec, "false_alarm_rate": 1 - prec, "lift": prec / base if base else np.nan,
            "recall": alerts["big"].sum() / universe["big"].sum() if universe["big"].sum() else np.nan,
            "avg_max_move": float(np.maximum(alerts["max_up"], -alerts["max_dn"]).mean()),
            "avg_overnight_gap_missed": float(alerts["gap_next"].abs().mean())}


def threshold_table(df: pd.DataFrame, col: str, thresholds, max_per_day: int | None) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        a = alerting.emit(df, th, col, max_per_day)
        rows.append({"threshold": th, **alert_stats(a, df)})
    return pd.DataFrame(rows)


def choose_threshold(val: pd.DataFrame, col: str, min_alerts_per_day: float, target_lift: float,
                     max_per_day: int | None, max_alerts_per_day: float = 5.0):
    """Lowest threshold on validation whose lift >= target with a usable alert rate."""
    ths = np.round(np.arange(0.03, 0.90, 0.01), 2)
    t = threshold_table(val, col, ths, max_per_day)
    ok = t[(t["alerts"] >= 30) & (t["lift"] >= target_lift) & (t["alerts_per_day"] >= min_alerts_per_day)
           & (t["alerts_per_day"] <= max_alerts_per_day)]
    if ok.empty:  # nothing meets the target: take the best lift with enough alerts, and say so
        ok = t[(t["alerts"] >= 30)].sort_values("lift", ascending=False).head(1)
    return float(ok["threshold"].iloc[0]), t


def direction_rule(alerts: pd.DataFrame, confs=(0.55, 0.6, 0.65, 0.7, 0.75), min_n: int = 40):
    """Pick a direction-confidence level only if its calls are significantly
    better than a coin flip on the validation alerts (one-sided binomial test,
    p < 0.05). Otherwise direction calls are disabled (returns 1.01)."""
    tried = []
    best = None
    for conf in confs:
        c = alerts["p_up"] / (alerts["p_up"] + alerts["p_dn"]).replace(0, np.nan)
        called = alerts[(c >= conf) | (c <= 1 - conf)]
        if len(called) < min_n:
            tried.append({"conf": conf, "n": len(called)}); continue
        call = np.where(called["p_up"] / (called["p_up"] + called["p_dn"]) >= conf, 1, -1)
        correct = int((np.sign(called["end_ret"]) == call).sum())
        p = binomtest(correct, len(called), 0.5, alternative="greater").pvalue
        tried.append({"conf": conf, "n": len(called), "accuracy": correct / len(called), "p_value": p})
        if p < 0.05 and (best is None or p < best[1]):
            best = (conf, p)
    return (best[0] if best else 1.01), tried


def direction_accuracy(a: pd.DataFrame, conf: float) -> dict:
    if conf > 1 or a.empty:
        return {"called": 0, "note": "direction calls disabled (no proven edge on validation)"}
    c = a["p_up"] / (a["p_up"] + a["p_dn"]).replace(0, np.nan)
    called = a[(c >= conf) | (c <= 1 - conf)].copy()
    if called.empty:
        return {"called": 0}
    called["call"] = np.where(called["p_up"] / (called["p_up"] + called["p_dn"]) >= conf, 1, -1)
    big = called[called["big"] == 1]
    return {"called": len(called), "share_called": len(called) / len(a),
            "correct_5d": float((np.sign(called["end_ret"]) == called["call"]).mean()),
            "correct_when_big_move": float((big["call"] == big["dir"]).mean()) if len(big) else np.nan}


def calibration(df: pd.DataFrame, col: str = "p_big", bins=(0, .05, .1, .15, .2, .3, .4, .5, .7, 1.0)):
    cut = pd.cut(df[col], bins, include_lowest=True)
    g = df.groupby(cut, observed=True)
    return pd.DataFrame({"predicted": g[col].mean(), "actual": g["big"].mean(), "n": g.size()})


def _with_actions(al: pd.DataFrame, rules: strategy.TradeRules) -> pd.DataFrame:
    al = al.copy()
    if al.empty:
        return al.assign(action=[], dir_label=[], why=[])
    sug = [strategy.suggest(r.p_up, r.p_dn, r.iv_rank, r.iv_hv, rules) for r in al.itertuples()]
    al["action"], al["dir_label"], al["why"] = zip(*sug)
    return al


def matched_random(alerts: pd.DataFrame, pool: pd.DataFrame, k: int = 3, seed: int = 1) -> pd.DataFrame:
    """For each alert, k random days of the same stock in the same period, with the same action:
    isolates whether the *timing* of the alert adds anything."""
    rng = np.random.default_rng(seed)
    by_sym = {s: g for s, g in pool.groupby("symbol")}
    rows = []
    for a in alerts.itertuples(index=False):
        g = by_sym.get(a.symbol)
        if g is None or len(g) < 2:
            continue
        g = g[g["date"] != a.date]
        pick = g.iloc[rng.integers(0, len(g), min(k, len(g)))].copy()
        pick["action"] = a.action
        rows.append(pick)
    return pd.concat(rows) if rows else pool.iloc[:0].assign(action=[])


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
        oos_by_kind[kind] = model.walk_forward(panel, feats, kind)
    oos = panel.merge(oos_by_kind["hgb"], on=["date", "symbol"], how="inner")
    if "logit" in oos_by_kind:
        lg = oos_by_kind["logit"].rename(columns={"p_big": "lg_big", "p_up": "lg_up", "p_dn": "lg_dn"})
        oos = oos.merge(lg, on=["date", "symbol"], how="left")
    oos["rule_score"] = explain.rule_scores(oos) / 100.0

    dates = np.sort(oos["date"].unique())
    split_i = int(len(dates) * (1 - holdout_frac))
    split = dates[split_i]
    val = oos[oos["date"] < dates[max(split_i - HORIZON, 1)]]     # embargo before the holdout
    hold = oos[oos["date"] >= split]
    res["validation"] = f"{pd.Timestamp(val['date'].min()).date()}..{pd.Timestamp(val['date'].max()).date()}"
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
    alerts_v = alerting.emit(val, th, "p_big", max_per_day)
    alerts_h = alerting.emit(hold, th, "p_big", max_per_day)
    res["validation_at_threshold"] = alert_stats(alerts_v, val)
    res["holdout_at_threshold"] = alert_stats(alerts_h, hold)
    res["holdout_raw_signal_days"] = alert_stats(hold[hold["p_big"] >= th], hold)  # without cooldown, for reference
    res["alert_rate_holdout"] = float((hold["p_big"] >= th).mean())
    hold_table = threshold_table(hold, "p_big", sorted(set([th] + list(np.round(np.arange(0.05, 0.71, 0.05), 2)))), max_per_day)
    rule_th, _ = choose_threshold(val, "rule_score", min_alerts_per_day, target_lift, max_per_day, max_alerts_per_day)
    res["rule_baseline_holdout"] = {"threshold": rule_th, **alert_stats(alerting.emit(hold, rule_th, "rule_score", max_per_day), hold)}
    res["hit_rate_ci"] = block_bootstrap(alerts_h, lambda d: d["big"].mean())
    res["base_rate_ci"] = block_bootstrap(hold, lambda d: d["big"].mean())

    dir_conf, tried = direction_rule(alerts_v)
    res["direction_conf"] = dir_conf
    res["direction_validation_tests"] = tried
    res["direction_holdout"] = direction_accuracy(alerts_h, dir_conf)
    res["calibration_holdout"] = calibration(hold).reset_index(drop=True).to_dict("records")

    # Options trade ideas on real option prices.
    if contracts is not None and len(contracts):
        fut = panel.set_index(["symbol", "date"])["fut_close"] if "fut_close" in panel else None
        book = strategy.ContractBook(contracts, fut)
        tdays = np.sort(panel["date"].unique())
        rules = strategy.TradeRules(dir_conf=dir_conf)
        trade_res = {}
        for name, al, pool in (("validation", alerts_v, val), ("holdout", alerts_h, hold)):
            al = _with_actions(al, rules)
            tr = strategy.simulate(al, book, tdays, rules)
            tr.to_csv(os.path.join(out_dir, f"trades_{name}.csv"), index=False)
            trade_res[name] = strategy.summarise_trades(tr).reset_index().to_dict("records")
            rnd = matched_random(al[al["action"] != "NO TRADE"], pool)
            if len(rnd):
                tb = strategy.simulate(rnd, book, tdays, rules)
                trade_res[name + "_same_stock_random_days"] = strategy.summarise_trades(tb).reset_index().to_dict("records")
        res["trades"] = trade_res
        res["trade_rules"] = rules.as_dict()

    # Feature importance (fit only on data whose labels end before the holdout).
    try:
        from sklearn.inspection import permutation_importance
        fitted = model.fit(panel[panel["date"] < dates[max(split_i - HORIZON, 1)]], feats)
        s = hold.sample(min(20000, len(hold)), random_state=0)
        X = model._clip(s[feats].astype(float), fitted.lo, fitted.hi)
        pi = permutation_importance(fitted.models["big"], X, s["big"].astype(int), scoring="roc_auc",
                                    n_repeats=3, random_state=0)
        res["top_features"] = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False).head(20).round(4).to_dict()
    except Exception as e:  # informative only
        res["top_features_error"] = str(e)

    val_table.to_csv(os.path.join(out_dir, "threshold_table_validation.csv"), index=False)
    hold_table.to_csv(os.path.join(out_dir, "threshold_table_holdout.csv"), index=False)
    oos.to_parquet(os.path.join(out_dir, "oos_predictions.parquet"), index=False)
    with open(os.path.join(out_dir, "backtest_summary.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=lambda x: x.item() if isinstance(x, (np.floating, np.integer)) else str(x))
    return res
