"""Human-readable backtest report (reports/BACKTEST.md)."""
from __future__ import annotations

import os

import pandas as pd


def _p(x, d=1):
    try:
        return f"{float(x) * 100:.{d}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def write(res: dict, out_dir: str) -> str:
    h = res.get("holdout_at_threshold", {})
    rb = res.get("rule_baseline_holdout", {})
    lines = [
        "# Backtest report", "",
        f"Data: {res['n_rows']:,} stock-days, {res['n_stocks']} F&O stocks, {res['start']} to {res['end']}.",
        f"Validation period (used to pick the threshold): {res['validation']}.",
        f"Holdout period (untouched until the end): {res['holdout']}.", "",
        "A **big move** = the stock closes at least 2x its normal 5-day move (and at least 4%) away from the "
        "**next day's open**, up or down, at some point in the next 5 trading days. The overnight gap after the "
        "signal is not counted, because you can only act at the next open.", "",
        "Alerts are counted the way they are emailed: no repeat for the same stock within 5 sessions unless its "
        "probability rises by 10 points.", "",
        "## Headline (holdout)", "",
        "| | Value |", "|---|---|",
        f"| Base rate (random F&O stock-day) | {_p(res['base_rate_holdout'])} (95% CI {_p(res['base_rate_ci'][0])} to {_p(res['base_rate_ci'][1])}) |",
        f"| Alert threshold (model probability) | {_f(res['threshold'], 2)} |",
        f"| Alerts per trading day | {_f(h.get('alerts_per_day'), 2)} |",
        f"| **Hit rate** (alert followed by a big move) | **{_p(h.get('hit_rate'))}** (95% CI {_p(res['hit_rate_ci'][0])} to {_p(res['hit_rate_ci'][1])}) |",
        f"| False alarm rate | {_p(h.get('false_alarm_rate'))} |",
        f"| Lift vs random | {_f(h.get('lift'), 2)}x |",
        f"| Share of all big moves caught (recall) | {_p(h.get('recall'))} |",
        f"| Simple points-score baseline hit rate | {_p(rb.get('hit_rate'))} ({_f(rb.get('alerts_per_day'), 2)} alerts/day) |",
        f"| Average overnight gap you could not trade | {_p(h.get('avg_overnight_gap_missed'))} |",
        "", "## Ranking quality (AUC, 0.5 = coin flip)", "",
        "| Signal | Validation | Holdout |", "|---|---|---|",
    ]
    for k, v in res.get("auc", {}).items():
        lines.append(f"| {k} | {_f(v.get('validation'))} | {_f(v.get('holdout'))} |")
    lines += ["", "AUC by year: " + ", ".join(f"{y}: {_f(v)}" for y, v in res.get("auc_by_year", {}).items()), ""]
    d = res.get("direction_holdout", {})
    lines += ["## Direction (UP / DOWN calls)", ""]
    if res.get("direction_conf", 1.01) > 1:
        lines += ["Direction calls are **disabled**: on the validation period no confidence level beat a coin flip "
                  "with statistical significance (binomial test p < 0.05), so trade ideas are direction-neutral.", ""]
    else:
        lines += [f"Direction is called when the model is at least {_p(res.get('direction_conf'), 0)} confident "
                  "(chosen on validation because it beat a coin flip, binomial p < 0.05).",
                  f"Holdout: share of alerts with a call {_p(d.get('share_called'))}; correct sign of the 5-day "
                  f"move {_p(d.get('correct_5d'))}; correct when a big move happened {_p(d.get('correct_when_big_move'))}.", ""]
    tests = res.get("direction_validation_tests", [])
    if tests:
        lines += ["| Confidence | Calls (validation) | Accuracy | p-value |", "|---|---|---|---|"]
        for t in tests:
            lines.append(f"| {_p(t['conf'], 0)} | {t['n']} | {_p(t.get('accuracy'))} | {_f(t.get('p_value'), 3)} |")
        lines.append("")
    cal = res.get("calibration_holdout", [])
    if cal:
        lines += ["## Calibration (holdout)", "", "| Predicted | Actual | Stock-days |", "|---|---|---|"]
        for c in cal:
            lines.append(f"| {_p(c['predicted'])} | {_p(c['actual'])} | {int(c['n']):,} |")
        lines.append("")
    tr = res.get("trades", {})
    if tr:
        lines += ["## Suggested trades on real option prices", "",
                  "Entry at the **next day's open** (only if that contract traded), exit after 5 sessions and always "
                  "before expiry, using an expiry at least 12 days away. Costs: 2% slippage per side (at least Rs 0.15), "
                  "brokerage, exchange and SEBI fees, GST, stamp duty and STT. Returns are on premium paid (buys) or on "
                  "~20% margin (sells). 'Same stock random days' = the same trade on random days of the same stocks.", ""]
        for name, rows in tr.items():
            if not rows:
                continue
            lines += [f"**{name.replace('_', ' ')}**", "",
                      "| Action | Trades | Not fillable | Win rate | Avg return | Median | Worst | Profit factor | Avg P&L per lot (Rs) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                lines.append(f"| {r['action']} | {int(r['trades'])} | {int(r.get('not_fillable', 0))} | {_p(r['win_rate'])} | "
                             f"{_p(r['avg_return'])} | {_p(r['median_return'])} | {_p(r.get('worst_return'))} | "
                             f"{_f(r['profit_factor'], 2)} | {r['avg_pnl_per_lot']:,.0f} |")
            lines.append("")
    tf = res.get("top_features")
    if tf:
        lines += ["## What drives the model (permutation importance, holdout AUC drop)", ""]
        lines += [f"- {k}: {v:.4f}" for k, v in tf.items()]
        lines.append("")
    path = os.path.join(out_dir, "BACKTEST.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path
