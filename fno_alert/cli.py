"""Command line entry point:  python -m fno_alert <command>

  build-data   raw downloads (data/raw) -> research tables (data/processed)
  backtest     walk-forward backtest + report (reports/)
  train        backtest, then fit the final model on all history (models/)
  scan         live scan via Upstox and email alerts
  test-email   send a sample email to check SMTP settings
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def load_config(path: str = "config.toml") -> dict:
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    return {}


def load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_build(a, cfg):
    from . import dataset
    dataset.build_all(a.raw, a.processed, a.workers)


def _backtest(a, cfg):
    import pandas as pd
    from . import backtest, panel
    bc = cfg.get("backtest", {})
    p = panel.build_panel(a.processed, bc.get("min_turnover_cr", 10.0))
    con_path = os.path.join(a.processed, "fo_contracts.parquet")
    con = pd.read_parquet(con_path) if os.path.exists(con_path) else None
    res = backtest.run(p, con, a.reports, holdout_frac=bc.get("holdout_frac", 0.4),
                       target_lift=bc.get("target_lift", 2.5),
                       min_alerts_per_day=bc.get("min_alerts_per_day", 0.3),
                       max_per_day=cfg.get("alerts", {}).get("max_per_email", 10),
                       max_alerts_per_day=bc.get("max_alerts_per_day", 5.0))
    from . import report
    report.write(res, a.reports)
    return p, res


def cmd_backtest(a, cfg):
    _, res = _backtest(a, cfg)
    print(json.dumps({k: res[k] for k in ("auc", "threshold", "holdout_at_threshold") if k in res},
                     indent=2, default=str))


def cmd_train(a, cfg):
    from . import model, strategy
    p, res = _backtest(a, cfg)
    p = p.dropna(subset=["big"])
    fitted = model.fit(p, res["features"])
    os.makedirs(a.models, exist_ok=True)
    fitted.save(os.path.join(a.models, "model.pkl"))
    h = res["holdout_at_threshold"]
    meta = {"threshold": res["threshold"], "features": res["features"], "trained_on": f"{res['start']}..{res['end']}",
            "holdout": res["holdout"], "holdout_hit_rate": h.get("hit_rate"), "base_rate_holdout": res["base_rate_holdout"],
            "holdout_lift": h.get("lift"), "direction_conf": res["direction_conf"],
            "trade_rules": strategy.TradeRules(dir_conf=res["direction_conf"]).as_dict()}
    json.dump(meta, open(os.path.join(a.models, "model_meta.json"), "w"), indent=2, default=str)
    print("Saved model to", a.models)


def cmd_scan(a, cfg):
    from . import scan
    only = [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None
    scan.run(cfg, a.mode, a.dry_run, only, a.state, a.models, a.save_html)


def cmd_test_email(a, cfg):
    from . import emailer
    sample = [{"symbol": "SAMPLE", "action": "BUY CE", "direction": "UP", "p_big": 0.31, "spot": 1234.5,
               "ret1": 0.021, "trade_text": "BUY CE 1240 27 Oct @ about Rs 32.10 (cost about Rs 16,050 per lot of 500)",
               "trade_why": "Bullish and options are not expensive", "atm_iv": 0.28, "iv_rank": 0.35,
               "straddle_pct": 0.055, "expiry": "2026-10-27", "put_wall": 1200, "call_wall": 1300,
               "max_pain": 1220, "pcr_oi": 1.1, "fut_oi_chg": 0.07, "buildup": "Long build-up",
               "reasons": ["This is a test email from your F&O alert tool"]}]
    s, t, h = emailer.render(sample, {"base_rate": 0.1, "hit_rate": 0.3, "holdout_period": "test"}, "TEST")
    emailer.send("[TEST] " + s, t, h)
    print("Sent test email to", os.environ.get("ALERT_TO", os.environ.get("SMTP_USER")))


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser(prog="fno_alert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-data", "backtest", "train"):
        s = sub.add_parser(name)
        s.add_argument("--raw", default="data/raw")
        s.add_argument("--processed", default="data/processed")
        s.add_argument("--reports", default="reports")
        s.add_argument("--models", default="models")
        s.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    s = sub.add_parser("scan")
    s.add_argument("--mode", choices=["auto", "intraday", "eod"], default="auto")
    s.add_argument("--dry-run", action="store_true", help="print alerts, do not email or save state")
    s.add_argument("--symbols", help="comma-separated subset, e.g. RELIANCE,TCS")
    s.add_argument("--state", default="state")
    s.add_argument("--models", default="models")
    s.add_argument("--save-html", help="also write the email HTML to this file")
    sub.add_parser("test-email")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    {"build-data": cmd_build, "backtest": cmd_backtest, "train": cmd_train, "scan": cmd_scan,
     "test-email": cmd_test_email}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
