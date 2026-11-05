"""Intraday selling engine on synthetic 1-minute data (no network)."""
import datetime as dt
import os
import tempfile

import numpy as np
import pandas as pd

from fno_alert.intraday import backtest, engine
from fno_alert.intraday.rules import Rules


def make_history(root, days=12, vol=0.14, seed=1, spike_day=None):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "NIFTY")
    os.makedirs(os.path.join(d, "opt"), exist_ok=True)
    sessions = [x.date() for x in pd.bdate_range("2025-01-06", periods=days)]
    mins = pd.date_range("09:15", "15:29", freq="1min").time
    spot = 23000.0
    rows = []
    for s in sessions:
        for m in mins:
            step = rng.normal(0, vol / np.sqrt(252 * 375)) * spot
            if spike_day and s == spike_day and m == dt.time(11, 0):
                step = 0.02 * spot
            o = spot
            spot += step
            rows.append((dt.datetime.combine(s, m), o, max(o, spot), min(o, spot), spot, 14.0))
    idx = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vix"])
    idx.to_parquet(os.path.join(d, "index_1m.parquet"))
    expiries = [s for s in sessions if s.weekday() == 1]
    for e in expiries:
        win = idx[(idx["ts"].dt.date <= e) & (idx["ts"].dt.date > e - dt.timedelta(days=8))]
        win = win[win["ts"].dt.minute % 1 == 0]
        out = []
        dte_of = {s: sum(1 for x in sessions if s < x <= e) for s in sessions}
        for k in range(22000, 24050, 50):
            for opt in ("CE", "PE"):
                T = np.array([engine.years_left(t.to_pydatetime(), dte_of[t.date()]) for t in win["ts"]])
                px = engine.bs_price(win["close"].values, float(k), T, vol, opt == "CE")
                px = np.maximum(px, 0.05)
                out.append(pd.DataFrame({"ts": win["ts"].values, "open": px, "high": px * 1.01, "low": px * 0.99,
                                         "close": px, "volume": 100, "oi": 1000, "strike": float(k), "opt": opt,
                                         "lot": 75.0, "expiry": pd.Timestamp(e)}))
        pd.concat(out).to_parquet(os.path.join(d, "opt", f"{e}.parquet"))
    return sessions


def test_backtest_runs_and_respects_risk():
    with tempfile.TemporaryDirectory() as tmp:
        make_history(tmp)
        r = Rules(max_dte=1, vix_max=99, or_max_pct=5, gap_max_pct=5, min_credit_frac=0.01, min_credit_pts=0.5)
        tr, log = backtest.run([backtest.History(tmp, "NIFTY")], r, {})
        assert len(log) > 0
        assert len(tr) > 0, log
        # hedged: no trade may lose more than its planned worst case
        assert (tr["loss_vs_plan"] <= 1.0 + 1e-9).all()
        assert (tr["planned_max_loss"] <= r.capital * r.risk_pct * 1.05).all()
        assert set(tr["exit"].map(lambda s: s <= "15:29"))
        s = backtest.summary(tr, r)
        assert s["trades"] == len(tr)


def test_unhedged_stop_can_slip_past_plan():
    with tempfile.TemporaryDirectory() as tmp:
        sessions = make_history(tmp, spike_day=None)
        spike = [s for s in sessions if s.weekday() == 1][0]
        make_history(tmp, spike_day=spike)
        r = Rules(hedged=False, trend="none", max_dte=1, vix_max=99, or_max_pct=5, gap_max_pct=5,
                  min_credit_pts=0.5)
        tr, _ = backtest.run([backtest.History(tmp, "NIFTY")], r, {}, start=spike, end=spike)
        assert len(tr) == 1
        assert tr.iloc[0]["reason"] == "stop-loss"
        assert tr.iloc[0]["loss_vs_plan"] > 1.0  # the spike filled the stop worse than planned


def test_event_day_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        sessions = make_history(tmp)
        e = [s for s in sessions if s.weekday() == 1][0]
        tr, log = backtest.run([backtest.History(tmp, "NIFTY")], Rules(vix_max=99), {e: "RBI policy"}, start=e, end=e)
        assert tr.empty
        assert log.iloc[0]["reason"] == "event: RBI policy"
