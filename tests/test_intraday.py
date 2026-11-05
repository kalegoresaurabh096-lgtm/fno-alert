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


def test_paper_runner_enters_and_exits(monkeypatch, tmp_path):
    """The live loop with a fake Upstox: one entry at 09:45, exit by the rules, state saved."""
    from fno_alert.intraday import live

    monkeypatch.chdir(tmp_path)
    clock = {"t": dt.datetime(2026, 11, 9, 9, 40)}  # a Monday; expiry Tuesday -> dte 1
    monkeypatch.setattr(live, "now_ist", lambda: clock["t"])
    monkeypatch.setattr(live.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + dt.timedelta(seconds=s)))
    expiry = dt.date(2026, 11, 10)
    spot = 23000.0

    def price(k, opt, t):
        T = engine.years_left(t, 1)
        decay = 1.0 - 0.6 * (t - dt.datetime(2026, 11, 9, 9, 45)).total_seconds() / (6 * 3600)
        return float(engine.bs_price(spot, k, T, 0.14, opt == "CE")) * max(decay, 0.2)

    class FakeUx:
        def get(self, path, params=None):
            return {}

        def option_chain(self, key, e):
            return [{"strike_price": k,
                     "call_options": {"instrument_key": f"K{k}CE",
                                      "market_data": {"ltp": price(k, "CE", clock["t"]), "bid_price": 0, "ask_price": 0}},
                     "put_options": {"instrument_key": f"K{k}PE",
                                     "market_data": {"ltp": price(k, "PE", clock["t"]), "bid_price": 0, "ask_price": 0}}}
                    for k in range(22000, 24050, 50)]

    monkeypatch.setattr(live, "Upstox", lambda: FakeUx())
    monkeypatch.setattr(live.Feed, "minutes_today", lambda self, key: pd.DataFrame(
        {"ts": pd.date_range("2026-11-09 09:15", periods=30, freq="1min"), "open": spot, "high": spot + 20,
         "low": spot - 20, "close": spot}))
    monkeypatch.setattr(live.Feed, "nearest_expiry", lambda self, u, d: (expiry, 65.0))
    monkeypatch.setattr(live.Feed, "prev_close", lambda self, key, d: spot)

    def ltp(self, keys):
        out = {"NSE_INDEX|India VIX": 13.0}
        for k in keys:
            if k.startswith("K"):
                out[k] = price(float(k[1:-2]), k[-2:], clock["t"])
        return out

    monkeypatch.setattr(live.Feed, "ltp", ltp)
    live.run("paper", {"intraday": {"vix_max": 99}}, poll_sec=60, dry=True, underlyings=("NIFTY",))
    book = live.Book("state/intraday/paper")
    tr = book.trades()
    assert len(tr) == 1, book.days()
    assert tr.iloc[0]["reason"] in ("profit target", "time exit")
    assert tr.iloc[0]["structure"] == "iron condor"
    assert book.open_positions() == {}
