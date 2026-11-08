"""Multi-strategy desk: risk engine, condor strategy and the combined backtest (no network)."""
import datetime as dt
import os

import pandas as pd

from fno_alert.desk import backtest as dbt
from fno_alert.desk.live import Ledger
from fno_alert.desk.risk import RiskEngine, RiskLimits
from fno_alert.desk.strategies import Condor, CondorParams, Strangle
from fno_alert.intraday.backtest import History
from fno_alert.intraday.rules import Rules
from tests.test_intraday import make_history

D = dt.date(2025, 1, 6)


def loose_rules(**kw):
    return Rules(max_dte=0, hedged=False, trend="none", entry_time="11:30", vix_max=99, or_max_pct=5,
                 gap_max_pct=5, min_credit_pts=0.5, **kw)


def test_daily_limit_needs_two_checks_then_blocks_entries():
    L = RiskLimits(capital=500_000, daily_loss_pct=0.02, confirm_checks=2)
    r = RiskEngine(L, kill_file="")
    r.start_day(D, 500_000)
    assert r.check(dt.datetime(2025, 1, 6, 10), 489_000) == ""          # first bad check: wait
    assert "today" in r.check(dt.datetime(2025, 1, 6, 10, 1), 489_000)  # second: close all
    assert r.allow_entry("strangle", 1000, 0, 0, D).startswith("stopped for today")
    r.start_day(D + dt.timedelta(days=1), 489_000)
    assert r.allow_entry("strangle", 1000, 0, 0, D + dt.timedelta(days=1)) == ""


def test_monthly_pause_stays_until_resumed():
    L = RiskLimits(capital=500_000, daily_loss_pct=0, monthly_loss_pct=0.06, confirm_checks=1)
    r = RiskEngine(L, kill_file="")
    r.start_day(D, 500_000)
    assert "month" in r.check(dt.datetime(2025, 1, 6, 10), 469_000)
    r.start_day(dt.date(2025, 2, 3), 469_000)
    assert r.allow_entry("condor", 1000, 0, 0, dt.date(2025, 2, 3)).startswith("desk paused")
    r.resume()
    assert r.allow_entry("condor", 1000, 0, 0, dt.date(2025, 2, 3)) == ""


def test_entry_checks(tmp_path):
    kill = tmp_path / "KILL"
    L = RiskLimits(capital=500_000, max_trade_loss_pct={"strangle": 0.02}, max_margin_frac=0.8)
    r = RiskEngine(L, events={D: "Budget"}, kill_file=str(kill))
    assert r.allow_entry("strangle", 1000, 0, 0, D).startswith("event day")
    d2 = D + dt.timedelta(days=1)
    assert "above" in r.allow_entry("strangle", 10_001, 0, 0, d2)
    assert r.allow_entry("strangle", 9_000, 300_000, 150_000, d2) == "not enough margin left"
    assert r.allow_entry("strangle", 9_000, 100_000, 150_000, d2) == ""
    kill.write_text("stop")
    assert "kill switch" in r.allow_entry("strangle", 9_000, 0, 0, d2)
    assert "kill switch" in r.check(dt.datetime(2025, 1, 7, 10), 500_000)


def test_strategy_auto_pause():
    r = RiskEngine(RiskLimits(auto_pause_mult=1.5), strategy_dd={"condor": 10_000}, kill_file="")
    r.update_strategy(None, "condor", 5_000)
    r.update_strategy(None, "condor", -9_000)      # 14k down from the peak: under 15k
    assert "condor" not in r.s.paused_strategies
    r.update_strategy(None, "condor", -10_500)     # 15.5k down
    assert "condor" in r.s.paused_strategies
    assert r.allow_entry("condor", 1, 0, 0, D).startswith("strategy paused")


def test_condor_backtest_holds_to_the_session_before_expiry(tmp_path):
    sessions = make_history(str(tmp_path), days=17)
    h = History(str(tmp_path), "NIFTY")
    c = Condor(CondorParams(short_delta=0.15, wing=300), loose_rules(), 500_000)
    tr, eq, ref, _ = dbt.run(h, [c], RiskLimits(), {}, use_limits=False)
    assert len(tr) >= 2, ref
    for _, t in tr.iterrows():
        e = pd.Timestamp(t["expiry"]).date()
        before = [s for s in sessions if s < e][-1]
        assert t["exit_date"] == before and t["exit"] == "15:20" and t["reason"] == "time exit"
        assert t["entry"] >= "15:20"
        assert t["structure"] == "iron condor" and len(t["bought"].split(" / ")) == 2
        assert -t["pnl"] <= t["planned_max_loss"] + 1
    assert len(eq) == len([s for s in sessions if s >= h.days[0]])


def test_desk_runs_both_and_refuses_a_trade_over_its_limit(tmp_path):
    make_history(str(tmp_path), days=17)
    h = History(str(tmp_path), "NIFTY")
    r = loose_rules()
    strats = [Strangle(r), Condor(CondorParams(short_delta=0.15, wing=300), r, 500_000)]
    tr, eq, ref, _ = dbt.run(h, strats, RiskLimits(), {}, use_limits=True)
    assert set(tr["strategy"]) == {"strangle", "condor"}
    # never two positions at once: the condor opens after the strangle closes on expiry day
    tight = RiskLimits(max_trade_loss_pct={"strangle": 0.025, "condor": 0.001})
    tr2, _, ref2, _ = dbt.run(h, strats, tight, {}, use_limits=True)
    assert "condor" not in set(tr2["strategy"])
    assert ref2["reason"].str.contains("above").any()


def test_ledger_keeps_an_open_condor_overnight(tmp_path):
    make_history(str(tmp_path), days=12)
    h = History(str(tmp_path), "NIFTY")
    c = Condor(CondorParams(short_delta=0.15, wing=300), loose_rules(), 500_000)
    from fno_alert.intraday import engine
    pos = None
    for day in h.days:  # first session where the synthetic chain has every strike
        e = c.target_expiry(day, h.expiries)
        fr = dbt.Frames(h, day)
        if e is None or fr.get(e) is None:
            continue
        i = 365
        spot = float(h.idx.loc[str(day), "close"].iloc[i])
        snap = engine.Snapshot("NIFTY", dt.datetime.combine(day, dt.time(15, 20)), spot, spot, spot, spot + 50,
                               spot - 50, 14, e, h.dte(day, e), 75, fr.chain(e, i))
        pos, why = c.plan(snap, 500_000)
        if pos is not None:
            break
    assert pos is not None
    led = Ledger(str(tmp_path / "state"))
    led.save_open({"condor": pos})
    back = led.load_open()["condor"]
    assert back.legs == pos.legs and back.expiry == pos.expiry and back.credit == pos.credit
    assert os.path.exists(led.p["open.json"])


def test_desk_paper_runner_one_day(monkeypatch, tmp_path):
    """Live loop with a fake Upstox on an expiry day: the strangle trades and closes,
    the condor for next week opens at 15:20 and is kept overnight."""
    from fno_alert.desk import live
    from fno_alert.intraday import engine

    monkeypatch.chdir(tmp_path)
    day = dt.date(2026, 11, 10)                       # a Tuesday expiry
    exps = [day, day + dt.timedelta(days=7), day + dt.timedelta(days=14)]
    clock = {"t": dt.datetime.combine(day, dt.time(9, 30))}
    monkeypatch.setattr(live, "now_ist", lambda: clock["t"])
    monkeypatch.setattr(live.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + dt.timedelta(seconds=s)))
    spot = 23000.0

    def price(k, opt, e):
        t = clock["t"]
        T = engine.years_left(t, max((e - t.date()).days * 5 // 7, 0))
        return max(float(engine.bs_price(spot, k, T, 0.14, opt == "CE")), 0.05)

    class FakeUx:
        def get(self, path, params=None):
            return {}

        def option_chain(self, key, e):
            return [{"strike_price": k,
                     "call_options": {"instrument_key": f"K{e}_{k}CE",
                                      "market_data": {"ltp": price(k, "CE", e), "bid_price": 0, "ask_price": 0}},
                     "put_options": {"instrument_key": f"K{e}_{k}PE",
                                     "market_data": {"ltp": price(k, "PE", e), "bid_price": 0, "ask_price": 0}}}
                    for k in range(21000, 25050, 50)]

    monkeypatch.setattr(live, "Upstox", lambda: FakeUx())
    F = live.Feed
    monkeypatch.setattr(F, "contracts", lambda self, u: pd.DataFrame({"expiry": exps, "lot_size": [65.0] * 3}))
    monkeypatch.setattr(F, "minutes_today", lambda self, key: pd.DataFrame(
        {"ts": pd.date_range(f"{day} 09:15", periods=30, freq="1min"), "open": spot, "high": spot + 20,
         "low": spot - 20, "close": spot}))
    monkeypatch.setattr(F, "prev_close", lambda self, key, d: spot)

    def ltp(self, keys):
        out = {"NSE_INDEX|India VIX": 13.0}
        for k in keys:
            if k.startswith("K"):
                e, rest = k[1:].split("_")
                out[k] = price(float(rest[:-2]), rest[-2:], dt.date.fromisoformat(e))
        return out

    monkeypatch.setattr(F, "ltp", ltp)
    cfg = {"intraday": {"vix_max": 99, "max_dte": 0, "hedged": False, "entry_time": "11:30"},
           "desk": {"risk": {"block_event_days": False}}}
    live.run("paper", cfg, poll_sec=60)
    led = live.Ledger("state/desk/paper")
    tr = led.trades()
    assert list(tr["strategy"]) == ["strangle"], open(led.p["log.csv"]).read()
    still = led.load_open()
    assert list(still) == ["condor"] and still["condor"].expiry == exps[1]
    eq = pd.read_csv(led.p["equity.csv"])
    assert len(eq) == 1 and {"strangle_pnl", "condor_pnl"} <= set(eq.columns)
    import json
    st = json.load(open(led.p["status.json"]))
    assert st["open"][0]["strategy"] == "condor"


def test_daily_limit_can_leave_hedged_positions_alone():
    r = RiskEngine(RiskLimits(daily_closes="unhedged", confirm_checks=1), kill_file="")
    r.start_day(D, 500_000)
    why = r.check(dt.datetime(2025, 1, 6, 10), 489_000)
    assert why and not r.closes(why, hedged=True) and r.closes(why, hedged=False)
    r2 = RiskEngine(RiskLimits(daily_closes="all", confirm_checks=1), kill_file="")
    r2.start_day(D, 500_000)
    assert r2.closes(r2.check(dt.datetime(2025, 1, 6, 10), 489_000), hedged=True)


def test_fit_margin_cuts_lots_to_the_room_left():
    from fno_alert.desk.strategies import fit_margin
    from fno_alert.intraday import engine
    s = Strangle(loose_rules())
    legs = [engine.Leg(23500, "CE", -1, 10.0), engine.Leg(22500, "PE", -1, 10.0)]
    pos = engine.Position("NIFTY", D, "short strangle", legs, 4 * 65, 4, 20.0, 8000, dt.datetime(2025, 1, 6, 11, 30),
                          400, False, notes={"spot": 23000})
    per_lot = s.margin(pos) / 4
    assert fit_margin(s, pos, per_lot * 2.5) and pos.lots == 2 and pos.qty == 130 and pos.max_loss == 4000
    assert not fit_margin(s, pos, per_lot * 0.5)
