import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fno_alert import black76, chain, features, indicators as ind, labels, prices, strategy
from fno_alert.upstox import chain_to_frames, candles_to_frame


def ohlc_from_close(c):
    c = pd.Series(c, index=pd.bdate_range("2024-01-01", periods=len(c)), dtype=float)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": 1e6})


def test_iv_roundtrip():
    F, K, T = 1000.0, np.array([900, 1000, 1100.0]), 30 / 365
    for sigma in (0.15, 0.35, 0.8):
        for is_call in (True, False):
            px = black76.price(F, K, T, sigma, is_call)
            iv = black76.implied_vol(px, F, K, T, is_call)
            np.testing.assert_allclose(iv, sigma, rtol=1e-4)


def test_iv_rejects_impossible_prices():
    iv = black76.implied_vol(np.array([0.0, 2000.0]), 1000, 1000, 0.1, True)
    assert np.isnan(iv).all()


def test_max_pain_simple():
    strikes = np.array([90, 100, 110.0])
    # Heavy call OI at 100 and put OI at 100 -> pain minimised at 100.
    assert chain.max_pain(strikes, np.array([0, 1000, 0.0]), np.array([0, 1000, 0.0])) == 100


def test_rsi_bounds_and_trend():
    up = ohlc_from_close(np.linspace(100, 200, 60))
    r = ind.rsi(up["close"]).dropna()
    assert (r > 90).all()
    down = ohlc_from_close(np.linspace(200, 100, 60))
    assert (ind.rsi(down["close"]).dropna() < 10).all()


def test_indicators_are_causal():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    full = features.technical_features(ohlc_from_close(c))
    cut = features.technical_features(ohlc_from_close(c[:250]))
    cols = [k for k in full.columns if k not in ("close",)]
    a = full.iloc[:250][cols]
    b = cut[cols]
    # Values up to day 250 must not change when later data is appended.
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_divergence_is_causal():
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    r = ind.rsi(c)
    full = ind.divergence(c, r)
    for n in (150, 220, 333):
        part = ind.divergence(c[:n], r[:n])
        assert (full[:n].to_numpy() == part.to_numpy()).all()


def test_labels_start_at_next_open():
    c = pd.Series([100, 101, 99, 100, 108, 100, 100, 100, 100, 100, 100.0])
    o = c.shift(1).fillna(100.0)
    s = pd.Series(0.005, index=c.index)  # tiny sigma -> threshold = 4% floor
    lab = labels.forward_labels(c, o, s, horizon=5)
    assert lab["big"].iloc[0] == 1 and lab["dir"].iloc[0] == 1
    assert lab["big"].iloc[5] == 0
    assert lab["big"].iloc[-5:].isna().all()   # unknown future
    # An overnight gap you could not trade is not a hit.
    c2 = pd.Series([100, 110, 110, 110, 110, 110, 110.0])
    o2 = pd.Series([100, 110, 110, 110, 110, 110, 110.0])
    lab2 = labels.forward_labels(c2, o2, pd.Series(0.005, index=c2.index), horizon=5)
    assert lab2["big"].iloc[0] == 0 and abs(lab2["gap_next"].iloc[0] - 0.10) < 1e-9


def test_alert_cooldown():
    from fno_alert import alerting
    d = pd.bdate_range("2024-01-01", periods=12)
    df = pd.DataFrame({"date": d, "symbol": "X", "p_big": [0.5] * 12})
    out = alerting.emit(df, 0.3)
    assert list(out["date"]) == [d[0], d[6]]       # repeats suppressed for 5 sessions
    df.loc[2, "p_big"] = 0.65                     # big enough jump re-alerts
    assert d[2] in set(alerting.emit(df, 0.3)["date"])


def test_real_crash_not_treated_as_split():
    c = np.r_[np.full(30, 1000.0), np.full(30, 760.0)]
    px = ohlc_from_close(c)
    px.iloc[30, px.columns.get_loc("open")] = 800.0     # -20% gap, -24% day
    adj = prices.adjust_splits(px)
    assert adj["corp_action"].sum() == 0
    assert adj["close"].iloc[29] == 1000.0


def test_split_needs_lot_change_when_lots_known():
    c = np.r_[np.full(30, 1000.0), np.full(30, 500.0)]
    px = ohlc_from_close(c)
    px.iloc[30, px.columns.get_loc("open")] = 500.0
    same_lot = pd.Series(250.0, index=px.index)
    assert prices.adjust_splits(px, same_lot)["corp_action"].sum() == 0
    doubled = pd.Series(np.r_[np.full(30, 250.0), np.full(30, 500.0)], index=px.index)
    assert prices.adjust_splits(px, doubled)["corp_action"].sum() == 1


def test_old_format_lot_from_gcd():
    from fno_alert import dataset
    d = dt.date(2023, 5, 2)
    rows = [{"date": d, "symbol": "X", "inst": "FUT", "expiry": d, "strike": np.nan, "open": 100, "high": 101,
             "low": 99, "close": 100, "settle": 100, "volume": 10, "oi": 250 * 400, "chg_oi": 250 * 3,
             "lot": np.nan, "underlying": np.nan, "turnover": 10 * 250 * 100 / 1e5, "fmt": "old"}]
    for k, m in ((100, 7), (110, 13), (90, 22)):
        rows.append({**rows[0], "inst": "CE", "strike": k, "oi": 250 * m, "turnover": np.nan})
    out = dataset.normalise_units(pd.DataFrame(rows))
    assert (out["lot"] == 250).all()
    assert out["oi"].iloc[0] == 250 * 400          # OI already in shares: unchanged


def test_futures_totals_have_no_expiry_artifact():
    from fno_alert.scan import fut_daily_from_contracts
    d = pd.bdate_range("2026-09-21", periods=6)
    e1, e2, e3 = pd.Timestamp("2026-09-24"), pd.Timestamp("2026-10-27"), pd.Timestamp("2026-11-24")
    rows = []
    for i, day in enumerate(d):
        for e, oi in ((e1, 1000), (e2, 2000 + 10 * i), (e3, 100)):
            if day <= e:
                rows.append({"date": day, "expiry": e, "oi": float(oi), "volume": 1.0, "close": 1.0})
    tot = fut_daily_from_contracts(pd.DataFrame(rows)).set_index("date")
    # The day after the near contract expires, the change must not show -1000.
    assert tot.loc[d[4], "fut_chg_oi"] == 10


def test_trade_enters_next_open_and_sell_pnl_sign():
    from fno_alert import strategy
    days = pd.bdate_range("2026-09-01", periods=12)
    exp = pd.Timestamp("2026-10-27")
    rows = []
    for i, day in enumerate(days):
        for K in (90.0, 100.0, 110.0):
            for opt in ("CE", "PE"):
                px = 10.0 if i <= 1 else 4.0       # premium collapses after entry
                rows.append({"date": day, "symbol": "X", "expiry": exp, "strike": K, "opt": opt,
                             "open": 9.0 if i == 1 else px, "close": px, "settle": px, "volume": 5, "oi": 100})
    book = strategy.ContractBook(pd.DataFrame(rows))
    al = pd.DataFrame([{"date": days[0], "symbol": "X", "action": "SELL PE", "F": 100.0, "atm_iv": 0.2, "lot": 100}])
    r = strategy.TradeRules(slippage=0.0, min_slippage=0.0, brokerage=0, exch_fee=0, sebi_fee=0, stamp_buy=0, stt_sell=0)
    tr = strategy.simulate(al, book, days, r)
    t = tr.iloc[0]
    assert t["entry_date"] == days[1] and t["status"] == "ok"
    assert abs(t["pnl_per_share"] - (9.0 - 4.0)) < 1e-9          # sold at next open 9, bought back at 4
    al2 = al.assign(action="BUY CE")
    t2 = strategy.simulate(al2, book, days, r).iloc[0]
    assert abs(t2["pnl_per_share"] - (4.0 - 9.0)) < 1e-9


def test_split_adjustment():
    c = np.r_[np.full(30, 1000.0), np.full(30, 500.0)]
    px = ohlc_from_close(c)
    px.iloc[30, px.columns.get_loc("open")] = 500.0
    adj = prices.adjust_splits(px)
    assert adj["corp_action"].sum() == 1
    assert np.allclose(adj["close"], 500.0)


def test_event_features_do_not_peek():
    dates = pd.bdate_range("2024-01-01", periods=40)
    meet = pd.DataFrame({"date": [dates[30]], "announced": [dates[25]]})
    f = features.event_features(dates, meet)
    assert (f["results_within7"].iloc[:25] == 0).all()     # not yet announced
    assert f["results_within7"].iloc[25] == 1
    assert f["days_to_results"].iloc[30] == 0


def _chain(F=1000.0, iv=0.3, day=dt.date(2026, 10, 1), exp=dt.date(2026, 10, 27)):
    T = (exp - day).days / 365
    rows = []
    for K in np.arange(800, 1201, 20.0):
        for opt in ("CE", "PE"):
            px = float(black76.price(F, K, T, iv, opt == "CE"))
            rows.append({"expiry": exp, "strike": K, "opt": opt, "close": px,
                         "oi": 1000.0 * (5 if (K == 1100 and opt == "CE") or (K == 900 and opt == "PE") else 1),
                         "chg_oi": 10.0, "volume": 5.0})
    fut = pd.DataFrame([{"expiry": exp, "close": F, "oi": 1e6, "chg_oi": 5e4, "volume": 100.0}])
    return pd.DataFrame(rows), fut, day


def test_snapshot_recovers_iv_and_walls():
    opts, fut, day = _chain()
    s = chain.snapshot(opts, fut, day)
    assert abs(s["atm_iv"] - 0.3) < 0.005
    assert s["call_wall"] == 1100 and s["put_wall"] == 900
    assert abs(s["skew"]) < 0.01
    assert s["straddle_pct"] == pytest.approx(0.8 * 0.3 * np.sqrt(26 / 365), rel=0.05)


def test_upstox_chain_units():
    lot = 500
    chain_js = [{"expiry": "2026-10-27", "strike_price": 1000, "underlying_spot_price": 998.0, "pcr": 1,
                 "call_options": {"market_data": {"ltp": 30, "volume": 5 * lot, "oi": 40 * lot, "prev_oi": 30 * lot}},
                 "put_options": {"market_data": {"ltp": 28, "volume": 7 * lot, "oi": 41 * lot, "prev_oi": 41 * lot}}}] * 12
    df, spot = chain_to_frames(chain_js, lot)
    assert spot == 998.0
    assert df["volume"].iloc[0] == 5          # shares -> contracts
    assert df["oi"].iloc[0] == 40 * lot       # already shares, kept


def test_candles_frame():
    df = candles_to_frame([["2026-10-20T00:00:00+05:30", 1, 2, 0.5, 1.5, 100, 7],
                           ["2026-10-21T00:00:00+05:30", 1.5, 2, 1, 1.8, 120, 9]])
    assert list(df["close"]) == [1.5, 1.8] and df["date"].iloc[1] == pd.Timestamp("2026-10-21")


def test_suggestions():
    r = strategy.TradeRules()
    assert strategy.suggest(0.3, 0.05, 0.3, 1.0, r)[0] == "BUY CE"
    assert strategy.suggest(0.05, 0.3, 0.3, 1.0, r)[0] == "BUY PE"
    assert strategy.suggest(0.3, 0.05, 0.9, 1.0, r)[0] == "SELL PE"
    assert strategy.suggest(0.05, 0.3, 0.9, 1.0, r)[0] == "SELL CE"
    assert strategy.suggest(0.2, 0.2, 0.3, 1.0, r)[0] == "BUY CE + BUY PE"
    assert strategy.suggest(0.2, 0.2, 0.9, 1.0, r)[0] == "NO TRADE"
