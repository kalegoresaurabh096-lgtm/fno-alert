import datetime as dt

import pandas as pd

from fno_alert import paper, strategy

D = dt.date


def _alert(sym="IEX"):
    exp = D(2026, 11, 23)
    return {"symbol": sym, "action": "BUY CE + BUY PE", "p_big": 0.31, "lot": 4350,
            "legs": [("BUY CE", 106.0, exp, 4.0), ("BUY PE", 106.0, exp, 3.62)]}


def _instr():
    ms = int(pd.Timestamp("2026-11-23 15:30", tz="Asia/Kolkata").timestamp() * 1000)
    return pd.DataFrame([
        {"segment": "NSE_FO", "underlying_symbol": "IEX", "instrument_type": t, "expiry": ms,
         "strike_price": 106.0, "instrument_key": f"NSE_FO|{t}106"} for t in ("CE", "PE")])


def test_paper_trade_lifecycle(tmp_path):
    now = dt.datetime(2026, 10, 29, 16, 20)
    book = paper.add_alerts(paper.load(str(tmp_path)), [_alert(), _alert()], _instr(), now)
    assert len(book) == 1 and book["status"].iloc[0] == paper.PENDING   # one trade per stock
    assert book["ce_key"].iloc[0] == "NSE_FO|CE106"

    sessions = [D(2026, 10, 27), D(2026, 10, 28), D(2026, 10, 29)]
    opens = {"NSE_FO|CE106": 1.70, "NSE_FO|PE106": 5.50}
    q_open = lambda keys: {k: {"ohlc": {"open": opens[k]}, "last_price": opens[k] + 0.1} for k in keys}
    no_candle = lambda k, d: (float("nan"), float("nan"))

    # Same day: nothing happens.
    book = paper.update(book, D(2026, 10, 29), sessions, True, q_open, no_candle)
    assert book["status"].iloc[0] == paper.PENDING
    # Next session, morning scan: entry at the open.
    sessions.append(D(2026, 10, 30))
    book = paper.update(book, D(2026, 10, 30), sessions, False, q_open, no_candle)
    assert book["status"].iloc[0] == paper.OPEN and book["entry_ce"].iloc[0] == 1.70
    # 4 sessions later, before the close: still open.
    for d in (D(2026, 11, 2), D(2026, 11, 3), D(2026, 11, 4), D(2026, 11, 5)):
        sessions.append(d)
    closes = {"NSE_FO|CE106": 0.5, "NSE_FO|PE106": 10.0}
    q_close = lambda keys: {k: {"ohlc": {"open": 1}, "last_price": closes[k]} for k in keys}
    book = paper.update(book, D(2026, 11, 5), sessions, False, q_close, no_candle)
    assert book["status"].iloc[0] == paper.OPEN
    # 5th session after the alert, after the close: exit.
    book = paper.update(book, D(2026, 11, 5), sessions, True, q_close, no_candle)
    r = book.iloc[0]
    assert r["status"] == paper.CLOSED and r["exit_date"] == "2026-11-05"
    R = strategy.TradeRules()
    exp_pnl = sum((x - strategy.side_cost(x, 4350, False, R)) - (e + strategy.side_cost(e, 4350, True, R))
                  for e, x in ((1.70, 0.5), (5.50, 10.0))) * 4350
    assert abs(r["pnl_per_lot"] - round(exp_pnl, 2)) < 0.01 and r["ret"] > 0
    paper.save(str(tmp_path), book)
    assert (tmp_path / "dash" / "paper.csv").exists()
    assert paper.load(str(tmp_path))["status"].iloc[0] == paper.CLOSED


def test_missed_entry_uses_candle_and_unknown_contract_is_skipped(tmp_path):
    now = dt.datetime(2026, 10, 29, 16, 20)
    book = paper.add_alerts(paper.load(str(tmp_path)), [_alert(), _alert("ABC")], _instr(), now)
    book = paper.resolve_keys(book, _instr())
    sessions = [D(2026, 10, 29), D(2026, 10, 30), D(2026, 11, 2)]
    candle = lambda k, d: (2.0, 2.1) if d == D(2026, 10, 30) else (float("nan"), float("nan"))
    book = paper.update(book, D(2026, 11, 2), sessions, False, lambda keys: {}, candle)
    iex, abc = book.set_index("symbol").loc["IEX"], book.set_index("symbol").loc["ABC"]
    assert iex["status"] == paper.OPEN and iex["entry_date"] == "2026-10-30" and iex["entry_pe"] == 2.0
    assert abc["status"] == paper.SKIPPED


def test_position_sizing(tmp_path):
    now = dt.datetime(2026, 10, 29, 16, 20)
    instr = _instr()
    book = paper.add_alerts(paper.load(str(tmp_path)), [_alert()], instr, now)
    sessions = [D(2026, 10, 29), D(2026, 10, 30)]
    q = lambda keys: {k: {"ohlc": {"open": 4.0}, "last_price": 4.5} for k in keys}
    nc = lambda k, d: (float("nan"), float("nan"))
    # one lot costs 8 * 4350 = Rs 34,800: fits a Rs 50k cap once, not a Rs 25k cap
    s = paper.Sizing(capital=500_000, max_trade_pct=0.10, max_open_pct=0.5)
    b = paper.update(book, D(2026, 10, 30), sessions, False, q, nc, sizing=s)
    assert b["lots"].iloc[0] == 1 and b["capital_used"].iloc[0] == 34800
    b = paper.mark_to_market(b, q(["NSE_FO|CE106", "NSE_FO|PE106"]))
    assert b["pnl_total"].iloc[0] == b["pnl_per_lot"].iloc[0]
    eq = paper.equity(b, s)
    assert eq["in_use"] == 34800 and eq["equity"] == 500_000
    small = paper.Sizing(capital=500_000, max_trade_pct=0.05)
    b2 = paper.update(book, D(2026, 10, 30), sessions, False, q, nc, sizing=small)
    assert b2["status"].iloc[0] == paper.SKIPPED and "cap" in b2["note"].iloc[0]
    assert paper.Sizing.from_cfg({"paper": {"capital": 1e6}}).capital == 1e6
