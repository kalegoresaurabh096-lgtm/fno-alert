import datetime as dt

import numpy as np
import pandas as pd

from fno_alert import momentum, scan
from tests.test_scan import FakeUpstox


def _closes(n=300):
    idx = pd.bdate_range(end="2026-10-30", periods=n)
    return {"UP": pd.Series(np.linspace(100, 200, n), idx), "FLAT": pd.Series(100.0, idx),
            "DOWN": pd.Series(np.linspace(200, 100, n), idx), "SHORT": pd.Series(100.0, idx[-100:])}


def test_scores_rank_by_12_1_return():
    r = momentum.Rules(min_turnover_cr=0)
    df = momentum.scores(_closes(), {}, r)
    assert list(df["symbol"]) == ["UP", "FLAT", "DOWN"]      # SHORT has too little history
    c = _closes()["UP"]
    assert abs(df.loc[0, "mom"] - (c.iloc[-22] / c.iloc[-253] - 1)) < 1e-12


def test_book_rebalance_fill_and_mark(tmp_path):
    r = momentum.Rules(top_n=2, capital=100_000, min_turnover_cr=0)
    b = momentum.Book(str(tmp_path), r)
    ranked = pd.DataFrame({"symbol": ["A", "B", "C"], "mom": [.5, .3, .1], "close": [100., 50., 10.], "rank": [1, 2, 3]})
    o = b.plan(ranked, dt.date(2026, 11, 2))
    assert set(o["symbol"]) == {"A", "B"} and set(o["side"]) == {"BUY"}
    b.fill(dt.date(2026, 11, 2), {"A": 100, "B": 50})        # same day: not filled
    assert b.hold.empty
    b.fill(dt.date(2026, 11, 3), {"A": 100, "B": 50})
    assert set(b.hold["symbol"]) == {"A", "B"} and b.cash < 1000
    eq = b.mark(dt.date(2026, 11, 3), {"A": 110, "B": 50}, 24000)
    assert eq > 100_000
    b.save()
    # next month: B drops out, C comes in
    b = momentum.Book(str(tmp_path), r)
    ranked2 = ranked.set_index("symbol").loc[["A", "C", "B"]].reset_index()
    o = b.plan(ranked2, dt.date(2026, 12, 1))
    assert {(x.symbol, x.side) for x in o.itertuples()} == {("B", "SELL"), ("C", "BUY")}
    b.fill(dt.date(2026, 12, 2), {"B": 60, "C": 10})
    assert set(b.hold["symbol"]) == {"A", "C"}
    assert (tmp_path / "dash" / "momentum.csv").exists()


def test_scan_runs_momentum(tmp_path, monkeypatch, capsys):
    from fno_alert import emailer
    sent = []
    monkeypatch.setattr(scan, "Upstox", FakeUpstox)
    monkeypatch.setattr(emailer, "send", lambda *a, **k: sent.append(a[0]))
    monkeypatch.setenv("SMTP_USER", "x@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    cfg = {"alerts": {"fallback_rule_threshold": 0.2}, "backtest": {"min_turnover_cr": 0},
           "momentum": {"top_n": 2, "min_turnover_cr": 0}}
    st = tmp_path / "state"
    scan.run(cfg, mode="eod", dry_run=False, state_dir=str(st), model_dir=str(tmp_path / "none"))
    out = capsys.readouterr().out
    assert "Momentum not updated" not in out and "momentum rebalance" in out
    assert any(s.startswith("Momentum picks") for s in sent)
    orders = pd.read_csv(st / "momentum" / "orders.csv")
    assert len(orders) == 2 and set(orders["side"]) == {"BUY"}
    # a second end-of-day run in the same month does not email again
    sent.clear()
    scan.run(cfg, mode="eod", dry_run=False, state_dir=str(st), model_dir=str(tmp_path / "none"))
    assert not any(s.startswith("Momentum picks") for s in sent)
