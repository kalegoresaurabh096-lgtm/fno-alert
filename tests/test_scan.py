"""End-to-end scan against a fake Upstox API (no network)."""
import datetime as dt

import numpy as np
import pandas as pd

from fno_alert import black76, scan
from fno_alert.upstox import IST


class FakeUpstox:
    def __init__(self, cache_dir=None, **_):
        self.calls = 0
        self.today = dt.datetime.now(IST).date()
        rng = np.random.default_rng(3)
        self.syms = {}
        for i, sym in enumerate(["AAA", "BBB", "CCC"]):
            days = pd.bdate_range(end=self.today - dt.timedelta(days=1), periods=420)
            c = 500 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.015, len(days))))
            if sym == "CCC":   # a tight squeeze followed by a big volume breakout today
                c[-40:] = c[-41] * (1 + rng.normal(0, 0.002, 40))
            self.syms[sym] = pd.DataFrame({"date": days, "open": c, "high": c * 1.01, "low": c * 0.99,
                                           "close": c, "volume": 1e6})
        e = self.today + dt.timedelta(days=20)
        self.exps = [e, e + dt.timedelta(days=35)]

    def instruments(self):
        rows = []
        for j, sym in enumerate(self.syms):
            ukey = f"NSE_EQ|ISIN{j}"
            for k, e in enumerate(self.exps):
                ms = int(pd.Timestamp(e, tz=IST).timestamp() * 1000) + 15 * 3600 * 1000
                rows.append({"segment": "NSE_FO", "underlying_type": "EQUITY", "underlying_symbol": sym,
                             "underlying_key": ukey, "instrument_type": "FUT", "instrument_key": f"NSE_FO|F{j}{k}",
                             "lot_size": 500, "expiry": ms, "name": sym})
                rows.append({"segment": "NSE_FO", "underlying_type": "EQUITY", "underlying_symbol": sym,
                             "underlying_key": ukey, "instrument_type": "CE", "instrument_key": f"NSE_FO|C{j}{k}",
                             "lot_size": 500, "expiry": ms, "name": sym})
        return pd.DataFrame(rows)

    fno_universe = staticmethod(scan.Upstox.fno_universe)

    def _sym_of_key(self, key):
        j = int(key.split("|")[1][-1]) if key.startswith("NSE_EQ") else int(key.split("|")[1][1])
        return list(self.syms)[j]

    def quotes(self, keys):
        out = {}
        for k in keys:
            if k.startswith("NSE_INDEX"):
                out[k] = {"instrument_token": k, "last_price": 20000.0 if "Nifty" in k else 13.0, "ohlc": {}}
                continue
            sym = self._sym_of_key(k)
            last = self.syms[sym]["close"].iloc[-1]
            move = 1.04 if sym == "CCC" else 1.0
            if k.startswith("NSE_EQ"):
                out[k] = {"instrument_token": k, "last_price": last * move, "volume": 4e6 if sym == "CCC" else 1e6,
                          "ohlc": {"open": last, "high": last * move * 1.005, "low": last * 0.995}}
            else:
                out[k] = {"instrument_token": k, "last_price": last * move * 1.005, "volume": 2000 * 500,
                          "oi": 1.2e6 if sym == "CCC" else 1e6}
        return out

    def daily_candles(self, key, start, end):
        if key.startswith("NSE_INDEX"):
            d = pd.bdate_range(start, end)
            return pd.DataFrame({"date": d, "open": 1, "high": 1, "low": 1,
                                 "close": np.linspace(19000, 20000, len(d)) if "Nifty" in key else 13.0,
                                 "volume": 0, "oi": 0})
        sym = self._sym_of_key(key)
        df = self.syms[sym]
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
        df["oi"] = 1e6 if key.startswith("NSE_FO") else 0
        return df

    def option_chain(self, ukey, expiry):
        sym = self._sym_of_key(ukey)
        S = self.syms[sym]["close"].iloc[-1] * (1.04 if sym == "CCC" else 1)
        T = (expiry - self.today).days / 365
        F = S * np.exp(0.065 * T)
        iv = 0.45 if sym == "CCC" else 0.25
        out = []
        for K in np.round(np.arange(0.8, 1.21, 0.025) * S, 0):
            leg = lambda cp: {"market_data": {"ltp": float(black76.price(F, K, T, iv, cp == "CE")),
                                              "volume": 50 * 500, "oi": 300 * 500, "prev_oi": 250 * 500},
                              "option_greeks": {"iv": iv * 100}}
            out.append({"expiry": expiry.isoformat(), "strike_price": float(K), "underlying_spot_price": S,
                        "call_options": leg("CE"), "put_options": leg("PE")})
        return out


def test_scan_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "Upstox", FakeUpstox)
    cfg = {"alerts": {"fallback_rule_threshold": 0.2, "max_per_email": 5}}
    out_html = tmp_path / "email.html"
    alerts = scan.run(cfg, mode="eod", dry_run=True, state_dir=str(tmp_path / "state"),
                      model_dir=str(tmp_path / "no_model"), out_html=str(out_html))
    syms = [a["symbol"] for a in alerts]
    assert "CCC" in syms
    a = [a for a in alerts if a["symbol"] == "CCC"][0]
    assert a["action"] in ("BUY CE", "BUY PE", "SELL PE", "SELL CE", "BUY CE + BUY PE", "NO TRADE")
    assert a["reasons"] and abs(a["atm_iv"] - 0.45) < 0.02
    assert out_html.exists() and "CCC" in out_html.read_text()


def test_repeat_runs_and_cooldown(tmp_path, monkeypatch):
    from fno_alert import emailer
    sent = []
    monkeypatch.setattr(scan, "Upstox", FakeUpstox)
    monkeypatch.setattr(emailer, "send", lambda *a, **k: sent.append(a[0]))
    monkeypatch.setenv("SMTP_USER", "x@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    cfg = {"alerts": {"fallback_rule_threshold": 0.2}, "backtest": {"min_turnover_cr": 0}}
    st = str(tmp_path / "state")
    for mode in ("intraday", "eod", "eod", "intraday"):
        scan.run(cfg, mode=mode, dry_run=False, state_dir=st, model_dir=str(tmp_path / "none"))
    # intraday + first EOD are sent; the repeated EOD and the later intraday are suppressed
    assert len(sent) == 2
    log = scan.State(st).load_table("alerts_log.csv")
    assert set(log["kind"]) == {"intraday", "eod"}


def test_dashboard_log(tmp_path, monkeypatch):
    from fno_alert import emailer
    monkeypatch.setattr(scan, "Upstox", FakeUpstox)
    monkeypatch.setattr(emailer, "send", lambda *a, **k: None)
    cfg = {"alerts": {"fallback_rule_threshold": 0.2}, "backtest": {"min_turnover_cr": 0}}
    st = tmp_path / "state"
    scan.run(cfg, mode="eod", dry_run=False, state_dir=str(st), model_dir=str(tmp_path / "none"))
    scan.run(cfg, mode="eod", dry_run=False, state_dir=str(st), model_dir=str(tmp_path / "none"))
    scores = pd.read_csv(st / "dash" / "scores.csv")
    runs = pd.read_csv(st / "dash" / "runs.csv")
    assert len(runs) == 2 and set(scores["symbol"]) == {"AAA", "BBB", "CCC"}
    assert scores.loc[scores["symbol"] == "CCC", "alerted"].iloc[0] == 1
    assert scores["score"].between(0, 1).all()


def test_no_email_settings(tmp_path, monkeypatch):
    from fno_alert import emailer
    sent = []
    monkeypatch.setattr(scan, "Upstox", FakeUpstox)
    monkeypatch.setattr(emailer, "send", lambda *a, **k: sent.append(a[0]))
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    cfg = {"alerts": {"fallback_rule_threshold": 0.2}, "backtest": {"min_turnover_cr": 0}}
    st = str(tmp_path / "state")
    alerts = scan.run(cfg, mode="eod", dry_run=False, state_dir=st, model_dir=str(tmp_path / "none"))
    assert alerts and not sent
    assert len(scan.State(st).load_table("alerts_log.csv")) == len(alerts)


def test_model_handles_infinite_features():
    import numpy as np
    import pandas as pd
    from fno_alert import model
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    df.loc[::17, "f2"] = np.inf
    df.loc[::23, "f1"] = -np.inf
    df["big"] = (rng.random(n) < 0.3).astype(float)
    df["dir"] = np.where(df["big"] == 1, rng.choice([-1.0, 1.0], n), 0.0)
    for kind in ("hgb", "logit"):
        m = model.fit(df, ["f1", "f2"], kind)
        p = m.predict(df)
        assert np.isfinite(p.to_numpy()).all()
