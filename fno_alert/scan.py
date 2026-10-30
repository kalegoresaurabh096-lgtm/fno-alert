"""Live scan: pull Upstox data, score every F&O stock, email the alerts.

Live features are built with the same code and definitions as the backtest
panel. Differences that remain are listed in the email:
* intraday runs use part-day data (volumes projected with a typical
  intraday volume curve), so they are early estimates, not backtested;
* the 15:45 IST end-of-day run is the backtested signal (entry next open).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import traceback

import numpy as np
import pandas as pd

from . import dashlog
from . import alerting, chain, emailer, explain, features as feat, panel, paper, prices, strategy
from .model import Fitted
from .upstox import IST, Upstox, chain_to_frames

INDEX_KEYS = {"nifty": "NSE_INDEX|Nifty 50", "vix": "NSE_INDEX|India VIX"}
# Typical share of a day's volume traded by each time (IST), for intraday projection.
VOLUME_CURVE = [(9.25, 0.0), (10.0, 0.22), (11.0, 0.38), (12.0, 0.50), (13.0, 0.60),
                (14.0, 0.72), (15.0, 0.88), (15.5, 1.0)]
PX_COLS = ["open", "high", "low", "close", "volume"]


def session_fraction(now: dt.datetime) -> float:
    h = now.hour + now.minute / 60
    xs, ys = zip(*VOLUME_CURVE)
    return float(np.clip(np.interp(h, xs, ys), 0.05, 1.0))


def prev_weekday(d: dt.date) -> dt.date:
    d -= dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


class State:
    """Everything the scanner remembers between runs (committed to the repo by the workflow)."""

    def __init__(self, root: str = "state"):
        self.root = root
        for d in ("ohlcv", "cache"):
            os.makedirs(os.path.join(root, d), exist_ok=True)

    def p(self, *a):
        return os.path.join(self.root, *a)

    def load_px(self, sym: str) -> pd.DataFrame:
        f = self.p("ohlcv", f"{sym}.csv")
        if not os.path.exists(f):
            return pd.DataFrame(columns=PX_COLS, index=pd.DatetimeIndex([], name="date"))
        return pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()

    def save_px(self, sym: str, px: pd.DataFrame):
        px[PX_COLS].to_csv(self.p("ohlcv", f"{sym}.csv"), index_label="date")

    def load_table(self, name: str) -> pd.DataFrame:
        f = self.p(name)
        if not os.path.exists(f):
            return pd.DataFrame()
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        for c in ("date", "expiry", "time"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
        return df

    def save_table(self, name: str, df: pd.DataFrame):
        f = self.p(name)
        df.to_parquet(f, index=False) if f.endswith(".parquet") else df.to_csv(f, index=False)


def merge_bars(px: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new is None or new.empty:
        return px
    new = new.set_index("date")[PX_COLS] if "date" in new.columns else new[PX_COLS]
    out = pd.concat([px[~px.index.isin(new.index)][PX_COLS], new]).sort_index()
    return out[~out.index.duplicated(keep="last")]


def update_history(ux: Upstox, st: State, sym: str, key: str, today: dt.date, days: int = 600) -> pd.DataFrame:
    """Daily bars up to yesterday, back-filling any missed sessions."""
    px = st.load_px(sym)
    target = prev_weekday(today)
    last = px.index.max().date() if len(px) else None
    if last is None or last < target:
        start = last + dt.timedelta(days=1) if last else today - dt.timedelta(days=days)
        new = ux.daily_candles(key, start, target)
        if len(new):
            px = merge_bars(px, new)
            st.save_px(sym, px)
    return px


def quote_bar(q: dict, today: dt.date) -> pd.DataFrame:
    o = (q or {}).get("ohlc") or {}
    if not q or not q.get("last_price"):
        return pd.DataFrame()
    return pd.DataFrame([{"date": pd.Timestamp(today), "open": o.get("open"), "high": o.get("high"),
                          "low": o.get("low"), "close": q.get("last_price"), "volume": q.get("volume")}])


def futures_today(fut_list, quotes: dict, lot: float, prev: pd.DataFrame) -> pd.DataFrame:
    """Live futures with OI change per contract vs the last stored session
    (a contract absent yesterday is new, so its change is its whole OI,
    the same as the bhavcopy's CHG_IN_OI)."""
    prev_oi = dict(zip(prev["expiry"].dt.date, prev["oi"])) if len(prev) else None
    rows = []
    for exp, key in fut_list:
        q = quotes.get(key) or {}
        if not q:
            continue
        vol = float(q.get("volume") or 0)
        if lot and vol and vol % lot == 0:
            vol = vol / lot          # quote volume is in shares
        oi = float(q.get("oi") or 0)
        chg = (oi - prev_oi.get(exp, 0.0)) if prev_oi is not None else np.nan
        rows.append({"expiry": exp, "close": float(q.get("last_price") or np.nan), "oi": oi,
                     "chg_oi": chg, "volume": vol})
    return pd.DataFrame(rows)


def backfill_futures(ux: Upstox, fut_list, lot: float, today: dt.date, last_expiry: dt.date | None) -> pd.DataFrame:
    """Per-contract daily futures history from candles. Only sessions after the
    last expiry are used: before it, the then-near contract (now expired) would be missing."""
    start = (last_expiry + dt.timedelta(days=1)) if last_expiry else today - dt.timedelta(days=35)
    frames = []
    for exp, key in fut_list:
        c = ux.daily_candles(key, start, prev_weekday(today))
        if len(c):
            vol = c["volume"].astype(float)
            if lot and len(vol) and (vol % lot == 0).mean() > 0.97:
                vol = vol / lot
            frames.append(pd.DataFrame({"date": c["date"], "expiry": pd.Timestamp(exp), "close": c["close"],
                                        "oi": c["oi"].astype(float), "volume": vol}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fut_daily_from_contracts(fc: pd.DataFrame) -> pd.DataFrame:
    """Per-contract rows -> one row per day like the bhavcopy totals."""
    if fc.empty:
        return pd.DataFrame(columns=["date", "fut_oi", "fut_chg_oi", "fut_vol", "fut_close"])
    fc = fc.sort_values(["expiry", "date"])
    fc["chg"] = fc.groupby("expiry")["oi"].diff()
    g = fc.groupby("date")
    out = pd.DataFrame({"fut_oi": g["oi"].sum(), "fut_chg_oi": g["chg"].sum(min_count=1),
                        "fut_vol": g["volume"].sum(),
                        "fut_close": fc.sort_values("expiry").groupby("date")["close"].first()})
    first = fc["date"].min()
    out.loc[out.index == first, "fut_chg_oi"] = np.nan
    return out.reset_index()


def market_was_open(quotes: dict, keys: list, today: dt.date) -> bool:
    """True if any stock traded today (guards against stale data on holidays)."""
    seen = 0
    for k in keys[:50]:
        t = (quotes.get(k) or {}).get("last_trade_time")
        if not t:
            continue
        seen += 1
        try:
            if dt.datetime.fromtimestamp(int(t) / 1000, IST).date() == today:
                return True
        except (TypeError, ValueError):
            return True
    return seen == 0  # no timestamps at all: do not block


def run(config: dict, mode: str = "auto", dry_run: bool = False, only: list[str] | None = None,
        state_dir: str = "state", model_dir: str = "models", out_html: str | None = None) -> list[dict]:
    now = dt.datetime.now(IST)
    today = now.date()
    if mode == "auto":
        mode = "eod" if (now.hour, now.minute) >= (15, 35) else "intraday"
    acfg = config.get("alerts", {})
    if mode == "intraday" and not acfg.get("intraday", True):
        print("Intraday alerts are switched off in config.toml.")
        return []
    frac = 1.0 if mode == "eod" else session_fraction(now)
    st = State(state_dir)
    ux = Upstox(cache_dir=st.p("cache"))
    instr = ux.instruments()
    uni = ux.fno_universe(instr, today)
    excl = set(config.get("universe", {}).get("exclude", []))
    uni = uni[~uni["symbol"].isin(excl)]
    if only:
        uni = uni[uni["symbol"].isin(only)]
    print(f"[{now:%Y-%m-%d %H:%M} IST] mode={mode} stocks={len(uni)} volume_fraction={frac:.2f}", flush=True)

    meta, fitted = {}, None
    if os.path.exists(os.path.join(model_dir, "model.pkl")):
        fitted = Fitted.load(os.path.join(model_dir, "model.pkl"))
        meta = json.load(open(os.path.join(model_dir, "model_meta.json")))
    threshold = float(meta.get("threshold", acfg.get("fallback_rule_threshold", 0.45)))
    rules = strategy.TradeRules(**{k: v for k, v in meta.get("trade_rules", {}).items()
                                   if k in strategy.TradeRules().__dict__})
    min_turnover = float(config.get("backtest", {}).get("min_turnover_cr", 10.0))

    keys = list(uni["underlying_key"]) + [k for fl in uni["futures"] for _, k in fl] + list(INDEX_KEYS.values())
    quotes = ux.quotes(keys)
    if not market_was_open(quotes, list(uni["underlying_key"]), today):
        print("No trades today on NSE (holiday or weekend): nothing to scan.")
        return []

    # Market context (NIFTY, India VIX).
    mkt = st.load_table("market.csv")
    mkt = mkt.set_index("date")[["nifty", "vix"]] if len(mkt) else pd.DataFrame(columns=["nifty", "vix"])
    last_m = mkt.index.max().date() if len(mkt) else None
    if last_m is None or last_m < prev_weekday(today):
        cols = {}
        start = (last_m + dt.timedelta(days=1)) if last_m else today - dt.timedelta(days=500)
        for name, key in INDEX_KEYS.items():
            c = ux.daily_candles(key, start, prev_weekday(today))
            cols[name] = c.set_index("date")["close"] if len(c) else pd.Series(dtype=float)
        add = pd.DataFrame(cols)
        mkt = pd.concat([mkt, add]) if len(mkt) else add
        mkt = mkt[~mkt.index.duplicated(keep="last")]
    mkt_today = mkt.copy()
    for name, key in INDEX_KEYS.items():
        q = quotes.get(key)
        if q and q.get("last_price"):
            mkt_today.loc[pd.Timestamp(today), name] = float(q["last_price"])
    mkt_today = mkt_today.sort_index().astype(float)
    mkt_f = feat.market_features(mkt_today)

    fut_hist = st.load_table("fut_contracts.csv")
    snap_hist = st.load_table("opt_daily.parquet")
    lots_hist = st.load_table("lots.csv")
    new_fc, new_snaps, results, skipped = [], [], [], {"history": 0, "turnover": 0, "chain": 0}
    errors = 0
    for u in uni.itertuples(index=False):
        try:
            px = update_history(ux, st, u.symbol, u.underlying_key, today)
            bar = quote_bar(quotes.get(u.underlying_key, {}), today)
            if not bar.empty:
                bar["volume"] = bar["volume"] / frac
            px_today = merge_bars(px, bar)
            if len(px_today) < 80:
                skipped["history"] += 1
                continue
            # Futures: per-contract OI vs the last stored session before today.
            fh = fut_hist[(fut_hist["symbol"] == u.symbol) & (fut_hist["date"] < pd.Timestamp(today))] \
                if len(fut_hist) else pd.DataFrame()
            if fh.empty:
                exps_past = [e for e in pd.to_datetime(snap_hist[snap_hist["symbol"] == u.symbol]["focus_expiry"]).dt.date
                             if e < today] if len(snap_hist) and "focus_expiry" in snap_hist else []
                bf = backfill_futures(ux, u.futures, u.lot, today, max(exps_past) if exps_past else None)
                if len(bf):
                    bf["symbol"] = u.symbol
                    new_fc.append(bf)
                    fh = bf
            prev_day = fh[fh["date"] == fh["date"].max()] if len(fh) else pd.DataFrame()
            fut = futures_today(u.futures, quotes, u.lot, prev_day)
            fc_today = fut.assign(date=pd.Timestamp(today), symbol=u.symbol,
                                  expiry=pd.to_datetime(fut["expiry"]))[["date", "symbol", "expiry", "close", "oi", "volume"]]
            new_fc.append(fc_today)

            exps = [e for e in u.option_expiries if e >= today][:2]
            frames, spot = [], math.nan
            for e in exps:
                f, sp = chain_to_frames(ux.option_chain(u.underlying_key, e), u.lot)
                if len(f):
                    frames.append(f)
                    spot = sp if sp == sp else spot
            if not frames:
                skipped["chain"] += 1
                continue
            opts = pd.concat(frames, ignore_index=True)
            snap = chain.snapshot(opts, fut, today, spot)
            for k in ("ce_vol", "pe_vol", "fut_vol"):
                snap[k] = snap.get(k, np.nan) / frac
            snap.update({"date": pd.Timestamp(today), "symbol": u.symbol, "lot": u.lot})
            new_snaps.append(snap)

            # Derivatives history = stored EOD snapshots, with futures totals filled
            # from per-contract history where snapshots are missing, plus today.
            hist = snap_hist[snap_hist["symbol"] == u.symbol].set_index("date") if len(snap_hist) else pd.DataFrame()
            fd = fut_daily_from_contracts(fh).set_index("date") if len(fh) else pd.DataFrame()
            if len(fd):
                hist = hist.combine_first(fd) if len(hist) else fd
            hist = pd.concat([hist, pd.DataFrame([snap]).set_index("date")])
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()
            for col in chain_cols():
                if col not in hist:
                    hist[col] = np.nan
            lots = lots_hist[lots_hist["symbol"] == u.symbol].set_index("date")["lot"] if len(lots_hist) else None
            if lots is not None:
                lots = pd.concat([lots[lots.index < pd.Timestamp(today)], pd.Series([u.lot], index=[pd.Timestamp(today)])])
            pxa = prices.adjust_splits(px_today, lots)
            row = panel.stock_frame(pxa, hist, mkt_f, None, with_labels=False).iloc[-1].to_dict()
            if row.get("dvol_cr", np.inf) < min_turnover:
                skipped["turnover"] += 1
                continue
            row.update({"symbol": u.symbol, "spot": spot if spot == spot else row.get("close"), "lot": u.lot,
                        "max_pain": snap.get("max_pain"), "call_wall": snap.get("call_wall"),
                        "put_wall": snap.get("put_wall"), "F": snap.get("F"), "dte": snap.get("dte"),
                        "corp_action_today": int(pxa["corp_action"].iloc[-1])})
            results.append((row, opts))
        except Exception as e:  # one bad stock must not kill the scan
            errors += 1
            print(f"  {u.symbol}: {type(e).__name__}: {e}", flush=True)
            if errors <= 3:
                traceback.print_exc()
    print(f"Scanned {len(results)} stocks ({errors} errors, skipped {skipped}, {ux.calls} API calls)", flush=True)

    if not dry_run:
        _persist(st, mode, today, uni, quotes, new_snaps, new_fc, fut_hist, snap_hist, lots_hist, mkt, mkt_today)

    if not results:
        return []
    df = pd.DataFrame([r for r, _ in results])
    chains = {r["symbol"]: c for r, c in results}
    if fitted is not None:
        for c in fitted.features:
            if c not in df:
                df[c] = np.nan
        df["missing_share"] = df[fitted.features].isna().mean(axis=1)
        df = pd.concat([df, fitted.predict(df).set_index(df.index)], axis=1)
        thin = df["missing_share"] > 0.25
        if thin.any():
            print(f"{int(thin.sum())} stocks have too little history for the model and were not scored.")
        df = df[~thin]
    else:
        df["p_big"] = explain.rule_scores(df) / 100.0
        df["p_up"] = df["p_dn"] = df["p_big"] / 2

    df["date"] = pd.Timestamp(today)
    cand = alerting.emit(df, threshold, "p_big", int(acfg.get("max_per_email", 10)), cooldown_days=10**6)
    log = st.load_table("alerts_log.csv")
    alerts = []
    for r in cand.to_dict("records"):
        if recently_alerted(log, r, mode, now):
            continue
        pts, why, dvote = explain.reasons(r)
        if fitted is None and abs(dvote) > 0.25:
            r["p_up"], r["p_dn"] = (r["p_big"] * 0.7, r["p_big"] * 0.3) if dvote > 0 else (r["p_big"] * 0.3, r["p_big"] * 0.7)
        action, dlabel, twhy = strategy.suggest(r["p_up"], r["p_dn"], r.get("iv_rank", np.nan), r.get("iv_hv", np.nan), rules)
        if r.get("corp_action_today"):
            why.insert(0, "Note: a split/bonus was detected today and prices were adjusted; check before trading")
        r.update({"action": action, "direction": dlabel, "trade_why": twhy, "reasons": why,
                  "trade_text": trade_text(action, r, chains[r["symbol"]], rules, today),
                  "legs": trade_legs(action, r, chains[r["symbol"]], rules, today),
                  "buildup": feat.buildup_label(r.get("ret1", np.nan), r.get("fut_oi_chg", np.nan)),
                  "expiry": str(strategy.pick_trade_expiry(chains[r["symbol"]]["expiry"].unique(),
                                                           pd.Timestamp(today), rules.min_trade_dte).date())})
        alerts.append(r)

    label = (f"{today:%d %b %Y} close (backtested signal, act at next open)" if mode == "eod"
             else f"{today:%d %b %Y} {now:%H:%M} IST early estimate (not backtested)")
    meta_view = {"base_rate": meta.get("base_rate_holdout"), "hit_rate": meta.get("holdout_hit_rate"),
                 "holdout_period": meta.get("holdout"), "direction_enabled": rules.dir_conf <= 1}
    if alerts:
        subject, text, body = emailer.render(alerts, meta_view, label)
        if fitted is None:
            subject = "[UNVALIDATED RULES] " + subject
        if out_html:
            open(out_html, "w").write(body)
        if dry_run:
            print(text)
        elif not (os.environ.get("SMTP_USER") and os.environ.get("SMTP_PASSWORD")):
            print("Email not set up (SMTP_USER/SMTP_PASSWORD missing), so this alert is only in the log and dashboard.")
        else:
            emailer.send(subject, text, body)
        if not dry_run:
            new_log = pd.DataFrame([{"time": now.replace(tzinfo=None), "date": pd.Timestamp(today),
                                     "kind": mode, "symbol": a["symbol"], "action": a["action"],
                                     "p_big": a["p_big"]} for a in alerts])
            st.save_table("alerts_log.csv", pd.concat([log, new_log], ignore_index=True) if len(log) else new_log)
        print(f"{len(alerts)} alert(s): " + ", ".join(f"{a['symbol']} {a['action']}" for a in alerts))
    else:
        print("No stock above the alert threshold.")
    if not dry_run:
        try:
            last = mkt_today.iloc[-1] if len(mkt_today) else {}
            dashlog.record(st.root, now, mode, df, alerts, threshold, "model" if fitted is not None else "rules",
                           len(results), errors, ux.calls, last.get("nifty"), last.get("vix"))
        except Exception as e:  # the dashboard log must never block alerts
            print(f"Dashboard log not written: {type(e).__name__}: {e}")
        try:
            run_paper(st, ux, instr, alerts, now, mode, list(mkt_today.index), rules, paper.Sizing.from_cfg(config))
        except Exception as e:  # paper trading must never block alerts
            print(f"Paper trades not updated: {type(e).__name__}: {e}")
    return alerts


def recently_alerted(log: pd.DataFrame, r: dict, mode: str, now: dt.datetime) -> bool:
    """End-of-day alerts follow the backtest rule (no repeat within 5 sessions unless
    the probability rose by 0.10). Intraday alerts are not repeated within 20 hours."""
    if log is None or log.empty:
        return False
    mine = log[log["symbol"] == r["symbol"]]
    if mine.empty:
        return False
    if mode == "eod":
        eod = mine[mine["kind"] == "eod"] if "kind" in mine else mine
        if eod.empty:
            return False
        last = eod.sort_values("date").iloc[-1]
        sessions = np.busday_count(pd.Timestamp(last["date"]).date(), now.date())
        return sessions <= alerting.COOLDOWN_DAYS and r["p_big"] < last["p_big"] + alerting.REALERT_BUMP
    recent = mine[pd.to_datetime(mine["time"]) >= pd.Timestamp(now.replace(tzinfo=None)) - pd.Timedelta(hours=20)]
    return len(recent) > 0


def _persist(st, mode, today, uni, quotes, new_snaps, new_fc, fut_hist, snap_hist, lots_hist, mkt, mkt_today):
    """Save history. Only end-of-day values become history; intraday runs only
    save back-filled past sessions."""
    t = pd.Timestamp(today)
    if new_fc:
        fc = pd.concat([fut_hist] + new_fc, ignore_index=True) if len(fut_hist) else pd.concat(new_fc, ignore_index=True)
        fc["date"] = pd.to_datetime(fc["date"])
        if mode != "eod":
            fc = fc[fc["date"] < t]
        fc = fc.drop_duplicates(["symbol", "expiry", "date"], keep="last")
        fc = fc[fc["date"] >= t - pd.Timedelta(days=120)]
        st.save_table("fut_contracts.csv", fc)
    st.save_table("market.csv", (mkt_today if mode == "eod" else mkt).reset_index().rename(columns={"index": "date"}))
    if mode != "eod":
        return
    if new_snaps:
        ns = pd.DataFrame(new_snaps)
        sh = pd.concat([snap_hist, ns], ignore_index=True) if len(snap_hist) else ns
        sh["date"] = pd.to_datetime(sh["date"])
        sh["focus_expiry"] = pd.to_datetime(sh["focus_expiry"])
        st.save_table("opt_daily.parquet", sh.drop_duplicates(["symbol", "date"], keep="last"))
    for u in uni.itertuples(index=False):
        bar = quote_bar(quotes.get(u.underlying_key, {}), today)
        if not bar.empty:
            st.save_px(u.symbol, merge_bars(st.load_px(u.symbol), bar))
    lots_new = pd.DataFrame({"date": t, "symbol": uni["symbol"], "lot": uni["lot"]})
    lots = pd.concat([lots_hist, lots_new], ignore_index=True) if len(lots_hist) else lots_new
    st.save_table("lots.csv", lots.drop_duplicates(["symbol", "date"], keep="last"))


def chain_cols():
    return ["atm_iv", "straddle_pct", "skew", "dte", "ce_oi", "pe_oi", "ce_chg_oi", "pe_chg_oi", "ce_vol",
            "pe_vol", "fut_oi", "fut_chg_oi", "fut_vol", "pe_chg_near", "ce_chg_near", "pe_oi_near",
            "ce_oi_near", "F", "max_pain", "call_wall", "put_wall", "fut_close", "lot"]


def run_paper(st, ux, instr, alerts, now, mode, session_dates, rules, sizing=paper.Sizing()):
    """Record new alerts as paper trades and fill entries/exits on real prices."""
    today = now.date()
    book = paper.load(st.root)
    book = paper.add_alerts(book, alerts, instr, now)
    if book.empty:
        return
    book = paper.resolve_keys(book, instr)

    def candle(key, day):
        c = ux.daily_candles(key, day, day)
        c = c[c["date"].dt.date == day] if len(c) else c
        return (float(c["open"].iloc[0]), float(c["close"].iloc[0])) if len(c) else (math.nan, math.nan)

    sessions = sorted({pd.Timestamp(d).date() for d in session_dates})
    after_close = mode == "eod" or now.time() >= dt.time(15, 30)
    book = paper.update(book, today, sessions, after_close, ux.quotes, candle, rules, sizing)
    open_keys = [k for _, r in book[book["status"] == paper.OPEN].iterrows() for k in (r["ce_key"], r["pe_key"])
                 if isinstance(k, str) and k]
    if open_keys:
        book = paper.mark_to_market(book, ux.quotes(sorted(set(open_keys))), rules)
    paper.save(st.root, book)
    done = book[book["status"] == paper.CLOSED]
    print(f"Paper trades: {int((book['status'] == paper.OPEN).sum())} open, "
          f"{int((book['status'] == paper.PENDING).sum())} waiting for the open, {len(done)} closed"
          + (f", total P&L Rs {done['pnl_total'].fillna(0).sum():,.0f}" if len(done) else "")
          + f", Rs {paper.equity(book, sizing)['in_use']:,.0f} in open trades")


def trade_legs(action: str, r: dict, opts: pd.DataFrame, rules: strategy.TradeRules, today: dt.date) -> list:
    """[(leg, strike, expiry date, last price)] for the suggested trade, or []."""
    if action == "NO TRADE":
        return []
    exp = strategy.pick_trade_expiry(opts["expiry"].unique(), pd.Timestamp(today), rules.min_trade_dte)
    if exp is None:
        return []
    ch = opts[pd.to_datetime(opts["expiry"]) == exp]
    F = r.get("F") or r.get("spot")
    if ch.empty or not F:
        return []
    legs = ["BUY CE", "BUY PE"] if action == "BUY CE + BUY PE" else [action]
    out = []
    for leg in legs:
        K = strategy.strike_for(leg, F, ch["strike"].unique(), r.get("atm_iv", np.nan), (exp.date() - today).days, rules)
        if not (K == K):
            return []
        q = ch[(ch["strike"] == K) & (ch["opt"] == leg[-2:])]
        ltp = float(q["close"].iloc[0]) if len(q) and q["close"].iloc[0] == q["close"].iloc[0] else math.nan
        out.append((leg, float(K), exp.date(), ltp))
    return out


def trade_text(action: str, r: dict, opts: pd.DataFrame, rules: strategy.TradeRules, today: dt.date) -> str:
    if action == "NO TRADE":
        return "No trade suggested"
    legs = trade_legs(action, r, opts, rules, today)
    if not legs:
        return action
    F = r.get("F") or r.get("spot")
    parts, total = [], 0.0
    lot = r.get("lot") or 0
    for leg, K, exp, ltp in legs:
        total += ltp if ltp == ltp else 0
        parts.append(f"{leg} {K:g} {exp:%d %b}" + (f" (last Rs {ltp:,.2f})" if ltp == ltp else ""))
    txt = " + ".join(parts)
    if lot and total:
        if action.startswith("SELL"):
            txt += f"; credit about Rs {total * lot:,.0f} per lot of {lot:g}, margin roughly Rs {rules.margin_frac * F * lot:,.0f}"
        else:
            txt += f"; cost about Rs {total * lot:,.0f} per lot of {lot:g}"
    return txt + f". Backtest rule: enter at the next open, exit after {rules.hold_days} sessions."
