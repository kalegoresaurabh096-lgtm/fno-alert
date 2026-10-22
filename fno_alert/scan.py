"""Live scan: pull Upstox data, score every F&O stock, email the alerts."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import traceback

import numpy as np
import pandas as pd

from . import chain, emailer, explain, features as feat, panel, prices, strategy
from .model import Fitted
from .upstox import IST, Upstox, chain_to_frames, candles_to_frame

INDEX_KEYS = {"nifty": "NSE_INDEX|Nifty 50", "vix": "NSE_INDEX|India VIX"}
# Typical share of a day's volume traded by each time (IST), for intraday projection.
VOLUME_CURVE = [(9.25, 0.0), (10.0, 0.22), (11.0, 0.38), (12.0, 0.50), (13.0, 0.60),
                (14.0, 0.72), (15.0, 0.88), (15.5, 1.0)]


def session_fraction(now: dt.datetime) -> float:
    h = now.hour + now.minute / 60
    xs, ys = zip(*VOLUME_CURVE)
    return float(np.clip(np.interp(h, xs, ys), 0.05, 1.0))


class State:
    def __init__(self, root: str = "state"):
        self.root = root
        for d in ("ohlcv", "cache"):
            os.makedirs(os.path.join(root, d), exist_ok=True)

    def p(self, *a):
        return os.path.join(self.root, *a)

    def load_px(self, sym: str) -> pd.DataFrame:
        f = self.p("ohlcv", f"{sym}.csv")
        if not os.path.exists(f):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()

    def save_px(self, sym: str, px: pd.DataFrame):
        px[["open", "high", "low", "close", "volume"]].to_csv(self.p("ohlcv", f"{sym}.csv"), index_label="date")

    def load_table(self, name: str) -> pd.DataFrame:
        f = self.p(name)
        if not os.path.exists(f):
            return pd.DataFrame()
        return pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, parse_dates=["date"])

    def save_table(self, name: str, df: pd.DataFrame):
        f = self.p(name)
        df.to_parquet(f, index=False) if f.endswith(".parquet") else df.to_csv(f, index=False)


def _merge_bar(px: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new is None or new.empty:
        return px
    new = new.set_index("date")[["open", "high", "low", "close", "volume"]] if "date" in new else new
    out = pd.concat([px[~px.index.isin(new.index)], new]).sort_index()
    return out[~out.index.duplicated(keep="last")]


def update_history(ux: Upstox, st: State, sym: str, key: str, today: dt.date, days: int = 600) -> pd.DataFrame:
    px = st.load_px(sym)
    yday = today - dt.timedelta(days=1)
    last = px.index.max().date() if len(px) else None
    if last is None or last < yday - dt.timedelta(days=4):
        start = (last - dt.timedelta(days=7)) if last else today - dt.timedelta(days=days)
        px = _merge_bar(px, ux.daily_candles(key, start, yday))
        st.save_px(sym, px)
    return px


def quote_bar(q: dict, today: dt.date) -> pd.DataFrame:
    o = q.get("ohlc") or {}
    if not q or not q.get("last_price"):
        return pd.DataFrame()
    return pd.DataFrame([{"date": pd.Timestamp(today), "open": o.get("open"), "high": o.get("high"),
                          "low": o.get("low"), "close": q.get("last_price"), "volume": q.get("volume")}])


def futures_today(fut_list, quotes: dict, lot: float) -> pd.DataFrame:
    rows = []
    for exp, key in fut_list:
        q = quotes.get(key) or {}
        if not q:
            continue
        vol = float(q.get("volume") or 0)
        if lot and vol and vol % lot == 0:
            vol = vol / lot
        rows.append({"expiry": exp, "close": float(q.get("last_price") or np.nan),
                     "oi": float(q.get("oi") or 0), "chg_oi": np.nan, "volume": vol})
    return pd.DataFrame(rows)


def backfill_fut_oi(ux: Upstox, fut_list, lot: float, today: dt.date) -> pd.DataFrame:
    """Daily total futures OI per day from the live contracts' candles."""
    frames = []
    for exp, key in fut_list:
        c = ux.daily_candles(key, today - dt.timedelta(days=120), today - dt.timedelta(days=1))
        if len(c):
            frames.append(c[["date", "oi", "volume", "close"]].assign(expiry=exp))
    if not frames:
        return pd.DataFrame()
    allc = pd.concat(frames)
    tot = allc.groupby("date").agg(fut_oi=("oi", "sum"), fut_vol=("volume", "sum"))
    near = allc.sort_values("expiry").groupby("date")["close"].first()
    tot["fut_close"] = near
    if lot and (tot["fut_vol"] % lot == 0).mean() > 0.97:
        tot["fut_vol"] = tot["fut_vol"] / lot
    return tot.reset_index()


def run(config: dict, mode: str = "auto", dry_run: bool = False, only: list[str] | None = None,
        state_dir: str = "state", model_dir: str = "models", out_html: str | None = None) -> list[dict]:
    now = dt.datetime.now(IST)
    today = now.date()
    if mode == "auto":
        mode = "eod" if (now.hour, now.minute) >= (15, 35) else "intraday"
    frac = 1.0 if mode == "eod" else session_fraction(now)
    st = State(state_dir)
    ux = Upstox(cache_dir=st.p("cache"))
    uni = ux.fno_universe(ux.instruments(), today)
    excl = set(config.get("universe", {}).get("exclude", []))
    uni = uni[~uni["symbol"].isin(excl)]
    if only:
        uni = uni[uni["symbol"].isin(only)]
    print(f"[{now:%Y-%m-%d %H:%M} IST] mode={mode} stocks={len(uni)} volume_fraction={frac:.2f}", flush=True)

    # Load the validated model, if one has been trained.
    meta, fitted = {}, None
    if os.path.exists(os.path.join(model_dir, "model.pkl")):
        fitted = Fitted.load(os.path.join(model_dir, "model.pkl"))
        meta = json.load(open(os.path.join(model_dir, "model_meta.json")))
    acfg = config.get("alerts", {})
    threshold = float(meta.get("threshold", acfg.get("fallback_rule_threshold", 0.45)))
    rules = strategy.TradeRules(**{k: v for k, v in meta.get("trade_rules", {}).items()
                                   if k in strategy.TradeRules().__dict__})

    # Batched quotes for every stock, future and the two indices.
    keys = list(uni["underlying_key"]) + [k for fl in uni["futures"] for _, k in fl] + list(INDEX_KEYS.values())
    quotes = ux.quotes(keys)
    if not market_was_open(quotes, list(uni["underlying_key"]), today):
        print("No trades today on NSE (holiday or weekend): nothing to scan.")
        return []

    # Market context.
    mkt_hist = st.load_table("market.csv")
    mkt_hist = mkt_hist.set_index("date") if len(mkt_hist) else pd.DataFrame(columns=["nifty", "vix"])
    if len(mkt_hist) < 260:
        cols = {}
        for name, key in INDEX_KEYS.items():
            c = ux.daily_candles(key, today - dt.timedelta(days=500), today - dt.timedelta(days=1))
            cols[name] = c.set_index("date")["close"] if len(c) else pd.Series(dtype=float)
        mkt_hist = pd.DataFrame(cols)
    for name, key in INDEX_KEYS.items():
        q = quotes.get(key)
        if q and q.get("last_price"):
            mkt_hist.loc[pd.Timestamp(today), name] = float(q["last_price"])
    mkt_hist = mkt_hist.sort_index()
    mkt_f = feat.market_features(mkt_hist.astype(float))

    fut_hist = st.load_table("fut_daily.csv")
    snap_hist = st.load_table("opt_daily.parquet")
    new_fut, new_snaps, results = [], [], []
    errors = 0
    for u in uni.itertuples(index=False):
        try:
            px = update_history(ux, st, u.symbol, u.underlying_key, today)
            bar = quote_bar(quotes.get(u.underlying_key, {}), today)
            if not bar.empty:
                bar["volume"] = bar["volume"] / frac
            px_today = _merge_bar(px, bar)
            if len(px_today) < 80:
                continue
            fh = fut_hist[fut_hist["symbol"] == u.symbol] if len(fut_hist) else pd.DataFrame()
            if len(fh) < 20:
                bf = backfill_fut_oi(ux, u.futures, u.lot, today)
                if len(bf):
                    bf["symbol"] = u.symbol
                    new_fut.append(bf)
                    fh = bf
            fut = futures_today(u.futures, quotes, u.lot)
            prev_oi = float(fh.sort_values("date")["fut_oi"].iloc[-1]) if len(fh) else np.nan
            focus = chain.pick_focus_expiry(u.option_expiries, today)
            exps = [e for e in u.option_expiries if e >= today][:2]
            frames, spot = [], math.nan
            for e in exps:
                f, sp = chain_to_frames(ux.option_chain(u.underlying_key, e), u.lot)
                if len(f):
                    frames.append(f)
                    spot = sp if sp == sp else spot
            if not frames:
                continue
            opts = pd.concat(frames, ignore_index=True)
            snap = chain.snapshot(opts, fut.assign(chg_oi=0.0), today, spot)
            snap["fut_chg_oi"] = snap["fut_oi"] - prev_oi if prev_oi == prev_oi else np.nan
            for k in ("ce_vol", "pe_vol", "fut_vol"):
                snap[k] = snap.get(k, np.nan) / frac
            snap.update({"date": pd.Timestamp(today), "symbol": u.symbol, "lot": u.lot})
            new_snaps.append(snap)
            new_fut.append(pd.DataFrame([{"date": pd.Timestamp(today), "symbol": u.symbol,
                                          "fut_oi": snap["fut_oi"], "fut_vol": snap["fut_vol"],
                                          "fut_close": snap.get("fut_close")}]))
            # Derivatives history for this stock: stored EOD snapshots + futures OI + today.
            hist = snap_hist[snap_hist["symbol"] == u.symbol].copy() if len(snap_hist) else pd.DataFrame()
            if len(fh):
                fo = fh.set_index("date")[["fut_oi", "fut_vol", "fut_close"]]
                hist = hist.set_index("date") if len(hist) else pd.DataFrame(index=fo.index)
                hist = hist.combine_first(fo).reset_index().rename(columns={"index": "date"})
            hist = pd.concat([hist, pd.DataFrame([snap])], ignore_index=True)
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.drop_duplicates("date", keep="last").set_index("date").sort_index()
            hist["fut_chg_oi"] = hist["fut_chg_oi"].fillna(hist["fut_oi"].diff()) if "fut_chg_oi" in hist else hist["fut_oi"].diff()
            for col in chain_cols():
                if col not in hist:
                    hist[col] = np.nan
            pxa = prices.adjust_splits(px_today)
            row = panel.stock_frame(pxa, hist, mkt_f, None, with_labels=False).iloc[-1].to_dict()
            row.update({"symbol": u.symbol, "spot": spot if spot == spot else row.get("close"), "lot": u.lot,
                        "max_pain": snap.get("max_pain"), "call_wall": snap.get("call_wall"),
                        "put_wall": snap.get("put_wall"), "expiry": str(focus), "F": snap.get("F"),
                        "focus_expiry": focus, "dte": snap.get("dte")})
            results.append((row, opts))
        except Exception as e:  # one bad stock must not kill the scan
            errors += 1
            print(f"  {u.symbol}: {type(e).__name__}: {e}", flush=True)
            if errors <= 3:
                traceback.print_exc()
    print(f"Scanned {len(results)} stocks ({errors} errors, {ux.calls} API calls)", flush=True)

    # Persist history (only end-of-day values become history).
    if mode == "eod" and not dry_run:
        if new_snaps:
            ns = pd.DataFrame(new_snaps)
            sh = pd.concat([snap_hist, ns], ignore_index=True) if len(snap_hist) else ns
            sh["date"] = pd.to_datetime(sh["date"])
            sh = sh.drop_duplicates(["symbol", "date"], keep="last")
            st.save_table("opt_daily.parquet", sh.astype({"focus_expiry": "string"}))
        for u in uni.itertuples(index=False):
            bar = quote_bar(quotes.get(u.underlying_key, {}), today)
            if not bar.empty:
                st.save_px(u.symbol, _merge_bar(st.load_px(u.symbol), bar))
        st.save_table("market.csv", mkt_hist.reset_index().rename(columns={"index": "date"}))
    if new_fut and not dry_run:
        fu = pd.concat([fut_hist] + new_fut, ignore_index=True) if len(fut_hist) else pd.concat(new_fut)
        fu["date"] = pd.to_datetime(fu["date"])
        if mode != "eod":
            fu = fu[fu["date"] < pd.Timestamp(today)]
        st.save_table("fut_daily.csv", fu.drop_duplicates(["symbol", "date"], keep="last"))

    if not results:
        return []
    df = pd.DataFrame([r for r, _ in results])
    chains = {r["symbol"]: c for r, c in results}
    if fitted is not None:
        for c in fitted.features:
            if c not in df:
                df[c] = np.nan
        df = pd.concat([df, fitted.predict(df).set_index(df.index)], axis=1)
        score_col = "p_big"
    else:
        df["p_big"] = explain.rule_scores(df) / 100.0
        df["p_up"] = df["p_dn"] = df["p_big"] / 2
        score_col = "p_big"
    cand = df[df[score_col] >= threshold].sort_values(score_col, ascending=False)
    cand = cand.head(int(acfg.get("max_per_email", 10)))
    alerts = []
    log = st.load_table("alerts_log.csv")
    cooldown = float(acfg.get("cooldown_hours", 20))
    for r in cand.to_dict("records"):
        pts, why, dvote = explain.reasons(r)
        if fitted is None and abs(dvote) > 0.25:
            r["p_up"], r["p_dn"] = (r["p_big"] * 0.7, r["p_big"] * 0.3) if dvote > 0 else (r["p_big"] * 0.3, r["p_big"] * 0.7)
        action, dlabel, twhy = strategy.suggest(r["p_up"], r["p_dn"], r.get("iv_rank", np.nan), r.get("iv_hv", np.nan), rules)
        if len(log):
            recent = log[(log["symbol"] == r["symbol"]) &
                         (pd.to_datetime(log["time"]) >= pd.Timestamp(now.replace(tzinfo=None)) - pd.Timedelta(hours=cooldown))]
            if len(recent) and recent["action"].iloc[-1] == action and r["p_big"] < recent["p_big"].max() + 0.1:
                continue
        r.update({"action": action, "direction": dlabel, "trade_why": twhy, "reasons": why,
                  "trade_text": trade_text(action, r, chains[r["symbol"]], rules),
                  "buildup": feat.buildup_label(r.get("ret1", np.nan), r.get("fut_oi_chg", np.nan))})
        alerts.append(r)
    label = f"{today:%d %b %Y} {'close' if mode == 'eod' else now.strftime('%H:%M') + ' IST (intraday estimate)'}"
    meta_view = {"base_rate": meta.get("base_rate_holdout"), "hit_rate": meta.get("holdout_hit_rate"),
                 "holdout_period": meta.get("holdout")}
    if fitted is None:
        meta_view["hit_rate"] = None
    if alerts:
        subject, text, body = emailer.render(alerts, meta_view, label)
        if fitted is None:
            subject = "[UNVALIDATED RULES] " + subject
        if out_html:
            open(out_html, "w").write(body)
        if dry_run:
            print(text)
        else:
            emailer.send(subject, text, body)
            new_log = pd.DataFrame([{"time": now.replace(tzinfo=None), "symbol": a["symbol"], "action": a["action"],
                                     "p_big": a["p_big"]} for a in alerts])
            st.save_table("alerts_log.csv", pd.concat([log, new_log], ignore_index=True) if len(log) else new_log)
        print(f"{len(alerts)} alert(s): " + ", ".join(f"{a['symbol']} {a['action']}" for a in alerts))
    else:
        print("No stock above the alert threshold.")
    return alerts


def market_was_open(quotes: dict, keys: list, today: dt.date) -> bool:
    """True if any stock traded today (guards against stale data on holidays)."""
    seen = 0
    for k in keys[:50]:
        t = (quotes.get(k) or {}).get("last_trade_time")
        if not t:
            continue
        seen += 1
        try:
            d = dt.datetime.fromtimestamp(int(t) / 1000, IST).date()
        except (TypeError, ValueError):
            return True
        if d == today:
            return True
    return seen == 0  # no timestamps at all: do not block


def chain_cols():
    return ["atm_iv", "straddle_pct", "skew", "dte", "ce_oi", "pe_oi", "ce_chg_oi", "pe_chg_oi", "ce_vol",
            "pe_vol", "fut_oi", "fut_chg_oi", "fut_vol", "pe_chg_near", "ce_chg_near", "pe_oi_near",
            "ce_oi_near", "F", "max_pain", "call_wall", "put_wall", "fut_close"]


def trade_text(action: str, r: dict, opts: pd.DataFrame, rules: strategy.TradeRules) -> str:
    if action == "NO TRADE":
        return "No trade suggested"
    focus = r.get("focus_expiry")
    ch = opts[opts["expiry"] == focus]
    F = r.get("F") or r.get("spot")
    if ch.empty or not F:
        return action
    strikes = ch["strike"].unique()
    legs = ["BUY CE", "BUY PE"] if action == "BUY CE + BUY PE" else [action]
    parts, total = [], 0.0
    lot = r.get("lot") or 0
    for leg in legs:
        K = strategy._strike_for(leg, F, strikes, r.get("atm_iv", np.nan), int(r.get("dte") or 20), rules)
        q = ch[(ch["strike"] == K) & (ch["opt"] == leg[-2:])]
        ltp = float(q["close"].iloc[0]) if len(q) and q["close"].iloc[0] == q["close"].iloc[0] else math.nan
        total += ltp if ltp == ltp else 0
        prem = f" @ about Rs {ltp:,.2f}" if ltp == ltp else ""
        parts.append(f"{leg} {K:g} {pd.Timestamp(focus):%d %b}{prem}")
    txt = " + ".join(parts)
    if lot and total:
        if action.startswith("SELL"):
            txt += f" (credit about Rs {total * lot:,.0f} per lot of {lot:g}; margin roughly Rs {rules.margin_frac * F * lot:,.0f})"
        else:
            txt += f" (cost about Rs {total * lot:,.0f} per lot of {lot:g})"
    return txt + f". Backtest exit: close after {rules.hold_days} trading days."
