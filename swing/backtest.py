"""Swing trade backtest: long NSE cash stocks, +3% target, max 15 trading days.

Data: daily OHLCV for the F&O stocks (all well above Rs 5,000 cr market cap) from the
`history-data` branch (eq/*.csv) and NIFTY from the `nifty-data` branch (index.csv).

    git archive origin/history-data eq | tar -x -C /tmp/hd
    git show origin/nifty-data:index.csv > /tmp/hd/nifty.csv
    python swing/backtest.py --data /tmp/hd --out swing/results

Rules are tuned on 2023-2024 only (TRAIN) and then run once on 2025-2026 (TEST).
Signals use the close of day t; entry is at the open of day t+1. A limit sell sits at
entry * 1.03; a stop (if any) sits below. If both are touched on the same day the stop is
assumed to fill first. Exit at the close of the 15th trading day otherwise.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2024-12-31")   # overridden by --train-end
TEST_START = pd.Timestamp("2025-01-01")
MIN_MCAP = 5_000e7  # Rs 5,000 crore
CAPITAL = 500_000.0
SLIPPAGE = 0.001  # 0.10% per side, on every fill


# ---------------------------------------------------------------- costs (delivery)
def round_trip_cost(buy_value: float, sell_value: float) -> float:
    """Upstox-style delivery charges in rupees for one buy + one sell."""
    brokerage = 20.0 + 20.0
    stt = 0.001 * (buy_value + sell_value)
    exch = 0.0000297 * (buy_value + sell_value)
    sebi = 0.000001 * (buy_value + sell_value)
    stamp = 0.00015 * buy_value
    gst = 0.18 * (brokerage + exch + sebi)
    dp = 20.0  # depository charge on the sell day
    return brokerage + stt + exch + sebi + stamp + gst + dp


# ---------------------------------------------------------------- data
def load(data_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    stocks = {}
    for f in sorted((data_dir / "eq").glob("*.csv")):
        d = pd.read_csv(f, parse_dates=["date"]).drop_duplicates("date").sort_values("date")
        d = d.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        # bad ticks: keep high/low within 20% of the day's open/close (NSE circuit range)
        d["high"] = d.high.clip(upper=d[["open", "close"]].max(axis=1) * 1.2)
        d["low"] = d.low.clip(lower=d[["open", "close"]].min(axis=1) * 0.8)
        if len(d) < 260:
            continue
        stocks[f.stem] = d
    uni = data_dir / "universe.csv"
    if uni.exists():  # price x today's shares = estimated market cap on each day
        sh = pd.read_csv(uni).dropna(subset=["shares"]).set_index("symbol").shares
        stocks = {s: d.assign(mcap=d.close * sh[s]) for s, d in stocks.items() if s in sh.index}
    nifty = pd.read_csv(data_dir / "nifty.csv", parse_dates=["date"]).set_index("date").sort_index()
    return stocks, nifty


def rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    c = d.close
    f["close"] = c
    f["sma10"] = c.rolling(10).mean()
    f["sma20"] = c.rolling(20).mean()
    f["sma50"] = c.rolling(50).mean()
    f["sma200"] = c.rolling(200).mean()
    f["rsi2"] = rsi(c, 2)
    f["rsi14"] = rsi(c, 14)
    tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["hh20"] = d.high.rolling(20).max().shift()
    f["vol_ratio"] = d.volume / d.volume.rolling(20).mean().shift()
    f["bb_low"] = f.sma20 - 2 * c.rolling(20).std()
    f["ret126"] = c / c.shift(126) - 1
    f["down3"] = (c.diff() < 0).astype(int).rolling(3).sum() == 3
    f["turnover"] = (c * d.volume).rolling(20).mean()
    f["mcap_ok"] = d.mcap > MIN_MCAP if "mcap" in d else True
    # demerger / bad-print gaps: never hold across them
    gap = d.open / c.shift() - 1
    f["bad_gap"] = (gap < -0.3) | (gap > 0.4)
    return f


# ---------------------------------------------------------------- rules
@dataclass(frozen=True)
class Rule:
    setup: str          # which entry pattern
    param: float        # setup threshold
    market: str         # none | nifty50 | nifty200
    stop: float         # stop distance as fraction (0 = no stop, time exit only)
    target: float = 0.03
    max_days: int = 15

    @property
    def name(self) -> str:
        stop = f"stop {self.stop:.0%}" if self.stop else "no stop"
        stop += f" | max {self.max_days} days"
        mkt = {"none": "any market", "nifty50": "NIFTY > 50-day avg", "nifty200": "NIFTY > 200-day avg"}[self.market]
        return f"{SETUPS[self.setup][0].format(p=self.param)} | {mkt} | {stop}"


SETUPS = {
    "rsi2": ("Uptrend dip: above 200-day avg, RSI(2) < {p:g}",
             lambda f, p: (f.close > f.sma200) & (f.rsi2 < p), lambda f: -f.rsi2),
    "down3": ("3 down days in uptrend (50 > 200-day avg), {p:g}% or more below 10-day avg",
              lambda f, p: f.down3 & (f.sma50 > f.sma200) & (f.close < f.sma10 * (1 - p / 100)),
              lambda f: f.sma10 / f.close),
    "bbands": ("Close below lower Bollinger band, above 200-day avg, RSI(14) < {p:g}",
               lambda f, p: (f.close < f.bb_low) & (f.close > f.sma200) & (f.rsi14 < p),
               lambda f: -f.rsi14),
    "breakout": ("20-day high breakout on {p:g}x volume, above 50-day avg",
                 lambda f, p: (f.close > f.hh20) & (f.vol_ratio > p) & (f.close > f.sma50),
                 lambda f: f.vol_ratio),
    "rspull": ("Top-30% 6-month strength, pulled back below 10-day avg, RSI(2) < {p:g}",
               lambda f, p: f.rs_top & (f.close < f.sma10) & (f.close > f.sma50) & (f.rsi2 < p),
               lambda f: f.ret126),
}

GRID = {
    "rsi2": [5, 10, 20],
    "down3": [0, 2, 4],
    "bbands": [100, 40, 30],
    "breakout": [1.5, 2.0, 3.0],
    "rspull": [10, 25, 50],
}
MARKETS = ["none", "nifty50", "nifty200"]
STOPS = [0.0, 0.03, 0.05, 0.08]


# ---------------------------------------------------------------- simulation
def trade_outcome(o, h, l, c, bad, i, rule: Rule):
    """Enter at open of bar i. Returns (exit_bar, exit_price, reason) or None."""
    n = len(o)
    entry = o[i]
    tgt = entry * (1 + rule.target)
    stp = entry * (1 - rule.stop) if rule.stop else -np.inf
    last = min(i + rule.max_days - 1, n - 1)
    low = entry  # worst price seen while holding
    for j in range(i, last + 1):
        if j > i and bad[j]:  # demerger / bad print: sell at the previous close
            return j - 1, c[j - 1], "gap", low / entry - 1
        if j > i:  # gaps through levels fill at the open
            if o[j] <= stp:
                return j, o[j], "stop", min(low, o[j]) / entry - 1
            if o[j] >= tgt:
                return j, o[j], "target", low / entry - 1
        if l[j] <= stp:
            return j, stp, "stop", stp / entry - 1
        if h[j] >= tgt:
            return j, tgt, "target", low / entry - 1
        low = min(low, l[j])
    if last < i + rule.max_days - 1:  # still open when the data ends: value at last close
        return last, c[last], "open", low / entry - 1
    return last, c[last], "time", low / entry - 1


def build_panel(stocks, nifty):
    feats = {s: features(d) for s, d in stocks.items()}
    dates = sorted(set().union(*[f.index for f in feats.values()]))
    ret126 = pd.DataFrame({s: f.ret126 for s, f in feats.items()}).reindex(dates)
    rs_rank = ret126.rank(axis=1, pct=True)
    for s, f in feats.items():
        f["rs_top"] = rs_rank[s].reindex(f.index) >= 0.7
    nc = nifty.close.reindex(dates).ffill()
    mkt = pd.DataFrame({"nifty50": nc > nc.rolling(50).mean(), "nifty200": nc > nc.rolling(200).mean(),
                        "none": True}, index=dates)
    return feats, mkt


def signals(feats, mkt, stocks, rule: Rule) -> pd.DataFrame:
    """All candidate trades for a rule (one per stock-signal day), with outcomes."""
    _, cond, score = SETUPS[rule.setup]
    rows = []
    for s, f in feats.items():
        d = stocks[s]
        sig = cond(f, rule.param) & mkt[rule.market].reindex(f.index).fillna(False).astype(bool)
        sig &= (f.turnover > 5e7) & f.mcap_ok  # Rs 5 cr traded a day, mcap > Rs 5,000 cr
        idx = np.flatnonzero(sig.to_numpy())
        if not len(idx):
            continue
        o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
        bad = f.bad_gap.to_numpy()
        sc = score(f).to_numpy()
        for k in idx:
            i = k + 1
            if i >= len(o) or bad[i]:
                continue
            out = trade_outcome(o, h, l, c, bad, i, rule)
            if out is None:
                continue
            j, px, why, mae = out
            rows.append((s, f.index[k], f.index[i], f.index[j], o[i], px, why, j - i + 1, mae, sc[k]))
    t = pd.DataFrame(rows, columns=["symbol", "signal_date", "entry_date", "exit_date", "entry", "exit",
                                    "reason", "days", "mae", "score"])
    return t


def portfolio(cands: pd.DataFrame, stocks, dates, slots: int, start, end):
    """Rs 5L split into equal slots; take the best-scored signals while slots are free."""
    cands = cands[(cands.entry_date >= start) & (cands.entry_date <= end)]
    by_day = {d: g.sort_values("score", ascending=False) for d, g in cands.groupby("entry_date")}
    cash = CAPITAL
    open_pos = []  # dicts
    trades, equity = [], []
    last_day = max(d for d in dates if d <= end)
    closes = {s: d.close for s, d in stocks.items()}
    held = set()
    for d in dates:
        if d < start or d > end:
            continue
        # exits scheduled today happen before new entries are counted for tomorrow
        if d in by_day:
            for _, t in by_day[d].iterrows():
                if len(open_pos) >= slots:
                    break
                if t.symbol in held:
                    continue
                eq_now = cash + sum(p["qty"] * closes[p["symbol"]].get(d, p["entry"]) for p in open_pos)
                alloc = min(cash, eq_now / slots)
                buy_px = t.entry * (1 + SLIPPAGE)
                qty = int(alloc // buy_px)
                if qty <= 0 or alloc < 10_000:
                    continue
                cash -= qty * buy_px
                open_pos.append(dict(symbol=t.symbol, qty=qty, entry=t.entry, buy_px=buy_px,
                                     exit_date=t.exit_date, exit=t.exit, reason=t.reason,
                                     entry_date=d, days=t.days, mae=t.mae, signal_date=t.signal_date))
                held.add(t.symbol)
        still = []
        for p in open_pos:
            if p["exit_date"] == d or d == last_day:  # anything still held is valued at the end
                if p["exit_date"] != d:
                    p["exit"], p["reason"] = closes[p["symbol"]].get(d, p["entry"]), "open"
                    p["days"] = int(((closes[p["symbol"]].index >= p["entry_date"]) &
                                     (closes[p["symbol"]].index <= d)).sum())
                sell_px = p["exit"] * (1 - SLIPPAGE)
                bv, sv = p["qty"] * p["buy_px"], p["qty"] * sell_px
                cost = round_trip_cost(bv, sv)
                pnl = sv - bv - cost
                cash += sv - cost
                held.discard(p["symbol"])
                trades.append(dict(symbol=p["symbol"], signal_date=p["signal_date"], entry_date=p["entry_date"],
                                   exit_date=d, entry=round(p["entry"], 2), exit=round(p["exit"], 2),
                                   qty=p["qty"], reason=p["reason"], days=p["days"], mae=round(p["mae"], 4),
                                   invested=round(bv),
                                   pnl=round(pnl), ret=pnl / bv))
            else:
                still.append(p)
        open_pos = still
        mtm = sum(p["qty"] * closes[p["symbol"]].get(d, p["entry"]) for p in open_pos)
        equity.append((d, cash + mtm, len(open_pos)))
    eq = pd.DataFrame(equity, columns=["date", "equity", "positions"]).set_index("date")
    return pd.DataFrame(trades), eq


def stats(trades: pd.DataFrame, eq: pd.DataFrame) -> dict:
    if trades.empty or eq.empty:
        return dict(trades=0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.equity.iloc[-1] / CAPITAL - 1
    dd = (eq.equity / eq.equity.cummax() - 1).min()
    wins = trades[trades.pnl > 0]
    loss = trades[trades.pnl <= 0]
    return dict(
        trades=int(len(trades)),
        win_rate=float(len(wins) / len(trades)),
        target_hit=float((trades.reason == "target").mean()),
        open_at_end=int((trades.reason == "open").sum()),
        open_avg_ret=float(trades[trades.reason == "open"].ret.mean()) if (trades.reason == "open").any() else 0.0,
        avg_ret=float(trades.ret.mean()),
        avg_win=float(wins.ret.mean()) if len(wins) else 0.0,
        avg_loss=float(loss.ret.mean()) if len(loss) else 0.0,
        worst=float(trades.ret.min()),
        avg_days=float(trades.days.mean()),
        total_ret=float(total),
        cagr=float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0,
        max_dd=float(dd),
        avg_positions=float(eq.positions.mean()),
        full_time=float((eq.positions >= eq.positions.max()).mean()),
        median_days=float(trades.days.median()),
        over15=float((trades.days > 15).mean()),
        over60=float((trades.days > 60).mean()),
        avg_mae=float(trades.mae.mean()),
        worst_mae=float(trades.mae.min()),
        mae_over10=float((trades.mae < -0.10).mean()),
        pnl=float(trades.pnl.sum()),
    )


def nifty_bh(nifty, start, end):
    s = nifty.close[(nifty.index >= start) & (nifty.index <= end)]
    eq = s / s.iloc[0] * CAPITAL
    years = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] / s.iloc[0] - 1
    return eq, dict(total_ret=float(total), cagr=float((1 + total) ** (1 / years) - 1),
                    max_dd=float((s / s.cummax() - 1).min()))


# ---------------------------------------------------------------- main
def main():
    global TRAIN_END, TEST_START
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("swing/results"))
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--train-end", default=str(TRAIN_END.date()))
    ap.add_argument("--start", default=None)
    ap.add_argument("--max-days", type=int, default=15)
    ap.add_argument("--stops", default="0,0.03,0.05,0.08")
    a = ap.parse_args()
    TRAIN_END = pd.Timestamp(a.train_end)
    TEST_START = TRAIN_END + pd.Timedelta(days=1)
    a.out.mkdir(parents=True, exist_ok=True)

    stocks, nifty = load(a.data)
    feats, mkt = build_panel(stocks, nifty)
    dates = list(mkt.index)
    first = max(dates[200], pd.Timestamp(a.start)) if a.start else dates[200]
    last = min(max(d.index[-1] for d in stocks.values()), nifty.index[-1])
    print(f"{len(stocks)} stocks, {first.date()} .. {last.date()}")

    stops = [float(x) for x in a.stops.split(",")]
    rules = [Rule(s, p, m, st, max_days=a.max_days)
             for s in GRID for p, m, st in itertools.product(GRID[s], MARKETS, stops)]
    grid_rows, cache = [], {}
    for r in rules:
        c = signals(feats, mkt, stocks, r)
        cache[r] = c
        tr_t, tr_e = portfolio(c, stocks, dates, a.slots, first, TRAIN_END)
        te_t, te_e = portfolio(c, stocks, dates, a.slots, TEST_START, last)
        row = dict(rule=r.name, **{k: v for k, v in asdict(r).items()})
        row.update({f"train_{k}": v for k, v in stats(tr_t, tr_e).items()})
        row.update({f"test_{k}": v for k, v in stats(te_t, te_e).items()})
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(a.out / "grid.csv", index=False)

    # pick on TRAIN only: best CAGR / drawdown, with enough trades
    ok = grid[(grid.train_trades >= 80)].copy()
    ok["train_score"] = ok.train_cagr / ok.train_max_dd.abs().clip(lower=0.05)
    best = ok.sort_values("train_score", ascending=False).iloc[0]
    rule = Rule(best.setup, best.param, best.market, best.stop, max_days=a.max_days)
    print("chosen on train:", rule.name)

    c = cache[rule]
    out = {"rule": rule.name, "rule_fields": asdict(rule), "slots": a.slots, "capital": CAPITAL,
           "period": {"first": str(first.date()), "train_end": str(TRAIN_END.date()),
                      "test_start": str(TEST_START.date()), "last": str(last.date())},
           "stocks": len(stocks)}
    for tag, s0, s1 in [("train", first, TRAIN_END), ("test", TEST_START, last)]:
        t, e = portfolio(c, stocks, dates, a.slots, s0, s1)
        neq, nst = nifty_bh(nifty, s0, s1)
        out[tag] = stats(t, e)
        out[f"{tag}_nifty"] = nst
        t.to_csv(a.out / f"trades_{tag}.csv", index=False)
        pd.DataFrame({"strategy": e.equity, "nifty": neq.reindex(e.index).ffill(),
                      "positions": e.positions}).to_csv(a.out / f"equity_{tag}.csv")
    (a.out / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
