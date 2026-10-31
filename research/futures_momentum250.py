"""250-day momentum traded with real NSE stock futures (stocks only, no index).

Uses the actual futures contracts in the Upstox download (history-data branch, fo/*/*.parquet):
real daily closes, real lot sizes, a roll on the trading day before each monthly expiry.

Rules (fixed before looking at results):
  * Signal: return over the past 250 trading days, from adjusted cash closes, known at the
    close of the day BEFORE the roll (no look-ahead). Two flavours:
      - "250d"    : plain 250-day return
      - "250d-1m" : 250-day return skipping the latest 21 days (the classic 12-1 momentum)
  * Universe: stocks that have a futures contract on the roll day and 260+ days of prices.
  * On each roll day: close all positions in the expiring contract, open new ones in the next
    month's contract, at that day's close. One lot per stock.
  * Costs per side: Rs 20 brokerage, exchange 0.00173%, SEBI 0.0001%, GST 18% on those,
    stamp 0.002% on buys, STT on sells (0.0125% to Sep 2024, 0.02% to Mar 2026, 0.05% from
    1 Apr 2026, Budget 2026), plus slippage (0.05% base case, 0.10% stress case).
  * Margin estimate: 20% of notional per lot (stock futures SPAN + exposure is ~15-30%).
    Capital = 2x peak margin, so at least half the account is always unused.

Run:  python research/futures_momentum250.py /path/to/history-data  (needs data/processed/ohlcv.parquet)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
HD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hd"
MARGIN = 0.20

# ---------- data
o = pd.read_parquet("data/processed/ohlcv.parquet")
o["date"] = pd.to_datetime(o["date"])
px = o.pivot_table(index="date", columns="symbol", values="close").sort_index()
nifty = pd.read_csv(os.path.join(HD, "index/nifty.csv"), parse_dates=["date"]).set_index("date")["close"]

cache = "data/processed/futures_only.parquet"
if os.path.exists(cache):
    fut = pd.read_parquet(cache)
else:
    parts = []
    for f in glob.glob(os.path.join(HD, "fo/*/*.parquet")):
        d = pd.read_parquet(f, columns=["date", "symbol", "inst", "expiry", "close", "settle", "volume", "lot"])
        parts.append(d[d["inst"] == "FUT"])
    fut = pd.concat(parts, ignore_index=True)
    fut["date"] = pd.to_datetime(fut["date"])
    fut["expiry"] = pd.to_datetime(fut["expiry"])
    fut["px"] = fut["close"].where(fut["close"] > 0, fut["settle"])
    fut = fut.dropna(subset=["px", "lot"]).drop_duplicates(["date", "symbol", "expiry"], keep="last")
    fut.to_parquet(cache, index=False)


def split_adjust(fut, px):
    """Upstox expired-futures rows carry today's lot size on every day, while prices before a
    split/bonus are unadjusted. Back-adjust each contract's prices to its final (post-split)
    units: where a futures day move differs from the adjusted cash move by >15%, use the cash move."""
    fut = fut.sort_values(["symbol", "expiry", "date"]).copy()
    out = []
    for (sym, exp), g in fut.groupby(["symbol", "expiry"], sort=False):
        lf = np.log(g["px"].to_numpy(float))
        rf = np.diff(lf)
        if sym in px.columns:
            ce = np.log(px[sym].reindex(g["date"]).ffill().to_numpy(float))
            re_ = np.diff(ce)
            bad = np.abs(rf - re_) > 0.15
            rf = np.where(bad & ~np.isnan(re_), re_, rf)
        adj = lf[-1] - np.concatenate([np.cumsum(rf[::-1])[::-1], [0.0]])
        out.append(pd.Series(np.exp(adj), index=g.index))
    fut["apx"] = pd.concat(out)
    return fut


fut = split_adjust(fut, px)
fixed = (np.abs(np.log(fut["apx"] / fut["px"])) > 0.15).groupby([fut["symbol"], fut["expiry"]]).any()
print("contracts with a split/bonus fixed:", int(fixed.sum()))
fut["px"] = fut["apx"]
fut = fut.set_index(["symbol", "expiry", "date"]).sort_index()
days = pd.DatetimeIndex(sorted(fut.index.get_level_values("date").unique()))
expiries = sorted(fut.index.get_level_values("expiry").unique())
# monthly expiries = the most common expiry dates (one per month)
cnt = fut.reset_index().groupby("expiry")["symbol"].nunique()
expiries = [e for e in expiries if cnt[e] > 50]
rolls = []
for e in expiries:
    before = days[days < e]
    if len(before):
        rolls.append((before[-1], e))   # roll day, expiring contract


def stt_rate(d):
    if d < pd.Timestamp("2024-10-01"):
        return 0.000125
    if d < pd.Timestamp("2026-04-01"):
        return 0.0002
    return 0.0005


def side_cost(notional, d, buy, slip):
    exch = notional * (0.0000173 + 0.000001)
    c = 20 + exch + 0.18 * (20 + exch) + notional * slip
    c += notional * 0.00002 if buy else notional * stt_rate(d)
    return c


def price(sym, exp, d):
    try:
        r = fut.loc[(sym, exp, d)]
        return float(r["px"]), float(r["lot"])
    except KeyError:
        return None


lr = np.log(px)
SIG = {"250d": lr - lr.shift(250), "250d-1m": lr.shift(21) - lr.shift(250)}


def backtest(sig_name, n_long=10, n_short=0, slip=0.0005, nifty_filter=False, budget=None):
    """Returns daily equity P&L (Rs), trades table, and per-roll margin."""
    sig = SIG[sig_name]
    ma200 = nifty.rolling(200).mean()
    daily = pd.Series(0.0, index=days)
    trades, margins = [], []
    for k in range(len(rolls) - 1):
        d0, _ = rolls[k]
        d1, exp_hold = rolls[k + 1]          # hold next month's contract until its roll day
        prev = px.index[px.index < d0]
        if len(prev) < 260:
            continue
        s = sig.loc[prev[-1]].dropna()
        avail = [x for x in s.index if price(x, exp_hold, d0) and price(x, exp_hold, d1)]
        s = s[avail].sort_values(ascending=False)
        if len(s) < 50:
            continue
        longs = list(s.index[:n_long])
        if nifty_filter and nifty.asof(prev[-1]) < ma200.asof(prev[-1]):
            longs = []
        shorts = list(s.index[-n_short:]) if n_short else []
        if budget:  # small account: take top names while margin fits in half the capital
            picked, used = [], 0.0
            for x in s.index:
                p, lot = price(x, exp_hold, d0)
                m = p * lot * MARGIN
                if used + m <= budget / 2:
                    picked.append(x); used += m
                if len(picked) >= n_long:
                    break
            longs = picked if longs else []
        margin = 0.0
        for side, names in ((1, longs), (-1, shorts)):
            for x in names:
                p0, lot0 = price(x, exp_hold, d0)
                p1, lot1 = price(x, exp_hold, d1)
                n0, n1 = p0 * lot0, p1 * lot0      # prices are split-adjusted to the contract's final lot
                cost = side_cost(n0, d0, side > 0, slip) + side_cost(n1, d1, side < 0, slip)
                pnl = side * (n1 - n0) - cost
                margin += n0 * MARGIN
                trades.append(dict(entry=d0, exit=d1, symbol=x, side=side, notional=n0,
                                   gross=side * (n1 - n0), cost=cost, pnl=pnl))
                # daily mark-to-market for drawdowns
                path = fut.loc[(x, exp_hold)]["px"] * lot0
                path = path[(path.index >= d0) & (path.index <= d1)]
                dd = (side * path.astype(float).diff().fillna(0))
                daily.loc[dd.index] += dd.to_numpy(dtype=float)
                daily.loc[d0] -= side_cost(n0, d0, side > 0, slip)
                daily.loc[d1] -= side_cost(n1, d1, side < 0, slip)
        margins.append(margin)
    t = pd.DataFrame(trades)
    return daily, t, margins


def summary(name, daily, t, margins, budget=None):
    if t.empty:
        print(f"{name}: no trades"); return None
    start, end = t["entry"].min(), t["exit"].max()
    daily = daily[(daily.index >= start) & (daily.index <= end)]
    cap = budget or 2 * max(margins)
    eq = cap + daily.cumsum()
    dd = (eq / eq.cummax() - 1).min()
    yrs = (end - start).days / 365.25
    tot = daily.sum() / cap
    cagr = (1 + tot) ** (1 / yrs) - 1 if tot > -1 else -1
    by_m = t.groupby("exit")["pnl"].sum()
    n_ret = nifty.asof(end) / nifty.asof(start) - 1
    interest = 0.06 * yrs * cap / 2          # spare half of the account in a liquid fund at ~6%
    row = dict(strategy=name, capital_lakh=round(cap / 1e5, 1), months=len(by_m),
               total=f"{tot:+.0%}", with_interest=f"{tot + interest / cap:+.0%}", per_year=f"{cagr:+.0%}", maxDD=f"{dd:.0%}",
               win_months=f"{(by_m > 0).mean():.0%}", gross_lakh=round(t['gross'].sum() / 1e5, 1),
               costs_lakh=round(t['cost'].sum() / 1e5, 1), nifty=f"{n_ret:+.0%}")
    return row


if __name__ == "__main__":
    print(f"rolls: {rolls[0][0].date()} .. {rolls[-1][0].date()} ({len(rolls)})")
    rows, keep = [], {}
    configs = [
        ("A long top 10, 250d", dict(sig_name="250d")),
        ("B long top 10, 250d-1m", dict(sig_name="250d-1m")),
        ("C long top10 / short bottom10, 250d-1m", dict(sig_name="250d-1m", n_short=10)),
        ("D long top 10, 250d-1m, only if NIFTY>200MA", dict(sig_name="250d-1m", nifty_filter=True)),
        ("B with double slippage (0.10%)", dict(sig_name="250d-1m", slip=0.001)),
        ("E Rs 5 lakh account, 250d-1m", dict(sig_name="250d-1m", budget=5e5)),
        ("F Rs 25 lakh account, 250d-1m", dict(sig_name="250d-1m", budget=25e5)),
    ]
    for name, kw in configs:
        daily, t, m = backtest(**kw)
        r = summary(name, daily, t, m, kw.get("budget"))
        if r:
            rows.append(r); keep[name] = (daily, t)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    # yearly split for the main variant
    for name in ("B long top 10, 250d-1m", "E Rs 5 lakh account, 250d-1m"):
        t = keep[name][1]
        print(f"\n{name}: P&L by year (Rs lakh)")
        print((t.groupby(t["exit"].dt.year)["pnl"].sum() / 1e5).round(2).to_string())
        t = t.assign(fret=t.gross / t.notional,
                     cret=[px[x].asof(b) / px[x].asof(a) - 1 for x, a, b in zip(t.symbol, t.entry, t.exit)])
        print("same picks bought as shares (no costs), sum of monthly avg return:",
              f"{t.groupby('exit').cret.mean().sum():+.1%}", " futures:", f"{t.groupby('exit').fret.mean().sum():+.1%}")
        print("avg names held:", round(t.groupby("entry").size().mean(), 1),
              " avg lot notional (lakh):", round(t["notional"].mean() / 1e5, 1))
    os.makedirs("research/out", exist_ok=True)
    res.to_csv("research/out/futures_momentum250.csv", index=False)
    keep["B long top 10, 250d-1m"][1].to_csv("research/out/futures_momentum250_trades.csv", index=False)
