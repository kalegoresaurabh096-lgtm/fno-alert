"""For every stock-day, what a 5-day ATM option buy entered at the next open
would have returned after costs (CE, PE and the straddle), plus a straddle-based
IV term-structure feature. Same execution rules as strategy.simulate."""
from __future__ import annotations

import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from fno_alert import strategy  # noqa: E402

R = strategy.TradeRules()


def leg_prices(c: pd.DataFrame):
    """dict[(sym, expiry, opt, strike)] -> Series(date -> (open, close/settle))."""
    c = c.sort_values("date")
    out = {}
    for k, g in c.groupby(["symbol", "expiry", "opt", "strike"], sort=False):
        out[k] = (g["date"].to_numpy(), g["open"].to_numpy(float),
                  np.where(np.isfinite(g["close"].to_numpy(float)), g["close"].to_numpy(float), g["settle"].to_numpy(float)))
    return out


def _px(arrs, d, which):
    dates, op, cl = arrs
    i = np.searchsorted(dates, d)
    if i < len(dates) and dates[i] == d:
        v = op[i] if which == "open" else cl[i]
        return v if v == v else math.nan
    return math.nan


def build(panel: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    c = contracts.copy()
    c["date"] = pd.to_datetime(c["date"]).values.astype("datetime64[ns]")
    c["expiry"] = pd.to_datetime(c["expiry"]).values.astype("datetime64[ns]")
    lp = leg_prices(c)
    chain = c.groupby(["symbol", "date", "expiry"])["strike"].unique()
    exps = c.groupby(["symbol", "date"])["expiry"].unique()
    td = np.sort(pd.to_datetime(panel["date"]).unique()).astype("datetime64[ns]")
    lot_by = panel.set_index(["symbol", "date"])["lot"] if "lot" in panel else None
    rows = []
    for r in panel[["symbol", "date", "F", "fut_close"]].itertuples(index=False):
        sym, d0 = r.symbol, np.datetime64(pd.Timestamp(r.date), "ns")
        rec = {"symbol": sym, "date": pd.Timestamp(d0), "ce_ret": np.nan, "pe_ret": np.nan, "strad_ret": np.nan,
               "term_slope": np.nan, "strad_cost_pct": np.nan}
        i0 = np.searchsorted(td, d0)
        try:
            ex = sorted(exps.loc[(sym, d0)])
        except KeyError:
            rows.append(rec); continue
        F = r.F if r.F == r.F else r.fut_close
        if not (F == F and F > 0):
            rows.append(rec); continue
        # Term structure: straddle-implied vol of the two nearest usable expiries.
        ivs = []
        for e in ex:
            dte = (e - d0) / np.timedelta64(1, "D")
            if dte < 5:
                continue
            ks = chain.loc[(sym, d0, e)]
            k = float(ks[np.argmin(np.abs(ks - F))])
            ce, pe = lp.get((sym, e, "CE", k)), lp.get((sym, e, "PE", k))
            if ce is None or pe is None:
                continue
            s = _px(ce, d0, "close") + _px(pe, d0, "close")
            if s == s and s > 0:
                ivs.append(s / (0.8 * F * math.sqrt(dte / 365)))
            if len(ivs) == 2:
                break
        if len(ivs) == 2:
            rec["term_slope"] = ivs[0] / ivs[1] - 1
        if i0 + R.hold_days >= len(td):
            rows.append(rec); continue
        d_in = td[i0 + 1]
        exp = next((e for e in ex if (e - d0) / np.timedelta64(1, "D") >= R.min_trade_dte), None)
        if exp is None:
            rows.append(rec); continue
        before = td[td < exp]
        d_exit = min(td[i0 + R.hold_days], before[-1])
        if d_exit <= d_in:
            rows.append(rec); continue
        ks = chain.loc[(sym, d0, exp)]
        k = float(ks[np.argmin(np.abs(ks - F))])
        lot = float(lot_by.get((sym, pd.Timestamp(d0)), np.nan)) if lot_by is not None else np.nan
        legs = {}
        for opt in ("CE", "PE"):
            a = lp.get((sym, exp, opt, k))
            if a is None:
                continue
            e_ = _px(a, d_in, "open")
            x_ = _px(a, d_exit, "close")
            if not (e_ == e_ and e_ > 0):
                continue
            if not (x_ == x_):
                u = r.fut_close  # rare: no exit price; skip rather than guess
                continue
            pnl = (x_ - strategy.side_cost(x_, lot, False, R)) - (e_ + strategy.side_cost(e_, lot, True, R))
            legs[opt] = (pnl, e_)
        if "CE" in legs:
            rec["ce_ret"] = legs["CE"][0] / legs["CE"][1]
        if "PE" in legs:
            rec["pe_ret"] = legs["PE"][0] / legs["PE"][1]
        if len(legs) == 2:
            rec["strad_ret"] = (legs["CE"][0] + legs["PE"][0]) / (legs["CE"][1] + legs["PE"][1])
            rec["strad_cost_pct"] = (legs["CE"][1] + legs["PE"][1]) / F
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from fno_alert import panel as pn
    p = pn.build_panel("data/processed", 10.0)
    con = pd.read_parquet("data/processed/fo_contracts.parquet")
    print(len(p), len(con), list(con.columns), flush=True)
    out = build(p, con)
    out.to_parquet("research/trade_labels.parquet", index=False)
    print(out.describe().T.to_string())
