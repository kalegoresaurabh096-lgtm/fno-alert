"""Turn a big-move alert into an options trade idea, and test those ideas
on real historical option prices.

Execution model used in the backtest (kept deliberately pessimistic):
* The alert is computed after the close of day t, so the trade is entered at
  the **open of day t+1**, only if that contract actually traded that day.
* Exit at the close of day t+hold_days (settlement price if it did not trade),
  and always before expiry week ends: the trade uses the first expiry at
  least `min_trade_dte` calendar days away, and exits no later than the last
  session before expiry, so physical settlement never happens.
* Costs per side: max(slippage % of price, minimum slippage in rupees) plus
  brokerage, exchange, SEBI, GST, stamp duty and STT on sells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeRules:
    dir_conf: float = 0.62        # p_up/(p_up+p_dn) needed to call a direction (>1 disables)
    iv_rich_rank: float = 0.75    # IV rank at/above this = options expensive
    iv_rich_ratio: float = 1.6    # or IV / realised vol at/above this
    sell_otm_sigma: float = 0.6   # short strike distance, in expected moves to expiry
    min_sell_otm: float = 0.03
    hold_days: int = 5
    min_trade_dte: int = 12       # calendar days to expiry at entry
    slippage: float = 0.02        # per side, fraction of option price
    min_slippage: float = 0.15    # per side, rupees per share (wide spreads on cheap options)
    brokerage: float = 20.0       # rupees per order
    exch_fee: float = 0.000355    # NSE option transaction charge, on premium turnover
    sebi_fee: float = 0.000001
    gst: float = 0.18             # on brokerage + exchange + SEBI fees
    stamp_buy: float = 0.00003    # on buy-side premium
    stt_sell: float = 0.001       # on sell-side premium
    margin_frac: float = 0.20     # rough SPAN+exposure margin as share of notional

    def as_dict(self):
        return dict(self.__dict__)


ACTIONS = ("BUY CE", "BUY PE", "SELL PE", "SELL CE", "BUY CE + BUY PE", "NO TRADE")


def suggest(p_up: float, p_dn: float, iv_rank: float, iv_hv: float, rules: TradeRules = TradeRules()):
    """Return (action, direction label, reason)."""
    tot = (p_up or 0) + (p_dn or 0)
    conf = p_up / tot if tot > 0 else 0.5
    rich = ((iv_rank == iv_rank and iv_rank >= rules.iv_rich_rank) or
            (iv_hv == iv_hv and iv_hv >= rules.iv_rich_ratio))
    if conf >= rules.dir_conf:
        d = "UP"
    elif conf <= 1 - rules.dir_conf:
        d = "DOWN"
    else:
        d = "EITHER WAY"
    if d == "UP":
        return ("SELL PE", d, "Bullish, but options are already expensive: collect premium below support") if rich \
            else ("BUY CE", d, "Bullish and options are not expensive")
    if d == "DOWN":
        return ("SELL CE", d, "Bearish, but options are already expensive: collect premium above resistance") if rich \
            else ("BUY PE", d, "Bearish and options are not expensive")
    if rich:
        return ("NO TRADE", d, "Big move likely but direction unclear and options already price it in")
    return ("BUY CE + BUY PE", d, "Big move likely, direction unclear, options not expensive (straddle)")


def strike_for(leg: str, F: float, strikes: np.ndarray, iv: float, dte: int, rules: TradeRules) -> float:
    strikes = np.sort(np.asarray(strikes, dtype=float))
    if len(strikes) == 0:
        return math.nan
    if leg.startswith("BUY"):
        return float(strikes[np.argmin(np.abs(strikes - F))])
    em = iv * math.sqrt(max(dte, 1) / 365) if iv == iv else 0.05
    dist = max(rules.min_sell_otm, rules.sell_otm_sigma * em)
    if leg == "SELL PE":
        cand = strikes[strikes <= F * (1 - dist)]
        return float(cand.max()) if len(cand) else math.nan
    cand = strikes[strikes >= F * (1 + dist)]
    return float(cand.min()) if len(cand) else math.nan


_strike_for = strike_for  # backwards name


def pick_trade_expiry(expiries, d: pd.Timestamp, min_dte: int):
    exps = sorted(pd.Timestamp(e) for e in expiries if pd.Timestamp(e) >= d)
    for e in exps:
        if (e - d).days >= min_dte:
            return e
    return None


def side_cost(price: float, lot: float, buy: bool, r: TradeRules) -> float:
    """All-in cost of one side, in rupees per share."""
    slip = max(r.slippage * price, r.min_slippage)
    lot = lot if lot and lot == lot else 1.0
    turnover = price * lot
    fees = r.brokerage + r.exch_fee * turnover + r.sebi_fee * turnover
    fees *= 1 + r.gst
    fees += (r.stamp_buy * turnover) if buy else (r.stt_sell * turnover)
    return slip + fees / lot


class ContractBook:
    """Fast lookups into fo_contracts (date, symbol, expiry, strike, opt, open, close, settle, volume, oi)."""

    def __init__(self, contracts: pd.DataFrame, fut_close: pd.Series | None = None):
        c = contracts.copy()
        c["symbol"] = c["symbol"].astype(str)
        c["opt"] = c["opt"].astype(str)
        c["date"] = pd.to_datetime(c["date"])
        c["expiry"] = pd.to_datetime(c["expiry"])
        self.c = c.set_index(["symbol", "expiry", "opt", "strike", "date"]).sort_index()
        self.chain_idx = c.groupby(["symbol", "date", "expiry"])["strike"].unique()
        self.exp_idx = c.groupby(["symbol", "date"])["expiry"].unique()
        self.fut = fut_close  # Series indexed by (symbol, date): near-month futures close

    def expiries(self, sym, date):
        try:
            return list(self.exp_idx.loc[(sym, date)])
        except KeyError:
            return []

    def strikes(self, sym, date, expiry):
        try:
            return np.asarray(self.chain_idx.loc[(sym, date, expiry)], dtype=float)
        except KeyError:
            return np.array([])

    def path(self, sym, expiry, opt, strike) -> pd.DataFrame:
        try:
            return self.c.loc[(sym, expiry, opt, strike)]
        except KeyError:
            return pd.DataFrame()

    def underlying(self, sym, date) -> float:
        if self.fut is None:
            return math.nan
        try:
            return float(self.fut.loc[(sym, date)])
        except KeyError:
            return math.nan


def simulate(alerts: pd.DataFrame, book: ContractBook, trading_days, rules: TradeRules = TradeRules()) -> pd.DataFrame:
    """alerts needs: date, symbol, action, F, atm_iv, lot. Returns one row per attempted
    trade, including ones that could not be filled (status column)."""
    td = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days)))
    out = []
    for a in alerts.itertuples(index=False):
        if a.action == "NO TRADE":
            continue
        rec = {"date": pd.Timestamp(a.date), "symbol": a.symbol, "action": a.action, "status": "ok"}
        legs = ["BUY CE", "BUY PE"] if a.action == "BUY CE + BUY PE" else [a.action]
        d0 = pd.Timestamp(a.date)
        i0 = td.searchsorted(d0)
        if i0 + rules.hold_days >= len(td):
            continue
        d_in = td[i0 + 1]
        exp = pick_trade_expiry(book.expiries(a.symbol, d0), d0, rules.min_trade_dte)
        if exp is None:
            out.append({**rec, "status": "no expiry"}); continue
        last_before_exp = td[td < exp]
        if len(last_before_exp) == 0:
            continue
        d_exit = min(td[i0 + rules.hold_days], last_before_exp[-1])
        if d_exit <= d_in:
            out.append({**rec, "status": "too close to expiry"}); continue
        strikes = book.strikes(a.symbol, d0, exp)
        F0 = a.F if a.F == a.F else book.underlying(a.symbol, d0)
        dte = (exp - d0).days
        lot = a.lot if a.lot == a.lot else np.nan
        pnl, capital, legs_txt, exit_kind = 0.0, 0.0, [], "close"
        status = "ok"
        for leg in legs:
            opt = leg[-2:]
            K = strike_for(leg, F0, strikes, a.atm_iv, dte, rules)
            p = book.path(a.symbol, exp, opt, K) if K == K else pd.DataFrame()
            if p.empty or d_in not in p.index or not (p.at[d_in, "open"] == p.at[d_in, "open"]) or p.at[d_in, "open"] <= 0:
                status = "not filled"; break
            entry = float(p.at[d_in, "open"])
            if d_exit in p.index:
                row = p.loc[d_exit]
                px_exit = float(row["close"]) if row["close"] == row["close"] else float(row["settle"])
            else:
                # Contract missing from the file on exit day (strike far from price):
                # value it at intrinsic using the futures price, a slight underestimate.
                u = book.underlying(a.symbol, d_exit)
                if not (u == u):
                    status = "no exit price"; break
                px_exit = max(u - K, 0.0) if opt == "CE" else max(K - u, 0.0)
                exit_kind = "intrinsic"
            buy = leg.startswith("BUY")
            if buy:
                pnl += (px_exit - side_cost(px_exit, lot, False, rules)) - (entry + side_cost(entry, lot, True, rules))
                capital += entry
            else:
                pnl += (entry - side_cost(entry, lot, False, rules)) - (px_exit + side_cost(px_exit, lot, True, rules))
                capital += rules.margin_frac * F0
            legs_txt.append(f"{leg} {K:g} {exp:%d%b}")
        if status != "ok":
            out.append({**rec, "status": status}); continue
        out.append({**rec, "legs": ", ".join(legs_txt), "entry_date": d_in, "exit_date": d_exit,
                    "exit_kind": exit_kind, "pnl_per_share": pnl, "capital_per_share": capital,
                    "ret_on_capital": pnl / capital if capital else np.nan,
                    "pnl_per_lot": pnl * lot if lot == lot else np.nan})
    return pd.DataFrame(out)


def summarise_trades(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty or "pnl_per_share" not in tr:
        return pd.DataFrame()
    def agg(g):
        done = g[g["status"] == "ok"]
        r = done["ret_on_capital"]
        wins = done.loc[done["pnl_per_share"] > 0, "pnl_per_share"].sum()
        losses = -done.loc[done["pnl_per_share"] < 0, "pnl_per_share"].sum()
        return pd.Series({"attempted": len(g), "trades": len(done),
                          "not_fillable": int((g["status"] != "ok").sum()),
                          "win_rate": (done["pnl_per_share"] > 0).mean() if len(done) else np.nan,
                          "avg_return": r.mean(), "median_return": r.median(),
                          "worst_return": r.min() if len(r) else np.nan,
                          "profit_factor": wins / losses if losses > 0 else np.inf,
                          "avg_pnl_per_lot": done["pnl_per_lot"].mean(),
                          "total_pnl_per_lot": done["pnl_per_lot"].sum()})
    return tr.groupby("action").apply(agg, include_groups=False)
