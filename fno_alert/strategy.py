"""Turn a big-move alert into an options trade idea, and test those ideas
on real historical option prices."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TradeRules:
    dir_conf: float = 0.62        # p_up/(p_up+p_dn) needed to call a direction
    iv_rich_rank: float = 0.75    # IV rank at/above this = options expensive
    iv_rich_ratio: float = 1.6    # or IV / realised vol at/above this
    sell_otm_sigma: float = 0.6   # short strike distance, in expected moves to expiry
    min_sell_otm: float = 0.03
    hold_days: int = 5
    slippage: float = 0.015       # per side, fraction of option price
    stt_sell: float = 0.001       # STT on option premium when selling
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


def _strike_for(action_leg: str, F: float, strikes: np.ndarray, iv: float, dte: int, rules: TradeRules) -> float:
    strikes = np.sort(strikes)
    if action_leg.startswith("BUY"):
        return float(strikes[np.argmin(np.abs(strikes - F))])
    em = iv * math.sqrt(max(dte, 1) / 365) if iv == iv else 0.05
    dist = max(rules.min_sell_otm, rules.sell_otm_sigma * em)
    if action_leg == "SELL PE":
        cand = strikes[strikes <= F * (1 - dist)]
        return float(cand.max()) if len(cand) else math.nan
    cand = strikes[strikes >= F * (1 + dist)]
    return float(cand.min()) if len(cand) else math.nan


class ContractBook:
    """Fast lookups into fo_contracts (date, symbol, expiry, strike, opt, close, settle, volume, oi)."""

    def __init__(self, contracts: pd.DataFrame):
        c = contracts.copy()
        c["symbol"] = c["symbol"].astype(str)
        c["opt"] = c["opt"].astype(str)
        self.c = c.set_index(["symbol", "expiry", "opt", "strike", "date"]).sort_index()
        self.chain_idx = c.groupby(["symbol", "date", "expiry"])["strike"].unique()

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


def simulate(alerts: pd.DataFrame, book: ContractBook, trading_days: np.ndarray,
             rules: TradeRules = TradeRules()) -> pd.DataFrame:
    """alerts needs: date, symbol, action, F, focus_expiry, atm_iv, dte, lot.
    Enter at that day's closing option price, exit `hold_days` trading days
    later at the close (settlement price if it did not trade, or on expiry)."""
    td = pd.DatetimeIndex(trading_days)
    out = []
    for a in alerts.itertuples(index=False):
        if a.action == "NO TRADE" or not (a.F == a.F):
            continue
        legs = ["BUY CE", "BUY PE"] if a.action == "BUY CE + BUY PE" else [a.action]
        exp = pd.Timestamp(a.focus_expiry)
        d0 = pd.Timestamp(a.date)
        i0 = td.searchsorted(d0)
        if i0 + rules.hold_days >= len(td):
            continue
        d_exit = min(td[i0 + rules.hold_days], exp)
        strikes = book.strikes(a.symbol, d0, exp)
        if len(strikes) == 0:
            continue
        pnl, prem, ok, notes = 0.0, 0.0, True, []
        for leg in legs:
            opt = leg[-2:]
            K = _strike_for(leg, a.F, strikes, a.atm_iv, a.dte, rules)
            p = book.path(a.symbol, exp, opt, K) if K == K else pd.DataFrame()
            if p.empty or d0 not in p.index or not (p.at[d0, "close"] == p.at[d0, "close"]):
                ok = False; break          # no traded price on entry day: not fillable
            entry = float(p.at[d0, "close"])
            after = p[(p.index > d0) & (p.index <= d_exit)]
            if after.empty:
                ok = False; break
            last = after.iloc[-1]
            exit_px = float(last["close"]) if (last["close"] == last["close"]) and last.name < exp else float(last["settle"])
            if not (exit_px == exit_px):
                ok = False; break
            if leg.startswith("BUY"):
                cost_in = entry * (1 + rules.slippage)
                cost_out = exit_px * (1 - rules.slippage) * (1 - rules.stt_sell)
                pnl += cost_out - cost_in
                prem += cost_in
            else:
                credit = entry * (1 - rules.slippage) * (1 - rules.stt_sell)
                debit = exit_px * (1 + rules.slippage)
                pnl += credit - debit
                prem += entry
            notes.append(f"{leg} {K:g}")
        if not ok:
            continue
        lot = a.lot if a.lot == a.lot else np.nan
        selling = legs[0].startswith("SELL")
        capital = rules.margin_frac * a.F if selling else prem
        out.append({"date": d0, "symbol": a.symbol, "action": a.action, "legs": ", ".join(notes),
                    "exit_date": pd.Timestamp(last.name), "pnl_per_share": pnl, "premium": prem,
                    "ret_on_capital": pnl / capital if capital else np.nan,
                    "pnl_per_lot": pnl * lot if lot == lot else np.nan})
    return pd.DataFrame(out)


def summarise_trades(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()
    def agg(g):
        r = g["ret_on_capital"]
        wins = g.loc[g["pnl_per_share"] > 0, "pnl_per_share"].sum()
        losses = -g.loc[g["pnl_per_share"] < 0, "pnl_per_share"].sum()
        return pd.Series({"trades": len(g), "win_rate": (g["pnl_per_share"] > 0).mean(),
                          "avg_return": r.mean(), "median_return": r.median(),
                          "profit_factor": wins / losses if losses > 0 else np.inf,
                          "avg_pnl_per_lot": g["pnl_per_lot"].mean(),
                          "total_pnl_per_lot": g["pnl_per_lot"].sum()})
    return tr.groupby("action").apply(agg, include_groups=False)
