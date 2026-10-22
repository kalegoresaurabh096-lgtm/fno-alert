"""Option chain and futures analytics for one stock on one day.

Input is a normalised snapshot so the same code serves the historical
bhavcopy files and the live NSE option chain:

options: DataFrame[expiry, strike, opt ('CE'|'PE'), close, oi, chg_oi, volume]
futures: DataFrame[expiry, close, oi, chg_oi, volume]

`close` must be NaN for contracts that did not trade that day.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from . import black76

MIN_DTE = 5          # calendar days; nearer expiries give noisy IV
NEAR_BAND = 0.05     # +-5% of the futures price counts as "near the money"


def _year_frac(trade_date: dt.date, expiry: dt.date) -> float:
    # Expiry at 15:30, valuation at today's close: count whole calendar days.
    return max((expiry - trade_date).days, 0.5) / 365.0


def pick_focus_expiry(expiries, trade_date: dt.date, min_dte: int = MIN_DTE):
    exps = sorted(e for e in set(expiries) if e >= trade_date)
    for e in exps:
        if (e - trade_date).days >= min_dte:
            return e
    return exps[-1] if exps else None


def max_pain(strikes: np.ndarray, ce_oi: np.ndarray, pe_oi: np.ndarray) -> float:
    """Strike at which option writers pay the least at expiry."""
    if len(strikes) == 0 or (ce_oi.sum() + pe_oi.sum()) <= 0:
        return math.nan
    K = strikes[:, None]
    s = strikes[None, :]
    pain = (ce_oi[None, :] * np.maximum(K - s, 0)).sum(1) + (pe_oi[None, :] * np.maximum(s - K, 0)).sum(1)
    return float(strikes[int(np.argmin(pain))])


def _iv_at(chain: pd.DataFrame, F: float, T: float) -> pd.DataFrame:
    traded = chain[(chain["volume"] > 0) & chain["close"].notna() & (chain["close"] > 0)].copy()
    if traded.empty:
        return traded.assign(iv=np.nan)
    traded["iv"] = black76.implied_vol(
        traded["close"].to_numpy(), F, traded["strike"].to_numpy(), T, (traded["opt"] == "CE").to_numpy()
    )
    return traded


def _atm_iv(ivs: pd.DataFrame, F: float) -> float:
    ivs = ivs.dropna(subset=["iv"])
    if ivs.empty:
        return math.nan
    per_strike = ivs.groupby("strike")["iv"].mean()
    strikes = per_strike.index.to_numpy(dtype=float)
    below = strikes[strikes <= F]
    above = strikes[strikes >= F]
    if len(below) and len(above):
        k1, k2 = below.max(), above.min()
        # Do not interpolate across a gap wider than 10% of F (illiquid chain).
        if (k2 - k1) > 0.10 * F:
            return math.nan
        if k1 == k2:
            return float(per_strike[k1])
        w = (F - k1) / (k2 - k1)
        return float((1 - w) * per_strike[k1] + w * per_strike[k2])
    nearest = strikes[np.argmin(np.abs(strikes - F))]
    if abs(nearest - F) > 0.03 * F:
        return math.nan
    return float(per_strike[nearest])


def _iv_near(ivs: pd.DataFrame, target: float, opt: str, tol: float) -> float:
    side = ivs[(ivs["opt"] == opt)].dropna(subset=["iv"])
    if side.empty:
        return math.nan
    i = (side["strike"] - target).abs().idxmin()
    if abs(side.at[i, "strike"] - target) > tol:
        return math.nan
    return float(side.at[i, "iv"])


def snapshot(options: pd.DataFrame, futures: pd.DataFrame, trade_date: dt.date,
             spot: float | None = None, min_dte: int = MIN_DTE) -> dict:
    """Summarise one stock's derivatives activity on `trade_date`."""
    out: dict = {}
    options = options.copy()
    futures = futures.copy()

    # ---- totals over the two nearest expiries (robust to monthly rollover;
    # the far month is tiny and the live scanner only fetches two chains) ----
    live_exps = sorted(e for e in options["expiry"].unique() if e >= trade_date)[:2]
    near2 = options[options["expiry"].isin(live_exps)]
    ce = near2[near2["opt"] == "CE"]
    pe = near2[near2["opt"] == "PE"]
    out["ce_oi"] = float(ce["oi"].sum())
    out["pe_oi"] = float(pe["oi"].sum())
    out["ce_chg_oi"] = float(ce["chg_oi"].sum())
    out["pe_chg_oi"] = float(pe["chg_oi"].sum())
    out["ce_vol"] = float(ce["volume"].sum())
    out["pe_vol"] = float(pe["volume"].sum())
    out["fut_oi"] = float(futures["oi"].sum())
    out["fut_chg_oi"] = float(futures["chg_oi"].sum())
    out["fut_vol"] = float(futures["volume"].sum())

    expiries = list(futures["expiry"]) + list(options["expiry"])
    focus = pick_focus_expiry(expiries, trade_date, min_dte)
    out["focus_expiry"] = focus
    if focus is None:
        return out
    out["dte"] = (focus - trade_date).days
    T = _year_frac(trade_date, focus)

    # Near-month futures close is the price we compare everything against.
    fut_sorted = futures[futures["expiry"] >= trade_date].sort_values("expiry")
    near = fut_sorted[fut_sorted["close"].notna()]
    out["fut_close"] = float(near["close"].iloc[0]) if len(near) else math.nan

    f_row = futures[(futures["expiry"] == focus) & futures["close"].notna()]
    if len(f_row):
        F = float(f_row["close"].iloc[0])
    elif spot:
        F = float(spot) * math.exp(black76.RISK_FREE * T)
    else:
        F = out["fut_close"]
    out["F"] = F
    if not F or math.isnan(F):
        return out

    chain = options[options["expiry"] == focus]
    if chain.empty:
        return out
    ivs = _iv_at(chain, F, T)
    atm_iv = _atm_iv(ivs, F)
    out["atm_iv"] = atm_iv
    out["put_iv_95"] = _iv_near(ivs, 0.95 * F, "PE", 0.03 * F)
    out["call_iv_105"] = _iv_near(ivs, 1.05 * F, "CE", 0.03 * F)
    out["skew"] = out["put_iv_95"] - out["call_iv_105"]

    # ATM straddle as a % of price: the move the option market is pricing in.
    strikes = np.sort(chain["strike"].unique())
    k_atm = float(strikes[np.argmin(np.abs(strikes - F))])
    out["atm_strike"] = k_atm
    atm = chain[(chain["strike"] == k_atm) & (chain["volume"] > 0) & chain["close"].notna()]
    if set(atm["opt"]) == {"CE", "PE"}:
        out["straddle_pct"] = float(atm["close"].sum()) / F
    else:
        out["straddle_pct"] = math.nan

    agg = chain.pivot_table(index="strike", columns="opt", values=["oi", "chg_oi"], aggfunc="sum", fill_value=0)
    k = agg.index.to_numpy(dtype=float)
    ce_oi = agg[("oi", "CE")].to_numpy(float) if ("oi", "CE") in agg else np.zeros(len(k))
    pe_oi = agg[("oi", "PE")].to_numpy(float) if ("oi", "PE") in agg else np.zeros(len(k))
    ce_chg = agg[("chg_oi", "CE")].to_numpy(float) if ("chg_oi", "CE") in agg else np.zeros(len(k))
    pe_chg = agg[("chg_oi", "PE")].to_numpy(float) if ("chg_oi", "PE") in agg else np.zeros(len(k))

    out["max_pain"] = max_pain(k, ce_oi, pe_oi)
    above, below = k >= F, k <= F
    out["call_wall"] = float(k[above][np.argmax(ce_oi[above])]) if above.any() and ce_oi[above].max() > 0 else math.nan
    out["put_wall"] = float(k[below][np.argmax(pe_oi[below])]) if below.any() and pe_oi[below].max() > 0 else math.nan
    nb = np.abs(k - F) <= NEAR_BAND * F
    out["ce_oi_near"] = float(ce_oi[nb].sum())
    out["pe_oi_near"] = float(pe_oi[nb].sum())
    out["ce_chg_near"] = float(ce_chg[nb].sum())
    out["pe_chg_near"] = float(pe_chg[nb].sum())
    out["focus_ce_oi"] = float(ce_oi.sum())
    out["focus_pe_oi"] = float(pe_oi.sum())
    return out
