"""Plain-language reasons and a transparent points score.

The ML model decides *whether* to alert. These rules explain *why* in words
a trader recognises, and the points score doubles as a simple baseline the
model has to beat in the backtest.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _v(row, k, default=np.nan):
    x = row.get(k, default) if hasattr(row, "get") else default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def reasons(row) -> tuple[float, list[str], float]:
    """Return (points 0..100, reasons, direction score -1..+1)."""
    pts, why, votes = 0.0, [], []

    # --- coiled spring: volatility compression ---
    r = _v(row, "bb_width_rank")
    if _ok(r) and r <= 0.10:
        pts += 12; why.append(f"Bollinger bands at their tightest {r:.0%} of the last 6 months (squeeze)")
    elif _ok(r) and r <= 0.20:
        pts += 7; why.append("Bollinger bands unusually tight")
    sd = _v(row, "squeeze_days")
    if _ok(sd) and sd >= 5:
        pts += 6; why.append(f"TTM squeeze on for {int(sd)} days (Bollinger inside Keltner)")
    if _v(row, "squeeze_fired") == 1:
        pts += 6; why.append(f"Squeeze just released after {int(_v(row, 'squeeze_fired_len', 0))} days")
    ar = _v(row, "atr_ratio_14_50")
    if _ok(ar) and ar <= 0.75:
        pts += 5; why.append(f"ATR contracted to {ar:.0%} of its 50-day level")
    if _v(row, "nr7") == 1:
        pts += 3; why.append("NR7: narrowest daily range of the last 7 days")

    # --- ignition: volume, range and breakouts ---
    rv = _v(row, "rvol")
    if _ok(rv) and rv >= 3:
        pts += 12; why.append(f"Volume {rv:.1f}x the 20-day average")
    elif _ok(rv) and rv >= 2:
        pts += 8; why.append(f"Volume {rv:.1f}x the 20-day average")
    tr = _v(row, "tr_atr")
    if _ok(tr) and tr >= 2:
        pts += 6; why.append(f"Today's range {tr:.1f}x the normal (ATR)")
    if _v(row, "brk_up55") == 1:
        pts += 8; votes.append(1.5); why.append("Closed above its 55-day high (breakout)")
    elif _v(row, "brk_up20") == 1:
        pts += 5; votes.append(1.0); why.append("Closed above its 20-day high")
    if _v(row, "brk_dn55") == 1:
        pts += 8; votes.append(-1.5); why.append("Closed below its 55-day low (breakdown)")
    elif _v(row, "brk_dn20") == 1:
        pts += 5; votes.append(-1.0); why.append("Closed below its 20-day low")

    # --- options market ---
    ivr = _v(row, "iv_rank")
    if _ok(ivr) and ivr >= 0.8:
        pts += 8; why.append(f"Implied volatility in the top {1 - ivr:.0%} of its 1-year range")
    ivc = _v(row, "iv_chg5")
    if _ok(ivc) and ivc >= 0.2:
        pts += 6; why.append(f"Implied volatility up {ivc:.0%} in 5 days (options traders expect a move)")
    orv = _v(row, "opt_rvol")
    if _ok(orv) and orv >= 2.5:
        pts += 8; why.append(f"Option volume {orv:.1f}x normal")
    oi = _v(row, "fut_oi_chg")
    ret = _v(row, "ret1")
    if _ok(oi) and _ok(ret) and abs(oi) >= 0.05 and abs(ret) >= 0.005:
        pts += 8
        if ret > 0 and oi > 0:
            votes.append(1.5); why.append(f"Long build-up: price +{ret:.1%}, futures OI +{oi:.0%}")
        elif ret < 0 and oi > 0:
            votes.append(-1.5); why.append(f"Short build-up: price {ret:.1%}, futures OI +{oi:.0%}")
        elif ret > 0 and oi < 0:
            votes.append(1.0); why.append(f"Short covering: price +{ret:.1%}, futures OI {oi:.0%}")
        else:
            votes.append(-1.0); why.append(f"Long unwinding: price {ret:.1%}, futures OI {oi:.0%}")
    wb = _v(row, "near_writing_bias")
    noc = _v(row, "near_oi_chg")
    if _ok(wb) and _ok(noc) and abs(wb) >= 0.5 and noc >= 0.05:
        pts += 5
        if wb > 0:
            votes.append(1.0); why.append("Heavy fresh put writing near the price (support building)")
        else:
            votes.append(-1.0); why.append("Heavy fresh call writing near the price (resistance building)")
    pcr = _v(row, "pcr_oi")
    if _ok(pcr) and pcr >= 1.5:
        votes.append(0.5); why.append(f"Put-call ratio high at {pcr:.2f}")
    elif _ok(pcr) and pcr <= 0.45:
        votes.append(-0.5); why.append(f"Put-call ratio low at {pcr:.2f}")

    # --- momentum / divergence ---
    if _v(row, "rsi_div") == 1 or _v(row, "macd_div") == 1:
        pts += 5; votes.append(1.0); why.append("Bullish divergence (price lower low, momentum higher low)")
    if _v(row, "rsi_div") == -1 or _v(row, "macd_div") == -1:
        pts += 5; votes.append(-1.0); why.append("Bearish divergence (price higher high, momentum lower high)")
    adx, slope = _v(row, "adx"), _v(row, "adx_slope5")
    if _ok(adx) and _ok(slope) and 18 <= adx <= 28 and slope >= 4:
        pts += 4; why.append(f"ADX rising from a low base ({adx:.0f}): a trend may be starting")
    mc = _v(row, "macd_cross")
    if mc == 1:
        votes.append(0.5)
    elif mc == -1:
        votes.append(-0.5)
    c20 = _v(row, "c_ema20"); c50 = _v(row, "c_ema50")
    if _ok(c20) and _ok(c50):
        votes.append(0.5 if (c20 > 0 and c50 > 0) else -0.5 if (c20 < 0 and c50 < 0) else 0.0)

    # --- scheduled events ---
    dtr = _v(row, "days_to_results")
    if _ok(dtr) and dtr <= 7:
        pts += 10; why.append(f"Quarterly results due in about {int(dtr)} days")

    direction = float(np.clip(np.sum(votes) / 4.0, -1, 1)) if votes else 0.0
    return min(pts, 100.0), why, direction


def rule_scores(df: pd.DataFrame) -> pd.Series:
    return pd.Series([reasons(r)[0] for r in df.to_dict("records")], index=df.index)
