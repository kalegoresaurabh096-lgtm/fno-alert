"""Feature engineering: one row per (stock, trading day), using only
information available at that day's close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

TRADING_DAYS = 252


def technical_features(px: pd.DataFrame) -> pd.DataFrame:
    """px: one stock, index=date (sorted), columns open, high, low, close, volume."""
    px = px.sort_index()
    o, h, l, c, v = (px[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=px.index)
    logret = np.log(c / c.shift(1))
    f["ret1"] = c / c.shift(1) - 1
    f["ret5"] = c / c.shift(5) - 1
    f["ret20"] = c / c.shift(20) - 1
    sd20 = logret.rolling(20).std()
    sd60 = logret.rolling(60).std()
    f["sigma20"] = sd20
    f["sigma60"] = sd60
    f["hv20"] = sd20 * np.sqrt(TRADING_DAYS)
    f["hv60"] = sd60 * np.sqrt(TRADING_DAYS)
    f["hv_ratio_10_60"] = logret.rolling(10).std() / sd60
    f["ret1_z"] = logret / sd20.shift(1)

    a14 = ind.atr(px, 14)
    a50 = ind.atr(px, 50)
    tr = ind.true_range(px)
    f["atr_pct"] = a14 / c
    f["atr_ratio_14_50"] = a14 / a50
    f["tr_atr"] = tr / a14.shift(1)              # today's range vs normal
    rng = h - l
    f["nr7"] = (rng == rng.rolling(7).min()).astype(float)
    f["inside_day"] = ((h <= h.shift(1)) & (l >= l.shift(1))).astype(float)
    f["gap"] = o / c.shift(1) - 1
    f["clv"] = ((c - l) - (h - c)) / rng.replace(0, np.nan)

    mid, bb_up, bb_lo = ind.bollinger(c, 20, 2.0)
    bbw = (bb_up - bb_lo) / mid
    f["bb_width"] = bbw
    f["bb_width_rank"] = ind.rolling_pct_rank(bbw, 126, 60)
    f["bb_pos"] = (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)
    _, kc_up, kc_lo = ind.keltner(px, 20, 1.5)
    sq_on = (bb_up < kc_up) & (bb_lo > kc_lo)
    f["squeeze_on"] = sq_on.astype(float)
    f["squeeze_days"] = ind.consecutive_true(sq_on).astype(float)
    f["squeeze_fired"] = ((~sq_on) & sq_on.shift(1, fill_value=False)).astype(float)
    # Length of the squeeze that just ended (0 if none fired today).
    f["squeeze_fired_len"] = f["squeeze_days"].shift(1).where(f["squeeze_fired"] > 0, 0.0)

    vol_avg20 = v.rolling(20).mean().shift(1)
    f["rvol"] = v / vol_avg20
    f["rvol5"] = v.rolling(5).mean() / v.rolling(50).mean().shift(5)
    f["dvol_cr"] = (c * v).rolling(20).mean() / 1e7  # avg daily turnover in Rs crore

    hi20 = h.rolling(20).max().shift(1)
    lo20 = l.rolling(20).min().shift(1)
    hi55 = h.rolling(55).max().shift(1)
    lo55 = l.rolling(55).min().shift(1)
    f["brk_up20"] = (c > hi20).astype(float)
    f["brk_dn20"] = (c < lo20).astype(float)
    f["brk_up55"] = (c > hi55).astype(float)
    f["brk_dn55"] = (c < lo55).astype(float)
    f["dist_hi20_atr"] = (hi20 - c) / a14
    f["dist_lo20_atr"] = (c - lo20) / a14
    hi252 = h.rolling(252, min_periods=120).max()
    lo252 = l.rolling(252, min_periods=120).min()
    f["pos_52w"] = (c - lo252) / (hi252 - lo252).replace(0, np.nan)
    rng20 = (h.rolling(20).max() - l.rolling(20).min())
    f["range20_atr"] = rng20 / a14   # tight multi-week base => small value

    r = ind.rsi(c, 14)
    f["rsi"] = r
    m_line, m_sig, m_hist = ind.macd(c)
    f["macd_hist_pct"] = m_hist / c
    cross_up = (m_line > m_sig) & (m_line.shift(1) <= m_sig.shift(1))
    cross_dn = (m_line < m_sig) & (m_line.shift(1) >= m_sig.shift(1))
    f["macd_cross"] = cross_up.astype(float) - cross_dn.astype(float)
    f["rsi_div"] = ind.divergence(c, r)
    f["macd_div"] = ind.divergence(c, m_hist)
    adx_, pdi, mdi = ind.adx(px, 14)
    f["adx"] = adx_
    f["adx_slope5"] = adx_ - adx_.shift(5)
    f["di_diff"] = pdi - mdi

    e20, e50, e200 = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
    f["c_ema20"] = c / e20 - 1
    f["c_ema50"] = c / e50 - 1
    f["c_ema200"] = c / e200 - 1
    f["ema20_slope5"] = e20 / e20.shift(5) - 1
    f["close"] = c
    return f


def option_features(snap: pd.DataFrame, tech: pd.DataFrame, close_raw: pd.Series | None = None) -> pd.DataFrame:
    """snap: one stock's daily derivatives snapshots, index=date (see chain.snapshot),
    already aligned to the stock's trading days (missing days = NaN rows).
    tech: the same stock's technical features (for realised vol and returns).
    close_raw: unadjusted cash close, for the futures basis."""
    snap = snap.sort_index()
    f = pd.DataFrame(index=snap.index)
    # Volumes compared over time in shares, so lot-size revisions do not look like spikes.
    lot = snap["lot"].ffill().bfill() if "lot" in snap else pd.Series(1.0, index=snap.index)
    lot = lot.fillna(1.0)
    iv = snap["atm_iv"]
    f["atm_iv"] = iv
    f["iv_rank"] = ind.rolling_pct_rank(iv, TRADING_DAYS, 60)
    f["iv_chg1"] = iv / iv.shift(1) - 1
    f["iv_chg5"] = iv / iv.shift(5) - 1
    f["iv_z60"] = (iv - iv.rolling(60, min_periods=30).mean()) / iv.rolling(60, min_periods=30).std()
    hv = tech["hv20"].reindex(f.index)
    f["iv_hv"] = iv / hv
    f["straddle_pct"] = snap["straddle_pct"]
    # Implied 5-day 1-sigma move, comparable to the label horizon.
    f["iv_move5"] = iv * np.sqrt(5 / TRADING_DAYS)
    f["skew"] = snap["skew"]
    f["skew_chg5"] = snap["skew"] - snap["skew"].shift(5)
    f["dte"] = snap["dte"]

    prev_fut_oi = snap["fut_oi"] - snap["fut_chg_oi"]
    f["fut_oi_chg"] = snap["fut_chg_oi"] / prev_fut_oi.replace(0, np.nan)
    f["fut_oi_chg5"] = snap["fut_oi"] / snap["fut_oi"].shift(5) - 1
    f["fut_oi_chg_z"] = (f["fut_oi_chg"] - f["fut_oi_chg"].rolling(60, min_periods=30).mean()) / f["fut_oi_chg"].rolling(60, min_periods=30).std()
    fv = snap["fut_vol"] * lot
    f["fut_rvol"] = fv / fv.rolling(20, min_periods=10).mean().shift(1)

    opt_oi = snap["ce_oi"] + snap["pe_oi"]
    prev_opt_oi = opt_oi - (snap["ce_chg_oi"] + snap["pe_chg_oi"])
    f["opt_oi_chg"] = (snap["ce_chg_oi"] + snap["pe_chg_oi"]) / prev_opt_oi.replace(0, np.nan)
    ov = (snap["ce_vol"] + snap["pe_vol"]) * lot
    f["opt_rvol"] = ov / ov.rolling(20, min_periods=10).mean().shift(1)
    f["opt_vol_oi"] = ov / opt_oi.replace(0, np.nan)
    f["pcr_oi"] = snap["pe_oi"] / snap["ce_oi"].replace(0, np.nan)
    f["pcr_vol"] = snap["pe_vol"] / snap["ce_vol"].replace(0, np.nan)
    f["pcr_chg5"] = f["pcr_oi"] - f["pcr_oi"].shift(5)
    f["pcr_z60"] = (f["pcr_oi"] - f["pcr_oi"].rolling(60, min_periods=30).mean()) / f["pcr_oi"].rolling(60, min_periods=30).std()
    num = snap["pe_chg_near"] - snap["ce_chg_near"]
    den = snap["pe_chg_near"].abs() + snap["ce_chg_near"].abs()
    f["near_writing_bias"] = num / den.replace(0, np.nan)     # +1 = put writing (support)
    f["near_oi_chg"] = (snap["pe_chg_near"] + snap["ce_chg_near"]) / (snap["pe_oi_near"] + snap["ce_oi_near"]).replace(0, np.nan)

    F = snap["F"]
    f["dist_max_pain"] = snap["max_pain"] / F - 1
    f["dist_call_wall"] = snap["call_wall"] / F - 1
    f["dist_put_wall"] = 1 - snap["put_wall"] / F

    # Futures OI build-up classification (price move x OI move).
    ret = tech["ret1"].reindex(f.index)
    oi = f["fut_oi_chg"]
    f["buildup_long"] = ((ret > 0) & (oi > 0)).astype(float) * oi.abs()
    f["buildup_short"] = ((ret < 0) & (oi > 0)).astype(float) * oi.abs()
    f["short_covering"] = ((ret > 0) & (oi < 0)).astype(float) * oi.abs()
    f["long_unwinding"] = ((ret < 0) & (oi < 0)).astype(float) * oi.abs()
    if close_raw is not None and close_raw.notna().any():
        f["fut_basis"] = snap["fut_close"] / close_raw.reindex(f.index) - 1
    else:
        f["fut_basis"] = np.nan
    return f


def buildup_label(ret1: float, oi_chg: float, min_ret=0.003, min_oi=0.02) -> str:
    if np.isnan(ret1) or np.isnan(oi_chg) or abs(ret1) < min_ret or abs(oi_chg) < min_oi:
        return "No clear build-up"
    if ret1 > 0 and oi_chg > 0:
        return "Long build-up"
    if ret1 < 0 and oi_chg > 0:
        return "Short build-up"
    if ret1 > 0 and oi_chg < 0:
        return "Short covering"
    return "Long unwinding"


def market_features(mkt: pd.DataFrame) -> pd.DataFrame:
    """mkt: index=date, columns nifty, vix (closes)."""
    f = pd.DataFrame(index=mkt.index)
    if "vix" in mkt:
        f["vix"] = mkt["vix"]
        f["vix_rank"] = ind.rolling_pct_rank(mkt["vix"], TRADING_DAYS, 60)
        f["vix_chg5"] = mkt["vix"] / mkt["vix"].shift(5) - 1
    if "nifty" in mkt:
        n = mkt["nifty"]
        f["nifty_ret5"] = n / n.shift(5) - 1
        f["nifty_hv20"] = np.log(n / n.shift(1)).rolling(20).std() * np.sqrt(TRADING_DAYS)
    return f


def event_features(dates: pd.DatetimeIndex, results) -> pd.DataFrame:
    """Calendar days until the next *announced* results board meeting.

    `results` is a DataFrame[date, announced] (announced may be NaT) or a
    Series of meeting dates. A meeting only counts from its announcement
    date onward, so the backtest cannot peek at meetings nobody knew about.
    When the announcement time is unknown we assume 7 calendar days notice
    (Indian companies must give at least 2 working days).
    """
    f = pd.DataFrame(index=dates)
    days = np.full(len(dates), 60.0)
    since = np.full(len(dates), 60.0)
    if results is not None and len(results):
        if isinstance(results, pd.Series):
            results = pd.DataFrame({"date": results.values, "announced": pd.NaT})
        r = results.dropna(subset=["date"])
        dvals = dates.values.astype("datetime64[ns]")
        for m, a in zip(pd.to_datetime(r["date"]), pd.to_datetime(r["announced"])):
            if pd.notna(a):
                # Announced after the close -> usable from the next day.
                known_from = a.normalize() + pd.Timedelta(days=1 if (a.hour, a.minute) >= (15, 30) else 0)
            else:
                known_from = m - pd.Timedelta(days=7)
            known_from = min(known_from, m)
            mask = (dvals >= np.datetime64(known_from)) & (dvals <= np.datetime64(m))
            dd = (np.datetime64(m) - dvals[mask]).astype("timedelta64[D]").astype(float)
            days[mask] = np.minimum(days[mask], dd)
            after = (dvals > np.datetime64(m)) & (dvals <= np.datetime64(m + pd.Timedelta(days=4)))
            sd = (dvals[after] - np.datetime64(m)).astype("timedelta64[D]").astype(float)
            since[after] = np.minimum(since[after], sd)
        f["days_to_results"] = np.minimum(days, 30.0)
        f["results_within7"] = (days <= 7).astype(float)
        f["results_just_out"] = (since <= 3).astype(float)
    else:
        f["days_to_results"] = np.nan
        f["results_within7"] = np.nan
        f["results_just_out"] = np.nan
    return f
