"""Technical indicators on daily OHLCV data.

Every function is causal: the value on day t only uses data up to and
including day t, so nothing here can leak future prices into a backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average (used by ATR, RSI, ADX)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = rma(delta.clip(lower=0), n)
    down = rma((-delta).clip(lower=0), n)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(up.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5):
    mid = ema(df["close"], n)
    a = atr(df, n)
    return mid, mid + mult * a, mid - mult * a


def adx(df: pd.DataFrame, n: int = 14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_n = rma(true_range(df), n)
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), n) / tr_n
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), n) / tr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, n), plus_di, minus_di


def rolling_pct_rank(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile (0..1) of today's value within the trailing window (today included)."""
    mp = min_periods or window
    def _rank(a: np.ndarray) -> float:
        last = a[-1]
        if np.isnan(last):
            return np.nan
        valid = a[~np.isnan(a)]
        return (valid < last).sum() / max(len(valid) - 1, 1)
    return s.rolling(window, min_periods=mp).apply(_rank, raw=True)


def consecutive_true(b: pd.Series) -> pd.Series:
    """Number of consecutive True values ending at each row."""
    b = b.fillna(False).astype(bool)
    groups = (~b).cumsum()
    return b.astype(int).groupby(groups).cumsum()


def _pivots(s: pd.Series, k: int, kind: str) -> pd.Series:
    """Confirmed swing points. A pivot at bar i is only *known* at bar i+k,
    so the returned series is shifted to the confirmation bar to stay causal."""
    win = 2 * k + 1
    if kind == "low":
        is_piv = s == s.rolling(win, center=True).min()
    else:
        is_piv = s == s.rolling(win, center=True).max()
    return is_piv.fillna(False)


def divergence(close: pd.Series, osc: pd.Series, k: int = 3, lookback: int = 40) -> pd.Series:
    """+1 bullish divergence, -1 bearish, 0 none, computed causally.

    Bullish: the latest confirmed swing low in price is lower than the previous
    one while the oscillator's value at those lows is higher. Bearish is the
    mirror image on swing highs. The signal stays active for k bars after the
    swing is confirmed.
    """
    n = len(close)
    out = np.zeros(n)
    lows = _pivots(close, k, "low").to_numpy()
    highs = _pivots(close, k, "high").to_numpy()
    c = close.to_numpy()
    o = osc.to_numpy()
    for kind, piv, sign in (("low", lows, 1), ("high", highs, -1)):
        idx = np.flatnonzero(piv)
        for j in range(1, len(idx)):
            i2, i1 = idx[j], idx[j - 1]
            if i2 - i1 > lookback or i2 - i1 < k:
                continue
            if np.isnan(o[i1]) or np.isnan(o[i2]):
                continue
            if kind == "low" and c[i2] < c[i1] and o[i2] > o[i1]:
                pass
            elif kind == "high" and c[i2] > c[i1] and o[i2] < o[i1]:
                pass
            else:
                continue
            start = i2 + k  # bar on which the pivot becomes known
            end = min(start + k + 1, n)
            if start < n:
                out[start:end] = sign
    return pd.Series(out, index=close.index)
