"""Black-76 option pricing and implied volatility.

NSE stock options are European and we price them against the futures price
of the same expiry, which removes the need to guess dividends and carry.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

RISK_FREE = 0.065  # only used for discounting; IV is insensitive to it


def price(F, K, T, sigma, is_call, r: float = RISK_FREE):
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    s = np.maximum(sigma, 1e-9)
    d1 = (np.log(F / K) + 0.5 * s * s * T) / (s * sqrtT)
    d2 = d1 - s * sqrtT
    df = np.exp(-r * T)
    call = df * (F * ndtr(d1) - K * ndtr(d2))
    put = df * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def implied_vol(px, F, K, T, is_call, r: float = RISK_FREE, lo=0.01, hi=5.0, iters=60):
    """Vectorised bisection. Returns NaN where the price is outside no-arbitrage bounds."""
    px, F, K, T = (np.asarray(x, dtype=float) for x in (px, F, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    df = np.exp(-r * T)
    intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0)) * df
    upper = np.where(is_call, F, K) * df
    valid = (px > intrinsic + 1e-6) & (px < upper) & (T > 0) & (F > 0) & (K > 0)
    a = np.full(px.shape, lo)
    b = np.full(px.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        pm = price(F, K, T, m, is_call, r)
        too_high = pm > px
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    iv = np.where(valid, iv, np.nan)
    # Bisection pinned at a bound means no real solution in range.
    iv = np.where((iv <= lo * 1.01) | (iv >= hi * 0.99), np.nan, iv)
    return iv
