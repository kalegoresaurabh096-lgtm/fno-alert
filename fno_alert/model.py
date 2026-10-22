"""Big-move probability models with strict walk-forward training."""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .labels import HORIZON

TECH = ["ret1", "ret5", "ret20", "hv20", "hv60", "hv_ratio_10_60", "ret1_z", "atr_pct", "atr_ratio_14_50",
        "tr_atr", "nr7", "inside_day", "gap", "clv", "bb_width", "bb_width_rank", "bb_pos", "squeeze_on",
        "squeeze_days", "squeeze_fired", "squeeze_fired_len", "rvol", "rvol5", "dvol_cr", "brk_up20",
        "brk_dn20", "brk_up55", "brk_dn55", "dist_hi20_atr", "dist_lo20_atr", "pos_52w", "range20_atr",
        "rsi", "macd_hist_pct", "macd_cross", "rsi_div", "macd_div", "adx", "adx_slope5", "di_diff",
        "c_ema20", "c_ema50", "c_ema200", "ema20_slope5"]
OPTS = ["atm_iv", "iv_rank", "iv_chg1", "iv_chg5", "iv_z60", "iv_hv", "straddle_pct", "iv_move5", "skew",
        "skew_chg5", "dte", "fut_oi_chg", "fut_oi_chg5", "fut_oi_chg_z", "fut_rvol", "opt_oi_chg", "opt_rvol",
        "opt_vol_oi", "pcr_oi", "pcr_vol", "pcr_chg5", "pcr_z60", "near_writing_bias", "near_oi_chg",
        "dist_max_pain", "dist_call_wall", "dist_put_wall", "buildup_long", "buildup_short",
        "short_covering", "long_unwinding", "fut_basis"]
MARKET = ["vix", "vix_rank", "vix_chg5", "nifty_ret5", "nifty_hv20"]
EVENTS = ["days_to_results", "results_within7", "results_just_out"]
ALL_FEATURES = TECH + OPTS + MARKET + EVENTS
TARGETS = {"big": lambda d: d["big"], "up": lambda d: ((d["big"] == 1) & (d["dir"] > 0)).astype(float),
           "dn": lambda d: ((d["big"] == 1) & (d["dir"] < 0)).astype(float)}


def feature_list(panel: pd.DataFrame, groups=("tech", "opts", "market", "events")) -> list[str]:
    cols = []
    for g, fl in (("tech", TECH), ("opts", OPTS), ("market", MARKET), ("events", EVENTS)):
        if g in groups:
            cols += [c for c in fl if c in panel.columns and panel[c].notna().mean() > 0.2]
    return cols


def make_hgb(seed: int = 0):
    return HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=350, max_leaf_nodes=15, min_samples_leaf=300,
        l2_regularization=1.0, max_bins=63, random_state=seed)


def make_logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000))


def _clip(X: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> pd.DataFrame:
    return X.clip(lower=lo, upper=hi, axis=1)


@dataclass
class Fitted:
    features: list
    models: dict          # target -> estimator
    lo: pd.Series
    hi: pd.Series
    kind: str = "hgb"

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _clip(df[self.features].astype(float), self.lo, self.hi)
        out = pd.DataFrame(index=df.index)
        for t, m in self.models.items():
            out[f"p_{t}"] = m.predict_proba(X)[:, 1]
        return out

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "Fitted":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def fit(train: pd.DataFrame, features: list, kind: str = "hgb", seed: int = 0) -> Fitted:
    train = train.dropna(subset=["big"])
    X = train[features].astype(float)
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    X = _clip(X, lo, hi)
    models = {}
    for t, fn in TARGETS.items():
        y = fn(train).astype(int)
        m = make_hgb(seed) if kind == "hgb" else make_logit()
        m.fit(X, y)
        models[t] = m
    return Fitted(features, models, lo, hi, kind)


def walk_forward(panel: pd.DataFrame, features: list, kind: str = "hgb", min_train_days: int = 250,
                 step_days: int = 63, horizon: int = HORIZON, verbose: bool = True) -> pd.DataFrame:
    """Expanding-window walk-forward. Each test block is predicted by a model
    trained only on rows whose label window ended before the block starts."""
    dates = np.sort(panel["date"].unique())
    preds = []
    start = min_train_days
    while start < len(dates):
        end = min(start + step_days, len(dates))
        # Embargo: a training row at day i needs closes up to i+horizon.
        train_end = dates[start - horizon - 1]
        train = panel[panel["date"] <= train_end]
        test = panel[(panel["date"] >= dates[start]) & (panel["date"] <= dates[end - 1])]
        m = fit(train, features, kind)
        p = m.predict(test)
        p["date"] = test["date"].values
        p["symbol"] = test["symbol"].values
        preds.append(p)
        if verbose:
            print(f"  fold {pd.Timestamp(dates[start]).date()}..{pd.Timestamp(dates[end - 1]).date()} "
                  f"train={len(train):,} test={len(test):,}", flush=True)
        start = end
    return pd.concat(preds)
