"""The Upstox history path must give the same derivatives tables as the NSE path."""
import glob
import os

import numpy as np
import pandas as pd

from fno_alert import dataset, panel
from tests.synthetic import generate


def _to_upstox(raw, up):
    """Re-shape synthetic bhavcopy files into scripts/upstox_backfill.py's layout:
    only traded days, volume in shares, no chg_oi."""
    fo = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(raw, "fo", "*.csv.gz")))
    fo = fo[fo["volume"] > 0].copy()
    fo["volume"] = fo["volume"] * fo["lot"]
    for (sym, e), g in fo.groupby(["symbol", "expiry"]):
        os.makedirs(os.path.join(up, "fo", sym), exist_ok=True)
        out = g[["date", "symbol", "inst", "expiry", "strike", "open", "high", "low", "close", "settle", "volume",
                 "oi", "lot"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out["expiry"] = pd.to_datetime(out["expiry"]).dt.date
        out["volume"] = out["volume"] / out["lot"]
        out.to_parquet(os.path.join(up, "fo", sym, f"{e}.parquet"), index=False)
    cm = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(raw, "cm", "*.csv.gz")))
    os.makedirs(os.path.join(up, "eq"), exist_ok=True)
    for sym, g in cm.groupby("symbol"):
        g[["date", "open", "high", "low", "close", "volume"]].to_csv(os.path.join(up, "eq", f"{sym}.csv"), index=False)
    os.makedirs(os.path.join(up, "index"), exist_ok=True)
    for name, src in (("nifty", "NIFTY"), ("vix", "INDIAVIX")):
        y = pd.read_csv(os.path.join(raw, "yahoo", f"{src}.csv"))
        y[["date", "close"]].to_csv(os.path.join(up, "index", f"{name}.csv"), index=False)
    # calendar must include every trading day
    days = sorted(cm["date"].unique())
    pd.DataFrame({"date": days, "close": 1.0}).merge(
        pd.read_csv(os.path.join(up, "index", "nifty.csv")), on="date", how="left", suffixes=("_x", "")
    )[["date", "close"]].fillna(1.0).to_csv(os.path.join(up, "index", "nifty.csv"), index=False)


def test_upstox_path_matches_nse(tmp_path):
    raw = tmp_path / "raw"
    generate(str(raw), n_symbols=4, n_days=160)
    up = tmp_path / "upstox"
    _to_upstox(str(raw), str(up))
    p_nse, p_up = tmp_path / "p_nse", tmp_path / "p_up"
    dataset.build_all(str(raw), str(p_nse), workers=1)
    raw2 = tmp_path / "raw2"
    dataset.build_all(str(raw2), str(p_up), workers=1, upstox_dir=str(up))
    a = pd.read_parquet(p_nse / "opt_daily.parquet").set_index(["symbol", "date"]).sort_index()
    b = pd.read_parquet(p_up / "opt_daily.parquet").set_index(["symbol", "date"]).sort_index()
    common = a.index.intersection(b.index)
    assert len(common) > 0.9 * len(a)
    for c in ("F", "atm_iv", "fut_oi"):
        x, y = a.loc[common, c].astype(float), b.loc[common, c].astype(float)
        ok = np.isclose(x, y, rtol=0.02, equal_nan=True)
        assert ok.mean() > 0.95, (c, ok.mean())
    # Synthetic data has OI on contracts before their first trade, which a candle feed
    # cannot show, so option OI totals only need to move together.
    for c in ("ce_oi", "pe_oi"):
        x, y = a.loc[common, c].astype(float), b.loc[common, c].astype(float)
        assert x.corr(y) > 0.9, (c, x.corr(y))
        assert ((y / x).median() > 0.8), (c, (y / x).median())
    eq = pd.read_parquet(p_up / "ohlcv.parquet")
    assert set(eq["source"]) == {"upstox"} and "close_raw" in eq
    pnl = panel.build_panel(str(p_up), min_turnover_cr=0)
    assert len(pnl) > 100 and pnl["big"].notna().any()
