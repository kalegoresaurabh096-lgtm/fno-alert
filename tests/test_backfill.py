import datetime as dt
import importlib.util
import os

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "bf", os.path.join(os.path.dirname(__file__), "..", "scripts", "upstox_backfill.py"))
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)


class FakeUx:
    def __init__(self):
        self.asked = []

    def expired_contracts(self, key, e, kind):
        if kind == "future":
            return pd.DataFrame([{"instrument_key": "F", "lot_size": 500}])
        return pd.DataFrame([{"instrument_key": f"O{k}{t}", "strike_price": k, "instrument_type": t, "lot_size": 500}
                             for k in range(60, 150, 10) for t in ("CE", "PE")])

    def expired_candles(self, key, s, e):
        self.asked.append(key)
        d = pd.bdate_range(e - dt.timedelta(days=30), e)
        return pd.DataFrame({"date": d, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                             "volume": 5000.0, "oi": 10000.0})


def test_fetch_unit_band_and_units():
    ux = FakeUx()
    df = bf.fetch_unit(ux, "AAA", "K", dt.date(2025, 10, 28), pd.DataFrame())
    strikes = sorted(df.loc[df["inst"] != "FUT", "strike"].unique())
    assert strikes == [90.0, 100.0, 110.0]          # within 15% of the futures price
    assert (df["volume"] == 10).all()               # shares -> contracts
    assert (df["inst"] == "FUT").sum() > 0 and df["date"].max() <= dt.date(2025, 10, 28)
