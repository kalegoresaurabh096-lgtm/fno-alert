"""One-off check of how much expired F&O history Upstox serves (needs Upstox Plus).

Prints counts and dates only, never the token.
"""
import datetime as dt
import sys
import traceback

sys.path.insert(0, ".")
from fno_alert.upstox import Upstox  # noqa: E402

ux = Upstox()
uni = ux.fno_universe(ux.instruments())
print(f"F&O stocks today: {len(uni)}")
today = dt.date.today()
for sym in ["RELIANCE", "HAVELLS", "TATASTEEL"]:
    row = uni[uni["symbol"] == sym]
    if row.empty:
        continue
    key = row["underlying_key"].iloc[0]
    print(f"\n== {sym} ({key})")
    try:
        c = ux.daily_candles(key, dt.date(2015, 1, 1), today)
        print(f"  stock daily candles: {len(c)} rows, from {c['date'].min().date() if len(c) else None}")
    except Exception as e:
        print(f"  stock candles failed: {e}")
    try:
        exps = ux.expired_expiries(key)
        print(f"  expired expiries: {len(exps)}; first {exps[:3]} last {exps[-3:]}")
    except Exception as e:
        print(f"  expiries failed: {e}")
        traceback.print_exc()
        continue
    for e in [exps[0], exps[len(exps) // 2], exps[-1]] if exps else []:
        try:
            oc = ux.expired_contracts(key, e, "option")
            fc = ux.expired_contracts(key, e, "future")
            print(f"  expiry {e}: {len(oc)} option contracts, {len(fc)} futures; option columns {list(oc.columns)[:14]}")
            if len(fc):
                fk = fc.iloc[0].get("instrument_key")
                cf = ux.expired_candles(fk, e - dt.timedelta(days=95), e)
                print(f"    future candles: {len(cf)} rows {cf['date'].min().date() if len(cf) else ''}..; "
                      f"oi>0 rows {(cf['oi'] > 0).sum() if len(cf) else 0}; sample key {fk}")
            if len(oc):
                strikes = oc["strike_price"].astype(float)
                mid = oc.iloc[(strikes - strikes.median()).abs().argsort().iloc[0]]
                co = ux.expired_candles(mid["instrument_key"], e - dt.timedelta(days=95), e)
                print(f"    option {mid.get('trading_symbol')}: {len(co)} rows; oi>0 rows "
                      f"{(co['oi'] > 0).sum() if len(co) else 0}; last {co.tail(1).to_dict('records') if len(co) else ''}")
        except Exception as ex:
            print(f"  expiry {e} failed: {ex}")
print(f"\nAPI calls used: {ux.calls}")
