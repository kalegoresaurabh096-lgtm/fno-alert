"""One-off check of how much intraday (minute) option history Upstox serves.

Prints counts and dates only, never the token.
"""
import datetime as dt
import sys
import urllib.parse

sys.path.insert(0, ".")
from fno_alert.upstox import Upstox  # noqa: E402

ux = Upstox()
q = lambda k: urllib.parse.quote(k, safe="")  # noqa: E731


def show(label, data):
    c = (data or {}).get("candles", [])
    if not c:
        print(f"  {label}: 0 candles")
        return
    vols = [x[5] for x in c]
    print(f"  {label}: {len(c)} candles {c[-1][0]} .. {c[0][0]}; median vol {sorted(vols)[len(vols)//2]}; "
          f"first {c[-1][:7]}")


for name, key in [("NIFTY", "NSE_INDEX|Nifty 50"), ("SENSEX", "BSE_INDEX|SENSEX"),
                  ("BANKNIFTY", "NSE_INDEX|Nifty Bank")]:
    print(f"\n== {name}")
    for d0, d1 in [(dt.date(2024, 9, 2), dt.date(2024, 9, 6)), (dt.date(2026, 10, 26), dt.date(2026, 10, 30))]:
        try:
            show(f"index 1min {d0}", ux.get(f"/v3/historical-candle/{q(key)}/minutes/1/{d1}/{d0}"))
        except Exception as e:
            print(f"  index 1min {d0} failed: {e}")
    try:
        exps = ux.expired_expiries(key)
    except Exception as e:
        print(f"  expiries failed: {e}")
        continue
    print(f"  expired expiries: {len(exps)}; first {exps[:2]} last {exps[-2:]}")
    for e in [exps[0], exps[len(exps) // 2], exps[-1]] if exps else []:
        try:
            oc = ux.expired_contracts(key, e, "option")
            k = oc["strike_price"].astype(float)
            for opt in ["CE", "PE"]:
                sub = oc[oc["instrument_type"] == opt]
                mid = sub.iloc[(sub["strike_price"].astype(float) - k.median()).abs().argsort().iloc[0]]
                ek = mid["instrument_key"]
                for iv in ["1minute", "5minute"]:
                    data = ux.get(f"/v2/expired-instruments/historical-candle/{q(ek)}/{iv}/{e}/{e - dt.timedelta(days=6)}")
                    show(f"{e} {mid.get('trading_symbol')} {iv}", data)
        except Exception as ex:
            print(f"  expiry {e} failed: {ex}")

print("\n== Stock option, RELIANCE")
uni = ux.fno_universe(ux.instruments())
key = uni[uni["symbol"] == "RELIANCE"]["underlying_key"].iloc[0]
exps = ux.expired_expiries(key)
e = exps[-1]
oc = ux.expired_contracts(key, e, "option")
k = oc["strike_price"].astype(float)
mid = oc.iloc[(k - k.median()).abs().argsort().iloc[0]]
show(f"{e} {mid.get('trading_symbol')} 1minute",
     ux.get(f"/v2/expired-instruments/historical-candle/{q(mid['instrument_key'])}/1minute/{e}/{e - dt.timedelta(days=6)}"))
print(f"\nAPI calls used: {ux.calls}")
