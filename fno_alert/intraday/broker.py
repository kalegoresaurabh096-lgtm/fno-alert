"""Order routing. Paper mode never sends anything. Live mode is switched OFF and
places real Upstox orders only when all three locks are open:

  1. config.toml  [intraday_live] enabled = true
  2. environment  LIVE_TRADING=I_ACCEPT_REAL_MONEY_RISK
  3. environment  UPSTOX_TRADING_TOKEN  (a daily login token with order permission;
                  the read-only Analytics Token cannot place orders)

Order sequence keeps the position hedged at every moment: on entry the hedges are
bought before the shorts are sold; on exit the shorts are bought back before the
hedges are sold. Orders are intraday (product "I"), so the broker also squares
them off near the close if anything is left.
"""
from __future__ import annotations

import os
import time

import requests

ORDER_URL = "https://api-hft.upstox.com/v2/order/place"
LIVE_PHRASE = "I_ACCEPT_REAL_MONEY_RISK"


class LiveTradingLocked(RuntimeError):
    pass


def live_locks(cfg: dict) -> list[str]:
    """Return the locks that are still closed (empty list = live trading allowed)."""
    closed = []
    if not cfg.get("enabled", False):
        closed.append("config.toml [intraday_live] enabled is false")
    if os.environ.get("LIVE_TRADING") != LIVE_PHRASE:
        closed.append("LIVE_TRADING is not set to the confirmation phrase")
    if not os.environ.get("UPSTOX_TRADING_TOKEN"):
        closed.append("UPSTOX_TRADING_TOKEN is missing")
    return closed


class Broker:
    def __init__(self, mode: str, cfg: dict, session: requests.Session | None = None):
        self.mode = mode
        self.cfg = cfg
        self.max_lots = int(cfg.get("max_lots", 1))
        if mode == "live":
            closed = live_locks(cfg)
            if closed:
                raise LiveTradingLocked("Live trading is OFF: " + "; ".join(closed))
            self.s = session or requests.Session()
            self.s.headers.update({"Authorization": f"Bearer {os.environ['UPSTOX_TRADING_TOKEN']}",
                                   "Accept": "application/json", "Content-Type": "application/json"})

    def _order(self, key: str, qty: int, side: str, tag: str) -> str:
        body = {"quantity": int(qty), "product": "I", "validity": "DAY", "price": 0, "tag": tag[:20],
                "instrument_token": key, "order_type": "MARKET", "transaction_type": side,
                "disclosed_quantity": 0, "trigger_price": 0, "is_amo": False}
        r = self.s.post(ORDER_URL, json=body, timeout=15)
        js = r.json() if r.content else {}
        if r.status_code != 200 or js.get("status") != "success":
            raise RuntimeError(f"order rejected: HTTP {r.status_code} {str(js)[:300]}")
        return js["data"]["order_id"]

    def open(self, pos, keys: dict) -> bool:
        if self.mode != "live":
            return True
        if pos.lots > self.max_lots:  # extra cap for the first live weeks
            pos.qty = pos.qty / pos.lots * self.max_lots
            pos.lots = self.max_lots
        done = []
        try:
            for l in sorted(pos.legs, key=lambda l: -l.side):  # buys (hedges) first
                k = keys[(pos.underlying, l.strike, l.opt)]
                done.append((k, l, self._order(k, pos.qty, "BUY" if l.side > 0 else "SELL", "fno-intraday")))
                time.sleep(0.3)
        except Exception:
            # undo whatever was filled so we are never left with a naked short
            for k, l, _ in reversed(done):
                try:
                    self._order(k, pos.qty, "SELL" if l.side > 0 else "BUY", "fno-undo")
                except Exception as e:
                    print(f"UNDO FAILED for {k}: {e} - check the Upstox app now", flush=True)
            raise
        pos.notes["orders"] = [o for _, _, o in done]
        return True

    def close(self, pos, keys: dict):
        if self.mode != "live":
            return
        for l in sorted(pos.legs, key=lambda l: l.side):  # buy back shorts first
            k = keys[(pos.underlying, l.strike, l.opt)]
            self._order(k, pos.qty, "BUY" if l.side < 0 else "SELL", "fno-exit")
            time.sleep(0.3)
