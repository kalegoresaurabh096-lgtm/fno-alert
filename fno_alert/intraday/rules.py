"""Saurabh's option-selling rules, turned into exact numbers.

Every number lives in config.toml [intraday] so the backtest, the paper run and
live trading all use the same rules. Rule numbers refer to his list:

 1  risk_pct            most a trade may lose, as a share of capital (1-2%)
 2  max loss known      hedged: (wing - credit) x qty. Unhedged: stop distance x qty,
                        which is only a plan; a spike can fill the stop worse.
 3  no averaging        a position is never added to (the engine has no "add" path)
 4  hedged              buy wings `wing` points beyond each short strike
 5  skip bad days       event days, big opening gaps, high VIX, wild first half hour,
                        too little premium: no trade
 6  follow the trend    index up more than trend_pct since the open: sell puts only.
                        Down more than that: calls only. In between: both sides
 7  take_profit         exit when this share of the credit has been earned (0.5 = 50%)
 8  daily_loss_pct      stop trading for the day after losing this much
 9  intraday only       every position is closed by exit_time; event days skipped
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from dataclasses import dataclass, field, fields, replace

HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS_CSV = os.path.join(HERE, "events.csv")


@dataclass(frozen=True)
class Rules:
    capital: float = 500_000
    risk_pct: float = 0.02          # rule 1 (top of the 1-2% range: 1% cannot fit one hedged NIFTY lot with a 100-point wing)
    daily_loss_pct: float = 0.02    # rule 8
    max_trades_per_day: int = 1     # 2 = one re-entry after the first trade closes
    reentry_gap_min: int = 15
    hedged: bool = True             # rule 4
    wing: dict = field(default_factory=lambda: {"NIFTY": 100, "SENSEX": 300})
    short_delta: float = 0.20
    take_profit: float = 0.50       # rule 7
    stop_mult: float = 1.0          # exit when loss reaches stop_mult x credit
    entry_time: str = "09:45"
    last_entry_time: str = "13:30"
    exit_time: str = "15:15"        # rule 9
    or_minutes: int = 30            # opening range used for the trend (rule 6)
    trend: str = "open"             # "open" = rule 6 as above, "none" = always both sides
    trend_pct: float = 0.25         # move since the open (%) that counts as a trend
    neutral: str = "both"           # no clear trend: "both" sides (condor) or "skip" the day
    iv_rv_min: float = 0.0          # sell only if at-the-money IV >= this x recent realised vol (0 = off)
    max_dte: int = 1                # trade only when the nearest expiry is <= this many sessions away
    gap_max_pct: float = 1.0        # rule 5: skip if the open gaps more than this
    vix_max: float = 22.0           # rule 5
    or_max_pct: float = 0.9         # rule 5: skip if the first half hour moved more than this
    min_credit_frac: float = 0.12   # rule 5: hedged credit must be >= this share of the wing
    min_credit_pts: float = 4.0     # rule 5: premium too small to be worth it
    max_margin_frac: float = 0.8    # never use more than this share of capital as margin
    naked_margin_frac: float = 0.13 # rough SPAN+exposure per naked short, share of notional
    # Execution and costs
    slippage: float = 0.005         # per leg per side, share of price
    min_slippage: float = 0.25      # rupees per unit
    stop_slippage: float = 0.02     # extra on stop-loss exits (market order in a fast market)
    stop_trigger: str = "close"     # backtest: "close" = position value at each minute's close
                                    # (how a watched stop works); "high" = worst print in the minute
    brokerage: float = 20.0         # per order
    exch_fee: dict = field(default_factory=lambda: {"NIFTY": 0.0003503, "SENSEX": 0.000325})
    sebi_fee: float = 0.000001
    gst: float = 0.18
    stamp_buy: float = 0.00003
    stt_sell: float = 0.001

    @classmethod
    def from_config(cls, cfg: dict | None, **over) -> "Rules":
        names = {f.name for f in fields(cls)}
        vals = {k: v for k, v in (cfg or {}).items() if k in names}
        vals.update(over)
        return cls(**vals)

    def with_(self, **kw) -> "Rules":
        return replace(self, **kw)

    def risk_rupees(self) -> float:
        return self.capital * self.risk_pct


def hm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def load_events(path: str = EVENTS_CSV) -> dict[dt.date, str]:
    """Rule 9: days with scheduled market-moving events (no trading on these days)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if (r.get("date") or "").strip() and not r["date"].startswith("#"):
                out[dt.date.fromisoformat(r["date"].strip())] = r.get("event", "").strip()
    return out
