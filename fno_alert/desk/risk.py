"""Shared risk engine for the multi-strategy desk.

Every strategy asks this engine before it opens a position, and the desk runner
asks it on every price check whether everything must be closed. The same code
runs in the backtest and in paper (and, later, live) trading.

Rules (numbers in config.toml [desk.risk]):
  * kill switch        config kill_switch = true, or a file state/desk/KILL: close all, open nothing
  * event days         no new position on a day listed in intraday/events.csv
  * per-trade loss     a strategy's planned loss on one trade <= its max_trade_loss_pct of capital
  * margin             estimated margin of all open positions <= max_margin_frac of capital
  * daily loss         account loss today (booked + open) >= daily_loss_pct: nothing new today and
                       close all (daily_closes = "all") or only unhedged positions ("unhedged")
  * monthly loss       account down monthly_loss_pct from the month's start: close all and pause
                       until Saurabh resumes it (the backtest resumes at the next month)
  * strategy pause     a strategy whose own drawdown passes auto_pause_mult x its worst
                       backtest drawdown is paused (paper and live only)
A loss limit must hold on `confirm_checks` checks in a row before it fires, so one
bad print on a thin far-away option does not close everything.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import asdict, dataclass, field

KILL_FILE = os.path.join("state", "desk", "KILL")


@dataclass(frozen=True)
class RiskLimits:
    capital: float = 500_000
    daily_loss_pct: float = 0.02
    monthly_loss_pct: float = 0.06
    max_margin_frac: float = 0.80
    max_trade_loss_pct: dict = field(default_factory=lambda: {"strangle": 0.025, "condor": 0.08})
    auto_pause_mult: float = 1.5
    block_event_days: bool = True
    confirm_checks: int = 2
    daily_closes: str = "all"           # what the daily limit closes: "all", or "unhedged" (positions
                                        # whose loss is capped by bought wings are left to their own rules)
    kill_switch: bool = False
    monthly_auto_resume: bool = False   # backtest only: resume at the next month

    @classmethod
    def from_config(cls, cfg: dict | None, **over) -> "RiskLimits":
        names = set(cls.__dataclass_fields__)
        vals = {k: v for k, v in (cfg or {}).items() if k in names}
        vals.update(over)
        return cls(**vals)


@dataclass
class RiskState:
    day: str = ""
    month: str = ""
    day_start_equity: float = 0.0
    month_start_equity: float = 0.0
    halted_today: str = ""          # reason, empty = trading allowed today
    paused: str = ""                # monthly pause reason (needs a manual resume when live)
    paused_strategies: dict = field(default_factory=dict)   # name -> reason
    strategy_peak: dict = field(default_factory=dict)       # name -> peak cumulative P&L
    breach_count: int = 0
    events: list = field(default_factory=list)              # log of risk actions

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "RiskState":
        d = dict(d or {})
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


class RiskEngine:
    def __init__(self, limits: RiskLimits, state: RiskState | None = None, events: dict | None = None,
                 strategy_dd: dict | None = None, kill_file: str = KILL_FILE):
        self.L = limits
        self.s = state or RiskState()
        self.event_days = events or {}
        self.strategy_dd = strategy_dd or {}   # name -> worst backtest drawdown in rupees (positive)
        self.kill_file = kill_file

    # ------------------------------------------------------------ helpers
    def _log(self, ts, what: str):
        self.s.events.append({"ts": str(ts)[:16], "action": what})
        self.s.events = self.s.events[-200:]

    def killed(self) -> str:
        if self.L.kill_switch:
            return "kill switch on in config.toml"
        if self.kill_file and os.path.exists(self.kill_file):
            return "kill switch file state/desk/KILL"
        return ""

    def start_day(self, day: dt.date, equity: float):
        """Call once at the start of each session with the account equity (open positions marked)."""
        d, m = str(day), str(day)[:7]
        if self.s.day == d:
            return
        self.s.day, self.s.day_start_equity, self.s.halted_today, self.s.breach_count = d, equity, "", 0
        if self.s.month != m:
            self.s.month, self.s.month_start_equity = m, equity
            if self.s.paused and self.L.monthly_auto_resume:
                self.s.paused = ""

    def resume(self, ts=None):
        """Saurabh's manual reset after a monthly pause or a strategy pause."""
        self.s.paused, self.s.paused_strategies, self.s.halted_today = "", {}, ""
        self._log(ts or dt.datetime.now(), "resumed by hand")

    # ------------------------------------------------------------ entries
    def allow_entry(self, strategy: str, planned_loss: float, margin: float, open_margin: float,
                    day: dt.date) -> str:
        """'' = allowed, otherwise the reason it is refused."""
        k = self.killed()
        if k:
            return k
        if self.s.paused:
            return f"desk paused: {self.s.paused}"
        if self.s.halted_today:
            return f"stopped for today: {self.s.halted_today}"
        if strategy in self.s.paused_strategies:
            return f"strategy paused: {self.s.paused_strategies[strategy]}"
        if self.L.block_event_days and day in self.event_days:
            return f"event day: {self.event_days[day]}"
        cap = self.L.max_trade_loss_pct.get(strategy, 0.02) * self.L.capital
        if planned_loss > cap:
            return f"planned loss Rs {planned_loss:,.0f} is above the Rs {cap:,.0f} limit"
        if open_margin + margin > self.L.max_margin_frac * self.L.capital:
            return "not enough margin left"
        return ""

    # ------------------------------------------------------------ every check
    def closes(self, reason: str, hedged: bool) -> bool:
        """Does this risk action close a position of this kind?"""
        if self.s.paused or "kill switch" in reason or self.L.daily_closes == "all":
            return True
        return not hedged

    def check(self, ts, equity: float) -> str:
        """Return a reason to close positions now (see closes()), or ''."""
        k = self.killed()
        if k:
            return k
        day_loss = self.s.day_start_equity - equity
        month_loss = self.s.month_start_equity - equity
        why = ""
        if self.L.monthly_loss_pct > 0 and month_loss >= self.L.monthly_loss_pct * self.s.month_start_equity:
            why = "monthly"
        elif self.L.daily_loss_pct > 0 and day_loss >= self.L.daily_loss_pct * self.L.capital:
            why = "daily"
        if not why:
            self.s.breach_count = 0
            return ""
        self.s.breach_count += 1
        if self.s.breach_count < self.L.confirm_checks:
            return ""
        if why == "monthly" and not self.s.paused:
            self.s.paused = f"down Rs {month_loss:,.0f} this month (limit {self.L.monthly_loss_pct:.0%})"
            self.s.halted_today = self.s.paused
            self._log(ts, "monthly loss limit: closed all, desk paused")
            return self.s.paused
        if why == "daily" and not self.s.halted_today:
            self.s.halted_today = f"down Rs {day_loss:,.0f} today (limit {self.L.daily_loss_pct:.0%})"
            self._log(ts, "daily loss limit: " + ("closed all" if self.L.daily_closes == "all"
                                                  else "closed unhedged positions") + ", nothing new today")
            return self.s.halted_today
        return self.s.halted_today or self.s.paused

    def update_strategy(self, ts, name: str, cum_pnl: float):
        """Pause a strategy whose drawdown from its own peak passes the limit."""
        peak = max(self.s.strategy_peak.get(name, 0.0), cum_pnl)
        self.s.strategy_peak[name] = peak
        worst = self.strategy_dd.get(name)
        if not worst or name in self.s.paused_strategies:
            return
        if peak - cum_pnl > self.L.auto_pause_mult * worst:
            self.s.paused_strategies[name] = (f"drawdown Rs {peak - cum_pnl:,.0f} is more than "
                                              f"{self.L.auto_pause_mult}x its worst backtest drawdown")
            self._log(ts, f"{name} paused")
