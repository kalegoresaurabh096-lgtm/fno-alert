# Multi-strategy desk: backtest

NIFTY, 1-minute Upstox history, Sep 2024 to Nov 2026, Rs 5 lakh, after slippage,
brokerage, STT, exchange fees, GST and stamp duty. `python -m fno_alert.desk.backtest`.

| Setup | Net P&L | Per year | Worst drawdown | Worst day | Before Nov 2025 | From Nov 2025 |
|---|---|---|---|---|---|---|
| **Desk (both strategies, risk limits on)** | **+Rs 45,276** | **4.2%** | **-Rs 22,826 (4.4%)** | -Rs 12,502 | +14,880 | +30,396 |
| Desk, no risk limits | +48,541 | 4.5% | -22,826 | -12,502 | +14,880 | +33,661 |
| Expiry-day strangle alone | +21,213 | 2.0% | -7,228 | -4,625 | +9,124 | +12,089 |
| Weekly condor alone | +26,120 | 2.4% | -21,351 | -12,502 | +5,756 | +20,364 |

* The two strategies' daily P&L is uncorrelated (-0.01), so together they draw
  down about as much as the condor alone while earning roughly the sum of both.
* The strangle's numbers are the same trades as its own backtest
  (`reports/intraday/trades_pick.csv`). It was chosen on data before Nov 2025;
  after that is its untouched holdout. The condor's settings come from the daily
  2-year test (`research/nifty_condor.py`), so its two halves are not a clean
  holdout.
* Risk limits cost about Rs 3,300 here (one condor skipped on an RBI day, a few
  strangles cut to fewer lots for margin). They are there for days the last two
  years did not have.
* The worst day (-12.5k, 2.5%) is above the 2% daily limit: the condor is held
  overnight and the loss came in a gap at the open, before any limit can act.
* Costs were Rs 27,776, about 38% of the gross profit.

## Choices the backtest informed

* **Daily limit leaves the condor alone** (`daily_closes = "unhedged"`). Closing
  everything at the 2% daily limit cut the condor at intraday lows 3 times and
  dropped the desk to +Rs 16,855. The condor's loss is capped by its wings, so the
  daily limit now closes the unhedged strangle and blocks new trades; the 6%
  monthly limit and the kill switch still close everything.
* **No loss stop on the condor.** A stop at 3% or 2% of capital turned the condor
  into a loser (-Rs 17,855 and -Rs 19,122): it sold at the lows of moves that
  came back.
* **The condor's per-trade limit is 8%**, not 2%: one lot with 500-point wings can
  lose up to about Rs 36,000 (lot 75) if NIFTY runs through a wing. The worst
  trade in two years was -Rs 10,748.

Two years is a short test, and both strategies sell options, so a market crash or
a volatility spike can hit both in the same week. These are paper results to
check, not a forecast.
