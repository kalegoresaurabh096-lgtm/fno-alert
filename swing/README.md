# Swing trade backtest: +3% target, mcap above Rs 5,000 crore

Research only. Nothing here trades, alerts or runs on a schedule.

**Result: no version beat buying and holding NIFTY on the untouched 2019-2026 years.**

| Exit if +3% is not hit | Years | Trades | Win rate | Avg trade after costs | Worst trade | Per year | NIFTY per year | Worst fall | NIFTY worst fall |
|---|---|---|---|---|---|---|---|---|---|
| Sell on day 15 | 2009-2018 (tuning) | 1,378 | 79% | +0.57% | -34% | 15.3% | 13.2% | -24% | -28% |
| Sell on day 15 | 2019-2026 (test) | 1,438 | 79% | +0.29% | -45% | **9.2%** | 11.4% | -36% | -38% |
| Hold up to 1 year, no stop | 2009-2018 (tuning) | 593 | 97% | +1.9% | -42% | 24.1% | 13.2% | -22% | -28% |
| Hold up to 1 year, no stop | 2019-2026 (test) | 470 | 96% | +0.8% | -69% | **7.8%** | 11.4% | -45% | -38% |

NIFTY is the price index (no dividends, about 1.2% a year more with them).

## The rule (picked on 2009-2018 only, best of 180 variants)

1. Rank stocks by 6-month gain; keep the top 30%.
2. Signal when the stock closes below its 10-day average, above its 50-day average, with RSI(2) under 10.
3. Buy at the next open (strongest first if more signals than free slots; at least Rs 5 cr traded a day).
4. Limit sell at +3%. Otherwise sell at the close on day 15, or (second version) hold up to 250 trading days. No stop-loss.

Rs 5 lakh split into 5 equal slots. Costs: Rs 20 brokerage per order, 0.1% STT each side, stamp 0.015% on buys,
exchange and SEBI fees, 18% GST, Rs 20 DP charge per sale, plus 0.1% slippage on every fill.

## Holding up to a year instead of a stop-loss

- 96% of trades end at +3%, but the losers average -38% and the worst lost 69% (2019-2026).
- 19% of trades stay open more than 15 days, 7% more than 60 days; 18% fall 10% or more at some point while held.
- All 5 slots are full 74% of the time (36% for the 15-day exit), so new signals get skipped while money sits in losers.
- It looked great on 2009-2018 (24% a year) and then trailed NIFTY on 2019-2026 with a deeper worst fall (-45%).

## What else was tested

Entry setups: uptrend dip (RSI(2)), three down days in an uptrend, close below the lower Bollinger band, 20-day high
breakout on high volume, and strong stocks pulling back (chosen). Each with 3 thresholds, 3 market filters (none,
NIFTY above 50-day or 200-day average) and stops of none, 3%, 5% and 8%. Stop-losses made every setup worse.
With the 15-day exit, 0 of 180 variants beat NIFTY on 2019-2026; with the 1-year hold, 2 of 45 did.
See `results/rules.csv`.

## Caveats

- Survivorship: the stock list is today's top 800. Companies that collapsed or delisted since 2008 are missing, which
  flatters the no-stop 1-year hold most.
- Market cap on past days = price x today's share count.
- When the +3% target and the day's low fall on the same day, the sale is counted at +3%.

## Run it

```
git archive origin/swing-data | tar -x -C /tmp/sd          # prices from the one-off download job
python swing/backtest.py --data /tmp/sd --out /tmp/res15  --train-end 2018-12-31
python swing/backtest.py --data /tmp/sd --out /tmp/res250 --train-end 2018-12-31 --max-days 250 --stops 0
```

`swing/fetch_history.py` (workflow "Swing backtest - download price history", manual only) rebuilds the
`swing-data` branch from Yahoo Finance.
