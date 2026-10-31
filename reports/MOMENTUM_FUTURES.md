# 250-day momentum on stock futures (backtest, 31 Oct 2026)

Script: `research/futures_momentum250.py` (run after `build-data`; numbers in `research/out/`).

## What was tested
- Real NSE stock-futures contracts from the Upstox download: daily closes, lot sizes, 22 monthly
  rolls from 30 Oct 2024 to 28 Sep 2026 (the futures history only goes back that far).
- Each roll (day before monthly expiry): rank F&O stocks by past 250-day return, known the day
  before; buy one lot of next month's future for each top pick; exit at the next roll.
- Costs: brokerage, exchange, SEBI, GST, stamp, STT on sells (0.02%, 0.05% from 1 Apr 2026) and
  0.05% slippage per side. Margin assumed 20% of notional; account = 2x peak margin.
- 74 contracts had a split or bonus during their life; prices were adjusted so these do not show
  up as fake losses (the futures returns match the cash returns, correlation 0.998).

## Results (Oct 2024 to Sep 2026, NIFTY -6%)

| Strategy | Account | Total | With 6% interest on spare cash | Worst fall |
|---|---|---|---|---|
| Long top 10, plain 250-day | Rs 43 L | +7% | +12% | -46% |
| Long top 10, 250-day skip last month | Rs 41 L | -4% | +2% | -44% |
| Long top 10 / short bottom 10 | Rs 61 L | 0% | +5% | -23% |
| Long top 10, only when NIFTY above 200-day average | Rs 41 L | -23% | -19% | -39% |
| Rs 5 lakh account (1-2 lots) | Rs 5 L | +32% | +38% | -92% |
| Rs 25 lakh account (top picks that fit) | Rs 25 L | -23% | -17% | -67% |

## Takeaways
- In these two years, momentum picks beat NIFTY by about 10 points but made roughly nothing in
  absolute terms. The +42% year (2024) of the earlier cash test falls before this futures window.
- Futures cost about 0.4% a month versus owning the same shares (futures trade at a premium to
  the stock). Same picks as shares: +5.8%; as futures: -3.0%, before interest on idle cash.
- Futures add leverage, not edge. One lot is Rs 5-10 lakh, so a Rs 5 lakh account holds only 1-2
  stocks at ~2.5x leverage; it fell to about Rs 42,000 before recovering, which in real life means
  margin calls and forced exits.
- Adding a market filter (only buy when NIFTY is in an uptrend) made it worse here.

## Caveats
- Only 22 monthly rolls: far too short to judge a yearly-scale strategy.
- Stock list is today's F&O list (survivorship bias flatters momentum).
- Margin is an estimate (real stock futures margin ~15-30%).
