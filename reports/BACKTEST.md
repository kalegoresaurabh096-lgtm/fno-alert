# Backtest report

Data: 88,101 stock-days, 210 F&O stocks, 2024-09-30 to 2026-08-25.
Validation period (used to pick the threshold): 2025-10-01..2026-04-09.
Holdout period (untouched until the end): 2026-04-20..2026-08-25.

A **big move** = the stock closes at least 2x its normal 5-day move (and at least 4%) away from the **next day's open**, up or down, at some point in the next 5 trading days. The overnight gap after the signal is not counted, because you can only act at the next open.

Alerts are counted the way they are emailed: no repeat for the same stock within 5 sessions unless its probability rises by 10 points.

## Headline (holdout)

| | Value |
|---|---|
| Base rate (random F&O stock-day) | 3.9% (95% CI 3.3% to 4.7%) |
| Alert threshold (model probability) | 0.32 |
| Alerts per trading day | 0.27 |
| **Hit rate** (alert followed by a big move) | **25.0%** (95% CI 9.1% to 39.5%) |
| False alarm rate | 75.0% |
| Lift vs random | 6.35x |
| Share of all big moves caught (recall) | 0.8% |
| Simple points-score baseline hit rate | 5.5% (1.44 alerts/day) |
| Average overnight gap you could not trade | 1.1% |

## Ranking quality (AUC, 0.5 = coin flip)

| Signal | Validation | Holdout |
|---|---|---|
| model | 0.626 | 0.695 |
| logistic | 0.659 | 0.708 |
| rule_points | 0.556 | 0.594 |
| iv_rank_only | 0.614 | 0.609 |
| bb_squeeze_only | 0.482 | 0.544 |

AUC by year: 2025: 0.654, 2026: 0.677

## Direction (UP / DOWN calls)

Direction calls are **disabled**: on the validation period no confidence level beat a coin flip with statistical significance (binomial test p < 0.05), so trade ideas are direction-neutral.

| Confidence | Calls (validation) | Accuracy | p-value |
|---|---|---|---|
| 55% | 163 | 44.8% | 0.921 |
| 60% | 148 | 45.3% | 0.891 |
| 65% | 131 | 43.5% | 0.942 |
| 70% | 107 | 39.3% | 0.990 |
| 75% | 92 | 38.0% | 0.992 |

## Calibration (holdout)

| Predicted | Actual | Stock-days |
|---|---|---|
| 2.8% | 2.3% | 11,859 |
| 6.9% | 5.1% | 4,293 |
| 12.1% | 8.0% | 1,340 |
| 17.2% | 10.9% | 496 |
| 23.7% | 17.5% | 315 |
| 33.6% | 25.0% | 56 |
| 45.4% | 33.3% | 6 |
| 51.8% | 50.0% | 2 |

## Suggested trades on real option prices

Entry at the **next day's open** (only if that contract traded), exit after 5 sessions and always before expiry, using an expiry at least 12 days away. Costs: 2% slippage per side (at least Rs 0.15), brokerage, exchange and SEBI fees, GST, stamp duty and STT. Returns are on premium paid (buys) or on ~20% margin (sells). 'Same stock random days' = the same trade on random days of the same stocks.

**validation**

| Action | Trades | Not fillable | Win rate | Avg return | Median | Worst | Profit factor | Avg P&L per lot (Rs) |
|---|---|---|---|---|---|---|---|---|
| BUY CE + BUY PE | 46 | 3 | 21.7% | -14.1% | -16.8% | -64.0% | 0.06 | -4,650 |

**validation same stock random days**

| Action | Trades | Not fillable | Win rate | Avg return | Median | Worst | Profit factor | Avg P&L per lot (Rs) |
|---|---|---|---|---|---|---|---|---|
| BUY CE + BUY PE | 133 | 14 | 30.8% | -4.5% | -9.3% | -61.9% | 0.77 | -896 |

## What drives the model (permutation importance, holdout AUC drop)

- hv60: 0.0732
- iv_hv: 0.0207
- iv_z60: 0.0154
- atm_iv: 0.0123
- straddle_pct: 0.0072
- atr_pct: 0.0055
- c_ema50: 0.0039
- nifty_ret5: 0.0038
- vix: 0.0024
- hv_ratio_10_60: 0.0021
- iv_chg5: 0.0015
- pcr_z60: 0.0012
- atr_ratio_14_50: 0.0009
- c_ema20: 0.0008
- pcr_chg5: 0.0006
- bb_width: 0.0006
- bb_width_rank: 0.0005
- short_covering: 0.0004
- vix_rank: 0.0004
- ema20_slope5: 0.0003
