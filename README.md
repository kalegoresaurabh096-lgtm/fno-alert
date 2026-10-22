# F&O big-move alerts (NSE stocks)

Emails you when an NSE F&O **stock** (indices excluded) shows the conditions that
historically come before a big move, up or down, and suggests an options trade
(Buy CE, Buy PE, Sell PE, Sell CE, or Buy CE + Buy PE).

> **Not investment advice.** No model can be 100% accurate: prices react to news
> nobody can see coming. This tool tells you how often its alerts were right in
> a strict backtest, so you can decide how much to trust them.

## What it looks at

| Group | Signals |
|---|---|
| Volatility compression | Bollinger band width rank, TTM squeeze (Bollinger inside Keltner) and its release, ATR contraction, NR7 / inside days, tight 20-day range |
| Ignition | Relative volume, range expansion vs ATR, 20/55-day breakouts and breakdowns, gaps |
| Momentum | RSI, MACD and crosses, RSI/MACD divergences, ADX rising from a low base, EMA 20/50/200 trend |
| Option chain | ATM implied volatility (own Black-76 calc), IV rank and IV change, IV vs realised volatility, ATM straddle (move priced in), put skew, PCR (OI and volume), fresh call vs put writing near the price, option volume spikes, max pain, call/put OI walls |
| Futures | OI change, OI build-up type (long build-up, short build-up, short covering, long unwinding), futures volume, basis |
| Market | India VIX level/rank/change, NIFTY trend and volatility |
| Events | Days to the next announced results meeting (when the data source has it) |

## How a "big move" is defined and tested

* **Big move**: within the next 5 trading days, the stock closes at least
  **2x its normal 5-day move** away from the **next day's open** (never less
  than 4%). The overnight gap after the signal is not counted, because the
  end-of-day email arrives after the close and you can only act at the next open.
* A gradient-boosted model (plus a logistic model and a simple points score as
  baselines) predicts the chance of a big move up and a big move down.
* **Walk-forward**: retrained every quarter on past data only, with a 5-day
  embargo so no training label overlaps the test period.
* The alert threshold, direction rule and trade rules are chosen on the first
  part of the out-of-sample period. The numbers are reported on a later
  **holdout** that was never used for any choice (with a 5-day gap between them).
* Alerts are counted the way they are emailed: the same stock is not
  re-alerted within 5 sessions unless its probability jumps by 10 points, so one
  volatile week is not counted as five "hits".
* **Direction (UP/DOWN)** is only called if, on the validation period, the
  calls beat a coin flip with statistical significance (binomial test, p < 0.05).
  Otherwise every alert is direction-neutral (straddle or no trade).
* 95% confidence intervals use a block bootstrap over calendar months.
* Suggested trades are replayed on real historical option prices: enter at the
  **next day's open** (only if that contract traded), exit after 5 sessions and
  always before expiry (no physical settlement), using an expiry at least
  12 days away. Costs: 2% slippage per side (at least Rs 0.15), brokerage,
  exchange and SEBI fees, GST, stamp duty and STT. The same trades on random
  days of the same stocks are the baseline.

Results go to `reports/BACKTEST.md`.

## Trade suggestion rules

| Model says | Options cheap (IV rank < 75% and IV/HV < 1.6) | Options expensive |
|---|---|---|
| Big move, UP | BUY CE (ATM) | SELL PE (below support) |
| Big move, DOWN | BUY PE (ATM) | SELL CE (above resistance) |
| Big move, direction unclear | BUY CE + BUY PE (straddle) | NO TRADE |

Thresholds are re-tuned on the validation period during training.

## Setup

### 1. Upstox read-only token (live data, free)
1. Open an Upstox account.
2. Go to the Upstox **Developer Apps** page, open the **Analytics** tab and click **Generate Token**.
   It is read-only (cannot trade), lasts 1 year and needs no daily login.
3. Keep it secret. You'll paste it as a GitHub secret (below) or into `.env`.

### 2. Gmail app password (sending the email)
1. Turn on 2-Step Verification for your Google account.
2. Create an **App password** (Google Account > Security > App passwords) and copy the 16 characters.

### 3a. Run on GitHub Actions (recommended, free)
1. Create a **private** GitHub repository and push this folder to it.
2. Settings > Secrets and variables > Actions > New repository secret, add:
   `UPSTOX_ACCESS_TOKEN`, `SMTP_USER` (your Gmail), `SMTP_PASSWORD` (app password), `ALERT_TO` (where alerts go).
3. The workflow in `.github/workflows/scan.yml` runs Monday to Friday at about
   10:30, 12:30 and 14:45 IST (intraday estimates) and 15:45 IST (end of day,
   the backtested signal). It saves history to `state/` in the repo.
4. Actions tab > F&O scan > Run workflow, with `dry_run=true`, to test.

### 3b. Or run on your Mac
```bash
cd ~/dev/fno-alert
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then edit .env
python -m fno_alert test-email
python -m fno_alert scan --dry-run --symbols RELIANCE,TCS
```
Schedule `python -m fno_alert scan` with `cron` or `launchd` (the Mac must be awake).

## Backtest and training

```bash
python -m fno_alert build-data      # data/raw -> data/processed
python -m fno_alert train           # walk-forward backtest + final model in models/
python -m fno_alert seed-state      # give the live scanner its history (IV rank, z-scores) from day one
```
Until a model is trained, the scanner uses the simple points score and marks
emails `[UNVALIDATED RULES]`.

## Commands
| Command | What it does |
|---|---|
| `python -m fno_alert scan [--dry-run] [--mode intraday\|eod] [--symbols A,B]` | Live scan and email |
| `python -m fno_alert test-email` | Send a sample email |
| `python -m fno_alert build-data` | Build research tables from downloads |
| `python -m fno_alert backtest` / `train` | Backtest only / backtest + save model |
| `python -m pytest -q tests` | Tests (no network needed) |

## Known limits
* Intraday scans project the full day's volume from a typical volume curve, so they
  are early estimates and are **not backtested**. The 15:45 IST end-of-day run is
  the backtested signal. Set `intraday = false` under `[alerts]` to get only that one.
* Results-day effects are not in the model yet: there is no results calendar
  available live within the data providers' terms.
* Splits/bonuses in live broker data are detected from the price gap plus the
  F&O lot-size change; a detected one is flagged in the email.
* News, results surprises and market-wide shocks can't be predicted from price
  and option data. Expect most alerts to be false alarms. The backtest tells you
  the true rate.
* Option selling has large risk. Margins shown are rough (about 20% of contract value).
