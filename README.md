# Small cap fund buys alert

Emails you when a mutual fund house starts a new position in a small cap stock
(for example a fund house first buying Gabriel India).

Every fund house publishes the full portfolio of each scheme by the 10th of the next month.
The "Fund house small cap buys" GitHub Actions workflow runs at 19:00 IST every day from the
1st to the 15th, reads any newly published month, and emails the small cap stocks a fund house
now holds that it held in none of its schemes the month before.

- Small cap = ranked 251st or lower in AMFI's half-yearly average market cap list (SEBI's definition).
- Skips index, ETF and arbitrage schemes, schemes new this month, and positions under 0.25% of a
  scheme unless worth Rs 10 crore or more.
- Fund houses read are listed in `fno_alert/mfsources.py` (SBI, Nippon, ABSL, DSP, Tata, Motilal,
  PPFAS, Mirae, Invesco, Baroda BNP); the email names any it could not read.
- Saved holdings: `state/mfwatch/holdings/<fund house>/<YYYY-MM>.csv.gz`; every alert is logged in
  `state/mfwatch/alerts_log.csv`.

> Not investment advice. A fund house buying a stock is information, not a recommendation.

## Setup

Repository secrets (Settings > Secrets and variables > Actions): `SMTP_USER`, `SMTP_PASSWORD`
(a Gmail app password) and `ALERT_TO`. Run the "Send test email" workflow to check them.

## Run it yourself

```
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env     # fill in the SMTP settings
python -m fno_alert mfwatch --dry-run --only dsp,ppfas    # print findings, no email
python -m fno_alert test-email
python -m pytest tests
```

## History

This repo used to hold an F&O big-move scanner, intraday and multi-strategy option selling
paper desks, and momentum research. They were removed on 2026-11-11 and can be recovered
from git history (any commit before that date).
