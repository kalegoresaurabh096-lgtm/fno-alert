"""Command line entry point:  python -m fno_alert <command>

  mfwatch      email new small cap positions from fund houses' monthly portfolios
  test-email   send a sample email to check SMTP settings
"""
from __future__ import annotations

import argparse
import os


def load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_mfwatch(a):
    from . import mfwatch
    only = [s.strip().lower() for s in a.only.split(",")] if a.only else None
    print(mfwatch.run(a.state, a.dry_run, only))


def cmd_test_email(a):
    from . import emailer
    text = "This is a test email from your small cap fund buys alert. If you can read it, email works."
    emailer.send("[TEST] Small cap fund buys alert", text, f"<html><body><p>{text}</p></body></html>")
    print("Sent test email to", os.environ.get("ALERT_TO", os.environ.get("SMTP_USER")))


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser(prog="fno_alert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test-email")
    s = sub.add_parser("mfwatch", help="email new small cap positions from fund houses' monthly portfolios")
    s.add_argument("--state", default="state")
    s.add_argument("--dry-run", action="store_true", help="print findings, do not email or save state")
    s.add_argument("--only", help="comma-separated fund houses, e.g. dsp,ppfas")
    a = ap.parse_args(argv)
    {"test-email": cmd_test_email, "mfwatch": cmd_mfwatch}[a.cmd](a)
