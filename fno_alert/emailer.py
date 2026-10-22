"""Build and send the alert email (SMTP, e.g. Gmail with an app password)."""
from __future__ import annotations

import html
import os
import smtplib
import ssl
from email.message import EmailMessage

DISCLAIMER = ("Automated statistical signal, not investment advice. Backtested hit rates are "
              "historical and can fail in new market conditions. Options can lose 100% of the "
              "premium paid; option selling has large or unlimited risk. Size positions accordingly.")

ACTION_COLOR = {"BUY CE": "#0a7f3f", "SELL PE": "#0a7f3f", "BUY PE": "#b3261e", "SELL CE": "#b3261e",
                "BUY CE + BUY PE": "#6b4fbb", "NO TRADE": "#666666"}


def _pct(x, digits=1):
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(x, digits=2):
    try:
        v = float(x)
        return f"{v:,.{digits}f}" if v == v else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def render(alerts: list[dict], meta: dict, run_label: str) -> tuple[str, str, str]:
    """Return (subject, plain text, html)."""
    n = len(alerts)
    names = ", ".join(a["symbol"] for a in alerts[:6]) + ("..." if n > 6 else "")
    subject = f"F&O big-move alert: {names} ({run_label})"
    base = meta.get("base_rate")
    hit = meta.get("hit_rate")
    stats_line = ""
    if meta.get("direction_enabled") is False:
        stats_line_dir = (" The backtest found no reliable edge in predicting the direction, so trade ideas "
                          "are direction-neutral (straddles).")
    else:
        stats_line_dir = ""
    if hit and base:
        stats_line = (f"In the backtest holdout ({meta.get('holdout_period', '')}), stocks flagged at this "
                      f"threshold made a big move within 5 trading days {hit:.0%} of the time, "
                      f"vs {base:.0%} for a random F&O stock-day." + stats_line_dir)
    text = [subject, "", stats_line, ""]
    cards = []
    for a in alerts:
        act = a.get("action", "")
        color = ACTION_COLOR.get(act, "#333")
        lines = [f"{a['symbol']}  {act}  ({a.get('direction', '')})",
                 f"  Big-move probability {_pct(a.get('p_big'), 0)} (normal {_pct(base, 0)})",
                 f"  Price {_num(a.get('spot'))}  change today {_pct(a.get('ret1'))}",
                 f"  Trade idea: {a.get('trade_text', '')}",
                 f"  Options price in a move of +/-{_pct(a.get('straddle_pct'))} to expiry {a.get('expiry', '')}",
                 f"  Support (put OI wall) {_num(a.get('put_wall'), 1)}  Resistance (call OI wall) "
                 f"{_num(a.get('call_wall'), 1)}  Max pain {_num(a.get('max_pain'), 1)}  PCR {_num(a.get('pcr_oi'))}",
                 "  Why:"] + [f"   - {r}" for r in a.get("reasons", [])]
        text += lines + [""]
        why = "".join(f"<li>{html.escape(r)}</li>" for r in a.get("reasons", []))
        cards.append(f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:14px 16px;margin:12px 0;font-family:Arial,sans-serif">
  <div style="font-size:18px;font-weight:bold">{html.escape(a['symbol'])}
    <span style="background:{color};color:#fff;border-radius:6px;padding:2px 8px;font-size:14px;margin-left:8px">{html.escape(act)}</span>
    <span style="color:#555;font-size:14px;margin-left:6px">{html.escape(a.get('direction', ''))}</span></div>
  <div style="margin-top:6px;font-size:14px">Big-move probability <b>{_pct(a.get('p_big'), 0)}</b>
    (normal {_pct(base, 0)}) &middot; Price {_num(a.get('spot'))} ({_pct(a.get('ret1'))} today)</div>
  <div style="margin-top:6px;font-size:14px"><b>Trade idea:</b> {html.escape(a.get('trade_text', ''))}</div>
  <div style="margin-top:4px;font-size:13px;color:#444">{html.escape(a.get('trade_why', ''))}</div>
  <table style="margin-top:8px;font-size:13px;border-collapse:collapse">
    <tr><td style="padding:2px 10px 2px 0">ATM IV</td><td>{_pct(a.get('atm_iv'), 0)} (rank {_pct(a.get('iv_rank'), 0)} of 1y)</td></tr>
    <tr><td style="padding:2px 10px 2px 0">Move priced in to {html.escape(str(a.get('expiry', '')))}</td><td>&plusmn;{_pct(a.get('straddle_pct'))}</td></tr>
    <tr><td style="padding:2px 10px 2px 0">Support / resistance (OI walls)</td><td>{_num(a.get('put_wall'), 1)} / {_num(a.get('call_wall'), 1)}</td></tr>
    <tr><td style="padding:2px 10px 2px 0">Max pain &middot; PCR</td><td>{_num(a.get('max_pain'), 1)} &middot; {_num(a.get('pcr_oi'))}</td></tr>
    <tr><td style="padding:2px 10px 2px 0">Futures OI change</td><td>{_pct(a.get('fut_oi_chg'))} ({html.escape(a.get('buildup', ''))})</td></tr>
  </table>
  <div style="margin-top:8px;font-size:13px"><b>Why:</b><ul style="margin:4px 0 0 18px;padding:0">{why}</ul></div>
</div>""")
    text += ["", DISCLAIMER]
    body = f"""<html><body style="max-width:680px;margin:auto">
<h2 style="font-family:Arial,sans-serif">F&amp;O big-move alerts &middot; {html.escape(run_label)}</h2>
<p style="font-family:Arial,sans-serif;font-size:14px;color:#333">{html.escape(stats_line)}</p>
{''.join(cards)}
<p style="font-family:Arial,sans-serif;font-size:12px;color:#777">{html.escape(DISCLAIMER)}</p>
</body></html>"""
    return subject, "\n".join(text), body


def send(subject: str, text: str, html_body: str, to: str | None = None) -> None:
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    to = to or os.environ.get("ALERT_TO", user)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_FROM", user)
    msg["To"] = to
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=60) as s:
            s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=60) as s:
            s.starttls(context=ctx)
            s.login(user, password)
            s.send_message(msg)
