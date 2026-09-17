"""Emails the matches of a today/big-loss scan (research_job.py) to
ALERT_EMAIL_TO - only when there is at least one; an empty or failed scan
is logged, not mailed. Triggered by POST /api/research/volatility/today/alert,
which an external scheduler (.github/workflows/big-loss-alert.yml) calls
twice a day - an in-process cron can't do this because the Render free
tier sleeps the process between requests.
"""

import datetime
import html
import logging
import smtplib
import threading
import time
from collections.abc import Callable
from email.message import EmailMessage
from urllib.parse import quote

from app import config
from app.services import research_job
from app.services.swiss_today_screener import LOSS_THRESHOLD_PCT

logger = logging.getLogger(__name__)

Sender = Callable[[str, str, str], None]

YAHOO_QUOTE_URL = "https://finance.yahoo.com/quote/"

SCAN_WAIT_SECONDS = 10 * 60
POLL_INTERVAL_SECONDS = 5

_pending_lock = threading.Lock()
_pending_started_at: str | None = None


def is_email_configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASSWORD and config.ALERT_EMAIL_TO)


def yahoo_quote_url(ticker: str) -> str:
    return YAHOO_QUOTE_URL + quote(ticker, safe="")


def format_alert(rows: list[dict], status: dict, now: datetime.datetime) -> tuple[str, str, str]:
    """(subject, plain-text body, HTML body) for a scan with at least one match."""
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    threshold = f"{abs(LOSS_THRESHOLD_PCT):g}"
    heading = f"{len(rows)} Swiss stock(s) down {threshold}% or more today (thinnest volume first)"
    footer = f"Universe: {status.get('universe_size')} tickers, {status.get('failed_ticker_count') or 0} failed to fetch."

    text_lines = [heading + ":", ""]
    html_rows = []
    for r in rows:
        ratio = r.get("volume_vs_10d_avg")
        ratio_text = f"{ratio:.2f}x 10d avg volume" if ratio is not None else "volume ratio n/a"
        url = yahoo_quote_url(r["ticker"])
        text_lines.append(f"{r['ticker']:<12} {r['change_pct']:+.2f}%   {r.get('name') or ''}")
        text_lines.append(f"{'':<12} {ratio_text}, price {r.get('price')}")
        text_lines.append(f"{'':<12} {url}")
        html_rows.append(
            "<tr>"
            f'<td><a href="{html.escape(url)}">{html.escape(r["ticker"])}</a></td>'
            f"<td>{r['change_pct']:+.2f}%</td>"
            f"<td>{html.escape(r.get('name') or '')}</td>"
            f"<td>{html.escape(ratio_text)}</td>"
            f"<td>{html.escape(str(r.get('price')))}</td>"
            "</tr>"
        )
    text = "\n".join(text_lines) + "\n\n" + footer + "\n"
    html_body = (
        f"<p>{html.escape(heading)}:</p>"
        '<table cellpadding="4">'
        "<tr><th align=\"left\">Ticker</th><th align=\"left\">Change</th><th align=\"left\">Name</th>"
        "<th align=\"left\">Volume</th><th align=\"left\">Price</th></tr>"
        + "".join(html_rows)
        + "</table>"
        f"<p>{html.escape(footer)}</p>"
    )
    return (f"[Swiss big-loss] {len(rows)} stock(s) {stamp}", text, html_body)


def send_email(subject: str, text: str, html_body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.ALERT_EMAIL_FROM
    msg["To"] = ", ".join(config.ALERT_EMAIL_TO)
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
        smtp.send_message(msg, to_addrs=list(config.ALERT_EMAIL_TO))


def _finished_at_or_after(status: dict, started_at: str) -> bool:
    if status.get("status") == "running" or not status.get("started_at"):
        return False
    return datetime.datetime.fromisoformat(status["started_at"]) >= datetime.datetime.fromisoformat(started_at)


def _wait_for_scan(started_at: str) -> dict:
    deadline = time.monotonic() + SCAN_WAIT_SECONDS
    while time.monotonic() < deadline:
        status = research_job.get_today_status()
        if _finished_at_or_after(status, started_at):
            return status
        time.sleep(POLL_INTERVAL_SECONDS)
    return {"status": "timeout", "error": f"scan still running after {SCAN_WAIT_SECONDS}s"}


def _run_alert(started_at: str, sender: Sender) -> None:
    global _pending_started_at
    try:
        status = _wait_for_scan(started_at)
        if status.get("status") != "done":
            logger.warning("Big-loss alert: scan did not complete (%s: %s) - no email", status.get("status"), status.get("error"))
            return
        rows = status.get("today_screener") or []
        if not rows:
            logger.info("Big-loss alert: no stock down %g%% or more today - no email", abs(LOSS_THRESHOLD_PCT))
            return
        subject, text, html_body = format_alert(rows, status, datetime.datetime.now(datetime.timezone.utc))
        try:
            sender(subject, text, html_body)
            logger.info("Big-loss alert sent: %s", subject)
        except Exception:
            logger.exception("Big-loss alert email failed to send")
    finally:
        with _pending_lock:
            if _pending_started_at == started_at:
                _pending_started_at = None


def start_today_alert(sender: Sender = send_email) -> dict:
    """Starts (or joins) the today scan and, if it finds any match, emails
    the table when it finishes. Returns the scan's start status
    immediately; a second call while an alert is already waiting on the
    same scan sends nothing extra."""
    global _pending_started_at
    job = research_job.start_today_scan()
    with _pending_lock:
        if _pending_started_at == job["started_at"]:
            return {**job, "alert": "already_pending"}
        _pending_started_at = job["started_at"]
    threading.Thread(target=_run_alert, args=(job["started_at"], sender), daemon=True).start()
    return {**job, "alert": "pending"}
