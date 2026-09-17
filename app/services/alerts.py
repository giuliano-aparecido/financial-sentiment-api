"""Emails the result of a today/big-loss scan (research_job.py) to
ALERT_EMAIL_TO. Triggered by POST /api/research/volatility/today/alert,
which an external scheduler (.github/workflows/big-loss-alert.yml) calls
twice a day - an in-process cron can't do this because the Render free
tier sleeps the process between requests.
"""

import datetime
import logging
import smtplib
import threading
import time
from collections.abc import Callable
from email.message import EmailMessage

from app import config
from app.services import research_job
from app.services.swiss_today_screener import LOSS_THRESHOLD_PCT

logger = logging.getLogger(__name__)

Sender = Callable[[str, str], None]

SCAN_WAIT_SECONDS = 10 * 60
POLL_INTERVAL_SECONDS = 5

_pending_lock = threading.Lock()
_pending_started_at: str | None = None


def is_email_configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASSWORD and config.ALERT_EMAIL_TO)


def format_alert(status: dict, now: datetime.datetime) -> tuple[str, str]:
    """(subject, body) for a finished today-scan status dict."""
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    if status.get("status") != "done":
        return (
            f"[Swiss big-loss] scan FAILED {stamp}",
            f"The big-loss scan did not complete.\n\nStatus: {status.get('status')}\nError: {status.get('error', 'unknown')}\n",
        )

    rows = status.get("today_screener") or []
    universe = status.get("universe_size")
    failed = status.get("failed_ticker_count") or 0
    footer = f"\nUniverse: {universe} tickers, {failed} failed to fetch.\n"

    if not rows:
        return (
            f"[Swiss big-loss] none today {stamp}",
            f"No Swiss stock is down {abs(LOSS_THRESHOLD_PCT):g}% or more today.\n{footer}",
        )

    lines = [f"{len(rows)} Swiss stock(s) down {abs(LOSS_THRESHOLD_PCT):g}% or more today (thinnest volume first):", ""]
    for r in rows:
        ratio = r.get("volume_vs_10d_avg")
        ratio_text = f"{ratio:.2f}x 10d avg volume" if ratio is not None else "volume ratio n/a"
        lines.append(f"{r['ticker']:<12} {r['change_pct']:+.2f}%   {r.get('name') or ''}")
        lines.append(f"{'':<12} {ratio_text}, price {r.get('price')}")
    return (f"[Swiss big-loss] {len(rows)} stock(s) {stamp}", "\n".join(lines) + "\n" + footer)


def send_email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.ALERT_EMAIL_FROM
    msg["To"] = ", ".join(config.ALERT_EMAIL_TO)
    msg.set_content(body)
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
        subject, body = format_alert(status, datetime.datetime.now(datetime.timezone.utc))
        try:
            sender(subject, body)
            logger.info("Big-loss alert sent: %s", subject)
        except Exception:
            logger.exception("Big-loss alert email failed to send")
    finally:
        with _pending_lock:
            if _pending_started_at == started_at:
                _pending_started_at = None


def start_today_alert(sender: Sender = send_email) -> dict:
    """Starts (or joins) the today scan and emails its result when it
    finishes. Returns the scan's start status immediately; a second call
    while an alert is already waiting on the same scan sends nothing extra."""
    global _pending_started_at
    job = research_job.start_today_scan()
    with _pending_lock:
        if _pending_started_at == job["started_at"]:
            return {**job, "alert": "already_pending"}
        _pending_started_at = job["started_at"]
    threading.Thread(target=_run_alert, args=(job["started_at"], sender), daemon=True).start()
    return {**job, "alert": "pending"}
