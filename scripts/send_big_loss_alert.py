"""Runs the big-loss alert from the GitHub Actions runner
(.github/workflows/big-loss-alert.yml): starts the today scan on the API,
polls GET /api/research/volatility/today/alert until it finishes, and
sends the email the API rendered over SMTP from here. Render's free tier
blocks outbound SMTP ports (25/465/587), so the API can't send it itself.

Standard library only, so the workflow needs no `pip install`.

Exits non-zero when the scan fails, times out, or the email can't be
sent, so GitHub's failed-run notification is the error signal. A scan
with no match sends nothing and exits 0.

Required environment variables (GitHub Actions secrets):
    API_BASE_URL, API_KEY       - the deployed API and its X-API-Key
    SMTP_HOST, SMTP_USER, SMTP_PASSWORD
    ALERT_EMAIL_TO              - one or more recipients, comma-separated
Optional: SMTP_PORT (default 587, STARTTLS), ALERT_EMAIL_FROM (default
SMTP_USER).
"""

import http.client
import json
import os
import smtplib
import sys
import time
import urllib.error
import urllib.request
from email.message import EmailMessage
from urllib.parse import quote

ALERT_PATH = "/api/research/volatility/today/alert"

START_ATTEMPTS = 4
START_RETRY_DELAY_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 15
SCAN_WAIT_SECONDS = 12 * 60


def _is_transient(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return isinstance(error, (OSError, http.client.HTTPException, ValueError))


def api_request(method: str, url: str, api_key: str) -> dict:
    req = urllib.request.Request(url, method=method, headers={"X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        return json.load(resp)


def start_scan(base_url: str, api_key: str) -> str:
    for attempt in range(1, START_ATTEMPTS + 1):
        try:
            return api_request("POST", base_url + ALERT_PATH, api_key)["started_at"]
        except Exception as e:
            if not _is_transient(e) or attempt == START_ATTEMPTS:
                raise
            print(f"Start attempt {attempt} failed ({e}); retrying in {START_RETRY_DELAY_SECONDS}s")
            time.sleep(START_RETRY_DELAY_SECONDS)


def wait_for_alert(base_url: str, api_key: str, started_at: str) -> dict:
    url = f"{base_url}{ALERT_PATH}?started_at={quote(started_at, safe='')}"
    deadline = time.monotonic() + SCAN_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            alert = api_request("GET", url, api_key)
            if alert["status"] != "running":
                return alert
        except Exception as e:
            if not _is_transient(e):
                raise
            print(f"Poll failed ({e}); retrying")
        time.sleep(POLL_INTERVAL_SECONDS)
    return {"status": "timeout", "error": f"scan still running after {SCAN_WAIT_SECONDS}s"}


def parse_recipients(raw: str) -> list[str]:
    return [address.strip() for address in raw.split(",") if address.strip()]


def send_email(email: dict, env: dict) -> None:
    recipients = parse_recipients(env["ALERT_EMAIL_TO"])
    msg = EmailMessage()
    msg["Subject"] = email["subject"]
    msg["From"] = env.get("ALERT_EMAIL_FROM") or env["SMTP_USER"]
    msg["To"] = ", ".join(recipients)
    msg.set_content(email["text"])
    msg.add_alternative(email["html"], subtype="html")
    with smtplib.SMTP(env["SMTP_HOST"], int(env.get("SMTP_PORT") or 587), timeout=30) as smtp:
        smtp.starttls()
        smtp.login(env["SMTP_USER"], env["SMTP_PASSWORD"])
        smtp.send_message(msg, to_addrs=recipients)


REQUIRED_ENV = ("API_BASE_URL", "API_KEY", "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO")


def main(env: dict) -> int:
    # An unset GitHub secret arrives as an empty string, not a missing key.
    missing = [name for name in REQUIRED_ENV if not env.get(name)]
    if missing:
        print(f"Missing required settings: {', '.join(missing)}", file=sys.stderr)
        return 1
    base_url = env["API_BASE_URL"].rstrip("/")
    started_at = start_scan(base_url, env["API_KEY"])
    print(f"Scan started_at={started_at}")
    alert = wait_for_alert(base_url, env["API_KEY"], started_at)
    if alert["status"] != "done":
        print(f"Big-loss scan did not complete ({alert['status']}: {alert.get('error')})", file=sys.stderr)
        return 1
    if not alert["email"]:
        print("No stock down enough today - no email")
        return 0
    send_email(alert["email"], env)
    print(f"Sent: {alert['email']['subject']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(os.environ))
