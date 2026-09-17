import datetime
import logging
import threading

import pytest

from app.services import alerts, research_job

NOW = datetime.datetime(2026, 9, 17, 15, 30, tzinfo=datetime.timezone.utc)

ROWS = [
    {"ticker": "ABCD.SW", "name": "Alpha AG", "change_pct": -7.25, "price": 12.3, "volume_vs_10d_avg": 0.4},
    {"ticker": "WXYZ.SW", "name": "Omega SA", "change_pct": -5.1, "price": 88.0, "volume_vs_10d_avg": None},
]


@pytest.fixture(autouse=True)
def _no_pending_alert(monkeypatch):
    monkeypatch.setattr(alerts, "_pending_started_at", None)


# --- format_alert ---


def test_format_alert_lists_each_row_with_change_and_volume_ratio():
    subject, body = alerts.format_alert(
        {"status": "done", "today_screener": ROWS, "universe_size": 120, "failed_ticker_count": 3}, NOW,
    )
    assert subject == "[Swiss big-loss] 2 stock(s) 2026-09-17 15:30 UTC"
    assert "ABCD.SW      -7.25%   Alpha AG" in body
    assert "0.40x 10d avg volume, price 12.3" in body
    assert "WXYZ.SW      -5.10%   Omega SA" in body
    assert "volume ratio n/a, price 88.0" in body
    assert "Universe: 120 tickers, 3 failed to fetch." in body


def test_format_alert_says_none_when_nothing_matched():
    subject, body = alerts.format_alert(
        {"status": "done", "today_screener": [], "universe_size": 120, "failed_ticker_count": 0}, NOW,
    )
    assert subject == "[Swiss big-loss] none today 2026-09-17 15:30 UTC"
    assert "No Swiss stock is down 5% or more today." in body


def test_format_alert_reports_a_failed_scan_instead_of_staying_silent():
    subject, body = alerts.format_alert({"status": "error", "error": "Yahoo Finance is currently rate-limiting"}, NOW)
    assert subject == "[Swiss big-loss] scan FAILED 2026-09-17 15:30 UTC"
    assert "Yahoo Finance is currently rate-limiting" in body


def test_format_alert_reports_a_timed_out_wait():
    subject, body = alerts.format_alert({"status": "timeout", "error": "scan still running after 900s"}, NOW)
    assert "FAILED" in subject
    assert "still running" in body


# --- start_today_alert / _wait_for_scan ---


def test_start_today_alert_starts_the_scan_and_emails_when_it_finishes(monkeypatch):
    statuses = iter([
        {"status": "running", "started_at": "2026-09-17T15:30:00+00:00"},
        {"status": "done", "started_at": "2026-09-17T15:30:00+00:00", "today_screener": ROWS, "universe_size": 1, "failed_ticker_count": 0},
    ])
    monkeypatch.setattr(research_job, "start_today_scan", lambda: {"status": "running", "started_at": "2026-09-17T15:30:00+00:00"})
    monkeypatch.setattr(research_job, "get_today_status", lambda: next(statuses))
    monkeypatch.setattr(alerts, "POLL_INTERVAL_SECONDS", 0)
    sent = []
    done = threading.Event()

    def sender(subject, body):
        sent.append((subject, body))
        done.set()

    job = alerts.start_today_alert(sender=sender)

    assert job["status"] == "running"
    assert job["alert"] == "pending"
    assert done.wait(timeout=5)
    assert sent[0][0].startswith("[Swiss big-loss] 2 stock(s)")


def test_start_today_alert_called_twice_for_the_same_scan_sends_one_email(monkeypatch):
    started_at = "2026-09-17T16:00:00+00:00"
    release = threading.Event()
    monkeypatch.setattr(research_job, "start_today_scan", lambda: {"status": "running", "started_at": started_at})
    monkeypatch.setattr(
        research_job, "get_today_status",
        lambda: {"status": "done", "started_at": started_at, "today_screener": []} if release.is_set()
        else {"status": "running", "started_at": started_at},
    )
    monkeypatch.setattr(alerts, "POLL_INTERVAL_SECONDS", 0)
    sent = []
    finished = threading.Event()

    def sender(subject, body):
        sent.append(subject)
        finished.set()

    first = alerts.start_today_alert(sender=sender)
    second = alerts.start_today_alert(sender=sender)
    release.set()

    assert first["alert"] == "pending"
    assert second["alert"] == "already_pending"
    assert finished.wait(timeout=5)
    assert len(sent) == 1


def test_wait_for_scan_accepts_a_newer_finished_scan_instead_of_waiting_for_an_exact_started_at_match(monkeypatch):
    monkeypatch.setattr(
        research_job, "get_today_status",
        lambda: {"status": "done", "started_at": "2026-09-17T15:31:00+00:00", "today_screener": []},
    )
    status = alerts._wait_for_scan("2026-09-17T15:30:00+00:00")
    assert status["status"] == "done"


def test_wait_for_scan_keeps_waiting_for_an_older_result_then_times_out(monkeypatch):
    monkeypatch.setattr(
        research_job, "get_today_status",
        lambda: {"status": "done", "started_at": "2026-09-17T15:29:00+00:00", "today_screener": []},
    )
    monkeypatch.setattr(alerts, "SCAN_WAIT_SECONDS", 0)
    status = alerts._wait_for_scan("2026-09-17T15:30:00+00:00")
    assert status["status"] == "timeout"


def test_run_alert_logs_and_survives_a_sender_failure(monkeypatch, caplog):
    monkeypatch.setattr(alerts, "_wait_for_scan", lambda started_at: {"status": "done", "today_screener": []})

    def broken_sender(subject, body):
        raise ConnectionError("smtp down")

    with caplog.at_level(logging.ERROR, logger="app.services.alerts"):
        alerts._run_alert("2026-09-17T15:30:00+00:00", broken_sender)

    assert "failed to send" in caplog.text
    assert "smtp down" in caplog.text


def test_finished_at_or_after_compares_timestamps_not_strings():
    base = "2026-09-17T15:30:00+00:00"
    assert alerts._finished_at_or_after({"status": "done", "started_at": "2026-09-17T15:30:00.000005+00:00"}, base)
    assert alerts._finished_at_or_after({"status": "done", "started_at": "2026-09-17T17:30:00+02:00"}, base)
    assert not alerts._finished_at_or_after({"status": "done", "started_at": "2026-09-17T15:29:59+00:00"}, base)
    assert not alerts._finished_at_or_after({"status": "running", "started_at": base}, base)
    assert not alerts._finished_at_or_after({"status": "idle"}, base)


# --- is_email_configured ---


REQUIRED_EMAIL_SETTINGS = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO")


@pytest.mark.parametrize("missing", REQUIRED_EMAIL_SETTINGS)
def test_is_email_configured_is_false_when_any_required_value_is_missing(monkeypatch, missing):
    from app import config

    for name in REQUIRED_EMAIL_SETTINGS:
        monkeypatch.setattr(config, name, ("x",) if name == "ALERT_EMAIL_TO" else "x")
    assert alerts.is_email_configured() is True
    monkeypatch.setattr(config, missing, () if missing == "ALERT_EMAIL_TO" else None)
    assert alerts.is_email_configured() is False


def test_send_email_addresses_every_recipient(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(config, "SMTP_USER", "bot@x.test")
    monkeypatch.setattr(config, "SMTP_PASSWORD", "pw")
    monkeypatch.setattr(config, "ALERT_EMAIL_FROM", "bot@x.test")
    monkeypatch.setattr(config, "ALERT_EMAIL_TO", ("a@x.test", "b@y.test"))
    calls = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls["connect"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            calls["tls"] = True

        def login(self, user, password):
            calls["login"] = user

        def send_message(self, msg, to_addrs=None):
            calls["to_header"] = msg["To"]
            calls["to_addrs"] = to_addrs

    monkeypatch.setattr(alerts.smtplib, "SMTP", FakeSMTP)
    alerts.send_email("subj", "body")

    assert calls["to_header"] == "a@x.test, b@y.test"
    assert calls["to_addrs"] == ["a@x.test", "b@y.test"]
    assert calls["tls"] and calls["login"] == "bot@x.test"
