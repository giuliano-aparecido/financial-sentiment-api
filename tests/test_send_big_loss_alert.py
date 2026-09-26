import http.client
import importlib.util
import pathlib
import urllib.error

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "send_big_loss_alert.py"
_spec = importlib.util.spec_from_file_location("send_big_loss_alert", _SCRIPT)
sender = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sender)

ENV = {
    "API_BASE_URL": "https://api.test/",
    "API_KEY": "k",
    "SMTP_HOST": "smtp.test",
    "SMTP_PORT": "",
    "SMTP_USER": "bot@x.test",
    "SMTP_PASSWORD": "pw",
    "ALERT_EMAIL_FROM": "",
    "ALERT_EMAIL_TO": " a@x.test, b@y.test ,,",
}
EMAIL = {"subject": "subj", "text": "plain body", "html": "<p>html body</p>"}
STARTED = "2026-09-17T15:30:00.123456+00:00"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sender.time, "sleep", lambda seconds: None)


class FakeAPI:
    def __init__(self, polls, start_failures=0):
        self.polls = iter(polls)
        self.start_failures = start_failures
        self.calls = []

    def __call__(self, method, url, api_key):
        self.calls.append((method, url))
        if method == "POST":
            if self.start_failures:
                self.start_failures -= 1
                raise urllib.error.URLError("cold start")
            return {"status": "running", "started_at": STARTED}
        result = next(self.polls)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def sent(monkeypatch):
    sent = []
    monkeypatch.setattr(sender, "send_email", lambda email, env: sent.append(email))
    return sent


def test_sends_the_rendered_email_once_the_scan_is_done(monkeypatch, sent):
    api = FakeAPI([{"status": "running"}, {"status": "done", "match_count": 2, "email": EMAIL}], start_failures=2)
    monkeypatch.setattr(sender, "api_request", api)

    assert sender.main(ENV) == 0
    assert sent == [EMAIL]
    assert [m for m, _ in api.calls] == ["POST", "POST", "POST", "GET", "GET"]
    assert api.calls[0][1] == "https://api.test/api/research/volatility/today/alert"
    assert api.calls[-1][1].endswith("?started_at=2026-09-17T15%3A30%3A00.123456%2B00%3A00")


def test_no_match_sends_nothing_and_succeeds(monkeypatch, sent):
    monkeypatch.setattr(sender, "api_request", FakeAPI([{"status": "done", "match_count": 0, "email": None}]))
    assert sender.main(ENV) == 0
    assert sent == []


def test_a_transient_poll_failure_keeps_polling(monkeypatch, sent):
    api = FakeAPI([urllib.error.URLError("blip"), {"status": "done", "match_count": 2, "email": EMAIL}])
    monkeypatch.setattr(sender, "api_request", api)
    assert sender.main(ENV) == 0
    assert sent == [EMAIL]


def test_a_dropped_connection_while_polling_keeps_polling(monkeypatch, sent):
    api = FakeAPI([http.client.RemoteDisconnected("dropped"), {"status": "done", "match_count": 2, "email": EMAIL}])
    monkeypatch.setattr(sender, "api_request", api)
    assert sender.main(ENV) == 0
    assert sent == [EMAIL]


def test_a_client_error_while_polling_fails_fast(monkeypatch, sent):
    unauthorized = urllib.error.HTTPError("https://api.test", 401, "Unauthorized", None, None)
    api = FakeAPI([unauthorized, {"status": "done", "match_count": 2, "email": EMAIL}])
    monkeypatch.setattr(sender, "api_request", api)
    with pytest.raises(urllib.error.HTTPError):
        sender.main(ENV)
    assert sent == []


def test_a_client_error_on_start_is_not_retried(monkeypatch):
    calls = []

    def unauthorized(method, url, api_key):
        calls.append(method)
        raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(sender, "api_request", unauthorized)
    with pytest.raises(urllib.error.HTTPError):
        sender.main(ENV)
    assert calls == ["POST"]


def test_a_failed_scan_fails_the_run_without_email(monkeypatch, sent, capsys):
    monkeypatch.setattr(sender, "api_request", FakeAPI([{"status": "error", "error": "rate-limited"}]))
    assert sender.main(ENV) == 1
    assert sent == []
    assert "error: rate-limited" in capsys.readouterr().err


def test_a_scan_that_never_finishes_times_out(monkeypatch, sent):
    monkeypatch.setattr(sender, "SCAN_WAIT_SECONDS", 0)
    monkeypatch.setattr(sender, "api_request", FakeAPI([]))
    assert sender.main(ENV) == 1
    assert sent == []


def test_start_gives_up_after_the_last_attempt(monkeypatch):
    monkeypatch.setattr(sender, "api_request", FakeAPI([], start_failures=sender.START_ATTEMPTS))
    with pytest.raises(urllib.error.URLError):
        sender.main(ENV)


@pytest.mark.parametrize("missing", sender.REQUIRED_ENV)
def test_an_unset_secret_fails_before_starting_a_scan(monkeypatch, capsys, missing):
    api = FakeAPI([])
    monkeypatch.setattr(sender, "api_request", api)
    assert sender.main({**ENV, missing: ""}) == 1
    assert api.calls == []
    assert missing in capsys.readouterr().err


def test_send_email_addresses_every_recipient(monkeypatch):
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
            calls["from"] = msg["From"]
            calls["to_header"] = msg["To"]
            calls["to_addrs"] = to_addrs
            calls["parts"] = [part.get_content_type() for part in msg.iter_parts()]
            calls["html"] = msg.get_body(preferencelist=("html",)).get_content()

    monkeypatch.setattr(sender.smtplib, "SMTP", FakeSMTP)
    sender.send_email(EMAIL, ENV)

    assert calls["connect"] == ("smtp.test", 587)
    assert calls["from"] == "bot@x.test"
    assert calls["to_header"] == "a@x.test, b@y.test"
    assert calls["to_addrs"] == ["a@x.test", "b@y.test"]
    assert calls["tls"] and calls["login"] == "bot@x.test"
    assert calls["parts"] == ["text/plain", "text/html"]
    assert "<p>html body</p>" in calls["html"]
