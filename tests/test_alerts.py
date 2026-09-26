import datetime

from app.services import alerts

NOW = datetime.datetime(2026, 9, 17, 15, 30, tzinfo=datetime.timezone.utc)

ROWS = [
    {"ticker": "ABCD.SW", "name": "Alpha AG", "change_pct": -7.25, "price": 12.3, "volume_vs_10d_avg": 0.4},
    {"ticker": "WXYZ.SW", "name": "Omega SA", "change_pct": -5.1, "price": 88.0, "volume_vs_10d_avg": None},
]


# --- format_alert ---


DONE = {"status": "done", "today_screener": ROWS, "universe_size": 120, "failed_ticker_count": 3}


def test_format_alert_plain_text_lists_each_row_with_its_yahoo_link():
    subject, text, _html = alerts.format_alert(ROWS, DONE, NOW)
    assert subject == "[Swiss big-loss] 2 stock(s) 2026-09-17 15:30 UTC"
    assert "ABCD.SW      -7.25%   Alpha AG" in text
    assert "0.40x 10d avg volume, price 12.3" in text
    assert "https://finance.yahoo.com/quote/ABCD.SW" in text
    assert "WXYZ.SW      -5.10%   Omega SA" in text
    assert "volume ratio n/a, price 88.0" in text
    assert "Universe: 120 tickers, 3 failed to fetch." in text


def test_format_alert_html_links_each_ticker_to_its_yahoo_quote_page():
    _subject, _text, html_body = alerts.format_alert(ROWS, DONE, NOW)
    assert '<a href="https://finance.yahoo.com/quote/ABCD.SW">ABCD.SW</a>' in html_body
    assert '<a href="https://finance.yahoo.com/quote/WXYZ.SW">WXYZ.SW</a>' in html_body
    assert "<td>-7.25%</td>" in html_body
    assert "Universe: 120 tickers, 3 failed to fetch." in html_body


def test_format_alert_escapes_html_in_names_and_urls_in_tickers():
    rows = [{"ticker": "A&B.SW", "name": "<Evil> & Co", "change_pct": -6.0, "price": 1.0, "volume_vs_10d_avg": 1.0}]
    _subject, text, html_body = alerts.format_alert(rows, {"universe_size": 1, "failed_ticker_count": 0}, NOW)
    assert "&lt;Evil&gt; &amp; Co" in html_body
    assert "<Evil>" not in html_body
    assert 'href="https://finance.yahoo.com/quote/A%26B.SW"' in html_body
    assert "https://finance.yahoo.com/quote/A%26B.SW" in text


def test_yahoo_quote_url_matches_the_web_pages_link_format():
    assert alerts.yahoo_quote_url("NESN.SW") == "https://finance.yahoo.com/quote/NESN.SW"


# --- alert_for ---

STARTED = "2026-09-17T15:30:00+00:00"


def test_alert_for_is_running_while_the_scan_is_running():
    assert alerts.alert_for({"status": "running", "started_at": STARTED}, STARTED, NOW) == {"status": "running"}


def test_alert_for_is_running_while_only_an_older_scan_has_finished():
    older = {"status": "done", "started_at": "2026-09-17T15:29:00+00:00", "today_screener": ROWS}
    assert alerts.alert_for(older, STARTED, NOW) == {"status": "running"}


def test_alert_for_renders_the_email_when_the_scan_found_matches():
    alert = alerts.alert_for({**DONE, "started_at": STARTED}, STARTED, NOW)
    assert alert["status"] == "done"
    assert alert["match_count"] == 2
    assert alert["email"]["subject"] == "[Swiss big-loss] 2 stock(s) 2026-09-17 15:30 UTC"
    assert "ABCD.SW" in alert["email"]["text"]
    assert "finance.yahoo.com/quote/ABCD.SW" in alert["email"]["html"]


def test_alert_for_has_no_email_when_nothing_is_down_enough():
    status = {"status": "done", "started_at": STARTED, "today_screener": [], "universe_size": 120}
    assert alerts.alert_for(status, STARTED, NOW) == {"status": "done", "match_count": 0, "email": None}


def test_alert_for_reports_a_failed_scan():
    status = {"status": "error", "started_at": STARTED, "error": "Yahoo Finance is currently rate-limiting"}
    assert alerts.alert_for(status, STARTED, NOW) == {"status": "error", "error": "Yahoo Finance is currently rate-limiting"}


def test_finished_at_or_after_compares_timestamps_not_strings():
    base = "2026-09-17T15:30:00+00:00"
    assert alerts._finished_at_or_after({"status": "done", "started_at": "2026-09-17T15:30:00.000005+00:00"}, base)
    assert alerts._finished_at_or_after({"status": "done", "started_at": "2026-09-17T17:30:00+02:00"}, base)
    assert not alerts._finished_at_or_after({"status": "done", "started_at": "2026-09-17T15:29:59+00:00"}, base)
    assert not alerts._finished_at_or_after({"status": "running", "started_at": base}, base)
    assert not alerts._finished_at_or_after({"status": "idle"}, base)

