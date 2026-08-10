import pandas as pd

import app.services.earnings as earnings_module
from app.services.earnings import earnings_block, fetch_earnings

FULL_EARNINGS = {
    "last_quarter_date": "2026-06-30",
    "revenue": 85.8e9,
    "yoy_growth_pct": 4.9,
    "eps_actual": 1.40,
    "eps_estimate": 1.35,
    "next_earnings_date": "2026-10-29",
}


def test_earnings_block_renders_full_shape_with_beat():
    block = earnings_block(FULL_EARNINGS)
    assert block == (
        "Last Quarter (2026-06-30): Revenue $85.8B (+4.9% YoY), EPS $1.40 (beat est. $1.35)\n"
        "Next Earnings Date: 2026-10-29"
    )


def test_earnings_block_reports_miss():
    earnings = {**FULL_EARNINGS, "eps_actual": 1.20}
    assert "missed est. $1.35" in earnings_block(earnings)


def test_earnings_block_reports_in_line():
    earnings = {**FULL_EARNINGS, "eps_actual": 1.35}
    assert "in line with est. $1.35" in earnings_block(earnings)


def test_earnings_block_omits_yoy_when_unavailable():
    earnings = {**FULL_EARNINGS, "yoy_growth_pct": None}
    block = earnings_block(earnings)
    assert "YoY" not in block
    assert "Revenue $85.8B, EPS" in block


def test_earnings_block_omits_eps_note_when_estimate_missing():
    earnings = {**FULL_EARNINGS, "eps_actual": None, "eps_estimate": None}
    block = earnings_block(earnings)
    assert block.startswith("Last Quarter (2026-06-30): Revenue $85.8B (+4.9% YoY)\n")
    assert "EPS" not in block


def test_earnings_block_omits_next_date_when_unavailable():
    earnings = {**FULL_EARNINGS, "next_earnings_date": None}
    assert "Next Earnings Date" not in earnings_block(earnings)


def test_earnings_block_data_unavailable_when_none():
    assert earnings_block(None) == "Data unavailable."


def test_earnings_block_data_unavailable_when_revenue_missing():
    assert earnings_block({"revenue": None}) == "Data unavailable."


def test_earnings_block_negative_yoy_has_no_plus_sign():
    earnings = {**FULL_EARNINGS, "yoy_growth_pct": -5.6}
    assert "(-5.6% YoY)" in earnings_block(earnings)


# --- fetch_earnings's resolution fallback ---
# Same root cause as fundamentals.fetch_fundamentals (see that module's
# test file): a bare ticker like "NESN" doesn't resolve on yfinance, so
# this retries once via resolve_ticker. resolve_ticker's OWN behavior
# (search ranking, failure handling) is already covered in
# test_fundamentals.py - these tests fake it directly rather than
# re-testing its internals, to keep the retry logic under test isolated.


class _FakeTicker:
    def __init__(self, income, earnings_dates=None):
        self.quarterly_income_stmt = income
        self.earnings_dates = earnings_dates


def _income_stmt(report_date, revenue):
    return pd.DataFrame({pd.Timestamp(report_date): {"Total Revenue": revenue}})


def test_fetch_earnings_succeeds_directly_without_resolution(monkeypatch):
    income = _income_stmt("2026-06-30", 85.8e9)
    monkeypatch.setattr(earnings_module.yf, "Ticker", lambda symbol: _FakeTicker(income))
    result = fetch_earnings("AAPL")
    assert result["revenue"] == 85.8e9


def test_fetch_earnings_retries_with_resolved_ticker_when_bare_symbol_fails(monkeypatch):
    tickers = {
        "NESN": _FakeTicker(pd.DataFrame()),  # no data - matches yfinance's behavior for an unresolved symbol
        "NESN.SW": _FakeTicker(_income_stmt("2026-06-30", 92.1e9)),
    }
    monkeypatch.setattr(earnings_module.yf, "Ticker", lambda symbol: tickers[symbol])
    monkeypatch.setattr(earnings_module, "resolve_ticker", lambda t: "NESN.SW")

    result = fetch_earnings("NESN")
    assert result["revenue"] == 92.1e9


def test_fetch_earnings_none_when_resolution_finds_no_match(monkeypatch):
    monkeypatch.setattr(earnings_module.yf, "Ticker", lambda symbol: _FakeTicker(pd.DataFrame()))
    monkeypatch.setattr(earnings_module, "resolve_ticker", lambda t: t)  # unresolved - same ticker back
    assert fetch_earnings("BOGUS") is None


def test_fetch_earnings_none_when_resolved_symbol_also_has_no_data(monkeypatch):
    monkeypatch.setattr(earnings_module.yf, "Ticker", lambda symbol: _FakeTicker(pd.DataFrame()))
    monkeypatch.setattr(earnings_module, "resolve_ticker", lambda t: "NESN.SW")
    assert fetch_earnings("NESN") is None
