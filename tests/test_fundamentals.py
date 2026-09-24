import datetime

import pandas as pd

import app.services.fundamentals as fundamentals_module
from app.services.fundamentals import (
    fetch_fundamentals,
    format_market_cap,
    market_data_block,
    price_move_on_date,
    resolve_ticker,
    value_screen_metrics,
)


def test_format_market_cap_uses_trillions_above_1e12():
    assert format_market_cap(2.95e12) == "$2.95T"


def test_format_market_cap_uses_billions_below_1e12():
    assert format_market_cap(13.8e9) == "$13.8B"


FULL_FUNDAMENTALS = {
    "price": 189.30,
    "market_cap": 2.95e12,
    "pe_trailing": 31.2,
    "pe_forward": 27.8,
    "eps_trailing": 6.07,
    "dividend_yield": 0.55,
    "year_low": 164.08,
    "year_high": 237.23,
    "book_value_per_share": 4.25,
    "free_cash_flow": 100e9,
    "total_revenue": 390e9,
    "sector": "Technology",
    "operating_margin": 0.30,
    "growth_0y": 0.08,
}


def test_market_data_block_renders_full_shape():
    block = market_data_block(FULL_FUNDAMENTALS)
    assert block == (
        "Price: $189.30 | Market Cap: $2.95T\n"
        "P/E (trailing): 31.2 | P/E (forward): 27.8\n"
        "EPS (trailing): $6.07 | Dividend Yield: 0.55%\n"
        "52-Week Range: $164.08 - $237.23\n"
        "Operating Margin: 30.0% | ROE: 142.8% | Price/Book: 44.5\n"
        "Price/Sales: 7.6 | FCF Yield: 3.4% | PEG: 3.9\n"
        "Sector Median P/E: 28.0 (Technology)"
    )


def test_market_data_block_falls_back_to_na_for_missing_optional_fields():
    partial = {**FULL_FUNDAMENTALS, "pe_forward": None, "year_low": None}
    block = market_data_block(partial)
    assert "P/E (forward): N/A" in block
    assert "52-Week Range: N/A" in block


def test_market_data_block_defaults_dividend_yield_to_zero_when_missing():
    partial = {**FULL_FUNDAMENTALS, "dividend_yield": None}
    assert "Dividend Yield: 0.00%" in market_data_block(partial)


def test_market_data_block_data_unavailable_when_none():
    assert market_data_block(None) == "Data unavailable."


def test_market_data_block_data_unavailable_when_market_cap_missing():
    assert market_data_block({"price": 189.30, "market_cap": None}) == "Data unavailable."


# --- value_screen_metrics ---


def test_value_screen_metrics_computes_all_fields():
    metrics = value_screen_metrics(FULL_FUNDAMENTALS)
    assert round(metrics["roe"], 4) == round(6.07 / 4.25, 4)
    assert metrics["operating_margin"] == 0.30
    assert round(metrics["price_to_sales"], 4) == round(2.95e12 / 390e9, 4)
    assert round(metrics["fcf_yield"], 4) == round(100e9 / 2.95e12, 4)
    assert round(metrics["peg_ratio"], 4) == round(31.2 / 8, 4)
    assert round(metrics["price_to_book"], 4) == round(189.30 / 4.25, 4)
    assert metrics["is_reit_sector"] is False
    assert metrics["sector_median_pe"] == 28.0


def test_value_screen_metrics_roe_none_when_book_value_missing():
    fnd = {**FULL_FUNDAMENTALS, "book_value_per_share": None}
    assert value_screen_metrics(fnd)["roe"] is None
    assert value_screen_metrics(fnd)["price_to_book"] is None


def test_value_screen_metrics_price_to_sales_none_when_revenue_missing():
    fnd = {**FULL_FUNDAMENTALS, "total_revenue": None}
    assert value_screen_metrics(fnd)["price_to_sales"] is None


def test_value_screen_metrics_fcf_yield_none_when_fcf_missing():
    fnd = {**FULL_FUNDAMENTALS, "free_cash_flow": None}
    assert value_screen_metrics(fnd)["fcf_yield"] is None


def test_value_screen_metrics_peg_none_when_growth_not_positive():
    # A negative/zero growth_0y would produce a negative or undefined PEG
    # that misleadingly reads as "cheap" under a naive "lower is better"
    # rule while actually describing a shrinking business - see
    # value_screen_metrics' own comment.
    fnd = {**FULL_FUNDAMENTALS, "growth_0y": -0.05}
    assert value_screen_metrics(fnd)["peg_ratio"] is None
    fnd_zero = {**FULL_FUNDAMENTALS, "growth_0y": 0.0}
    assert value_screen_metrics(fnd_zero)["peg_ratio"] is None


def test_value_screen_metrics_is_reit_sector_true_for_real_estate():
    fnd = {**FULL_FUNDAMENTALS, "sector": "Real Estate"}
    assert value_screen_metrics(fnd)["is_reit_sector"] is True


def test_value_screen_metrics_sector_median_pe_none_for_real_estate():
    # Deliberately omitted from SECTOR_MEDIAN_PE - REITs are valued via
    # dividends/P/B, not a P/E comparison (see the table's own comment).
    fnd = {**FULL_FUNDAMENTALS, "sector": "Real Estate"}
    assert value_screen_metrics(fnd)["sector_median_pe"] is None


def test_value_screen_metrics_sector_median_pe_none_for_unknown_sector():
    fnd = {**FULL_FUNDAMENTALS, "sector": "Some New GICS Category"}
    assert value_screen_metrics(fnd)["sector_median_pe"] is None


# --- resolve_ticker / fetch_fundamentals's resolution fallback ---
# Confirmed live: app.services.ticker.extract_ticker has no exchange-suffix
# awareness, so a query about e.g. Nestle extracts the bare "NESN", but
# Yahoo requires "NESN.SW" for non-US listings - yf.Ticker("NESN").info
# 404s while yf.Search("NESN").quotes ranks "NESN.SW" as the top EQUITY
# match. These tests fake both yfinance entry points rather than hitting
# the network, matching this project's existing monkeypatch style.


class _FakeTicker:
    def __init__(self, info):
        self.info = info


class _FakeSearch:
    def __init__(self, quotes):
        self.quotes = quotes


def test_resolve_ticker_returns_top_equity_match(monkeypatch):
    quotes = [
        {"symbol": "NESNX", "quoteType": "MUTUALFUND"},
        {"symbol": "NESN.SW", "quoteType": "EQUITY"},
        {"symbol": "NESNN.MX", "quoteType": "EQUITY"},
    ]
    monkeypatch.setattr(fundamentals_module.yf, "Search", lambda q: _FakeSearch(quotes))
    assert resolve_ticker("NESN") == "NESN.SW"


def test_resolve_ticker_falls_back_to_original_when_no_equity_match(monkeypatch):
    quotes = [{"symbol": "NESNX", "quoteType": "MUTUALFUND"}]
    monkeypatch.setattr(fundamentals_module.yf, "Search", lambda q: _FakeSearch(quotes))
    assert resolve_ticker("NESN") == "NESN"


def test_resolve_ticker_falls_back_to_original_on_search_failure(monkeypatch):
    def raise_error(q):
        raise RuntimeError("search unavailable")

    monkeypatch.setattr(fundamentals_module.yf, "Search", raise_error)
    assert resolve_ticker("NESN") == "NESN"


def test_fetch_fundamentals_succeeds_directly_without_resolution(monkeypatch):
    info = {"currentPrice": 189.30, "marketCap": 2.95e12}
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    result = fetch_fundamentals("AAPL")
    assert result["resolved_ticker"] == "AAPL"
    assert result["price"] == 189.30


# --- pence/pound normalization (see _normalize_pence_quote's comment) ---


def test_fetch_fundamentals_converts_a_pence_quote_to_pounds(monkeypatch):
    info = {
        "currentPrice": 4198.0,
        "marketCap": 90355507200,
        "currency": "GBp",
        "financialCurrency": "GBP",
        "fiftyTwoWeekLow": 3677.0,
        "fiftyTwoWeekHigh": 5368.0,
        "trailingEps": 2.91,
        "bookValue": 21.472,
        "dividendRate": 2.45,
    }
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    result = fetch_fundamentals("BATS.L")
    assert result["currency"] == "GBP"
    assert result["price"] == 41.98
    assert result["year_low"] == 36.77
    assert result["year_high"] == 53.68
    # Already-in-pounds fields are untouched.
    assert result["eps_trailing"] == 2.91
    assert result["book_value_per_share"] == 21.472
    assert result["dividend_rate"] == 2.45


def test_fetch_fundamentals_leaves_a_pound_quote_alone(monkeypatch):
    info = {"currentPrice": 189.30, "marketCap": 2.95e12, "currency": "USD"}
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    result = fetch_fundamentals("AAPL")
    assert result["currency"] == "USD"
    assert result["price"] == 189.30


def test_fetch_fundamentals_converts_gbx_the_same_as_gbp_pence(monkeypatch):
    info = {"currentPrice": 4198.0, "marketCap": 90355507200, "currency": "GBX"}
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    result = fetch_fundamentals("BATS.L")
    assert result["currency"] == "GBP"
    assert result["price"] == 41.98


def test_fetch_fundamentals_pence_conversion_is_none_safe_for_a_missing_year_range(monkeypatch):
    info = {"currentPrice": 4198.0, "marketCap": 90355507200, "currency": "GBp"}
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    result = fetch_fundamentals("BATS.L")
    assert result["price"] == 41.98
    assert result["year_low"] is None
    assert result["year_high"] is None


def test_fetch_fundamentals_retries_the_same_symbol_before_resolving(monkeypatch):
    # Regression: a transient/degraded yfinance response (no price, no
    # exception) looks identical to a genuinely wrong symbol at the point
    # _fetch_price_info returns None - confirmed in the wild for an
    # ordinary NYSE equity (see portfolio-manager-backend's yahoo_provider
    # history). A same-symbol retry should succeed without ever calling
    # resolve_ticker's Search.
    # First construction (the bare-symbol price check) gets an empty,
    # no-price info; every construction after that (the same-symbol retry,
    # plus the unrelated growth-consensus/earnings-surprise calls further
    # down in fetch_fundamentals) gets a real quote.
    call_count = {"n": 0}

    def fake_ticker(symbol):
        call_count["n"] += 1
        info = {} if call_count["n"] == 1 else {"currentPrice": 189.30, "marketCap": 2.95e12}
        return _FakeTicker(info)

    # A plain raise here would be silently swallowed by resolve_ticker's
    # own broad `except Exception` (logged as a warning, not propagated) -
    # count calls instead so the assertion below checks the real
    # observable behavior, not an exception that might never surface.
    search_calls = {"n": 0}

    def counting_search(q):
        search_calls["n"] += 1
        return _FakeSearch([])

    monkeypatch.setattr(fundamentals_module.yf, "Ticker", fake_ticker)
    monkeypatch.setattr(fundamentals_module.yf, "Search", counting_search)

    result = fetch_fundamentals("AAPL")
    assert search_calls["n"] == 0  # same-symbol retry succeeded, resolve_ticker's Search never needed
    assert result["resolved_ticker"] == "AAPL"
    assert result["price"] == 189.30
    assert call_count["n"] >= 2  # the price check was retried at least once


def test_fetch_fundamentals_retries_with_resolved_ticker_when_bare_symbol_fails(monkeypatch):
    tickers = {
        "NESN": {},  # no price - 404-equivalent
        "NESN.SW": {"currentPrice": 92.5, "marketCap": 250e9},
    }
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker(tickers[symbol]))
    monkeypatch.setattr(fundamentals_module.yf, "Search", lambda q: _FakeSearch([{"symbol": "NESN.SW", "quoteType": "EQUITY"}]))

    result = fetch_fundamentals("NESN")
    assert result["resolved_ticker"] == "NESN.SW"
    assert result["price"] == 92.5


def test_fetch_fundamentals_none_when_resolution_finds_no_match(monkeypatch):
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker({}))
    monkeypatch.setattr(fundamentals_module.yf, "Search", lambda q: _FakeSearch([]))
    assert fetch_fundamentals("BOGUS") is None


def test_fetch_fundamentals_none_when_resolved_symbol_also_has_no_price(monkeypatch):
    tickers = {"NESN": {}, "NESN.SW": {}}
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeTicker(tickers[symbol]))
    monkeypatch.setattr(fundamentals_module.yf, "Search", lambda q: _FakeSearch([{"symbol": "NESN.SW", "quoteType": "EQUITY"}]))
    assert fetch_fundamentals("NESN") is None


# --- price_move_on_date ---


class _FakeHistoryTicker:
    def __init__(self, hist):
        self._hist = hist

    def history(self, start=None, end=None):
        return self._hist


def _make_hist(rows):
    """rows: list of (date, close) - builds a tz-naive DatetimeIndex'd
    DataFrame the way yfinance's Ticker.history returns one."""
    index = pd.DatetimeIndex([d for d, _ in rows])
    return pd.DataFrame({"Close": [c for _, c in rows]}, index=index)


# Mon 06-15 through Fri 06-19, then Mon 06-22 - a real trading calendar
# (weekend gap between 06-19 and 06-22) used by several tests below.
_WEEK_HIST = _make_hist([
    (datetime.date(2026, 6, 15), 100.0),
    (datetime.date(2026, 6, 16), 102.0),
    (datetime.date(2026, 6, 17), 108.0),
    (datetime.date(2026, 6, 18), 107.0),
    (datetime.date(2026, 6, 19), 110.0),
    (datetime.date(2026, 6, 22), 112.0),
])


def test_price_move_on_date_computes_single_day_move_not_trailing_window(monkeypatch):
    # published on 06-17 (close 108) vs the PRECEDING day 06-16 (close
    # 102) - not a multi-day trailing/cumulative figure.
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeHistoryTicker(_WEEK_HIST))
    text, move_fraction = price_move_on_date("AAPL", datetime.date(2026, 6, 17))
    assert move_fraction == (108.0 - 102.0) / 102.0
    assert text == "AAPL moved +5.9% on the day this was published."


def test_price_move_on_date_rolls_forward_across_weekend(monkeypatch):
    # published Saturday 06-20 (not a trading day) - day-0 should roll
    # forward to the next trading day, Monday 06-22 (close 112), vs the
    # preceding trading day 06-19 (close 110).
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeHistoryTicker(_WEEK_HIST))
    text, move_fraction = price_move_on_date("AAPL", datetime.date(2026, 6, 20))
    assert move_fraction == (112.0 - 110.0) / 110.0
    assert "AAPL moved" in text


def test_price_move_on_date_none_when_published_date_is_none(monkeypatch):
    called = []
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: called.append(symbol))
    text, move_fraction = price_move_on_date("AAPL", None)
    assert text == "Data unavailable."
    assert move_fraction is None
    assert called == []  # no yfinance call at all - nothing to look up


def test_price_move_on_date_none_when_no_prior_trading_day_in_window(monkeypatch):
    # published_date lands on the FIRST row of the fetched history - no
    # preceding close available inside the window.
    hist = _make_hist([(datetime.date(2026, 6, 17), 108.0), (datetime.date(2026, 6, 18), 107.0)])
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeHistoryTicker(hist))
    text, move_fraction = price_move_on_date("AAPL", datetime.date(2026, 6, 17))
    assert text == "Data unavailable."
    assert move_fraction is None


def test_price_move_on_date_none_when_history_empty(monkeypatch):
    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _FakeHistoryTicker(pd.DataFrame({"Close": []})))
    text, move_fraction = price_move_on_date("AAPL", datetime.date(2026, 6, 17))
    assert text == "Data unavailable."
    assert move_fraction is None


def test_price_move_on_date_none_on_fetch_failure(monkeypatch):
    class _RaisingTicker:
        def history(self, start=None, end=None):
            raise RuntimeError("yfinance unavailable")

    monkeypatch.setattr(fundamentals_module.yf, "Ticker", lambda symbol: _RaisingTicker())
    text, move_fraction = price_move_on_date("AAPL", datetime.date(2026, 6, 17))
    assert text == "Data unavailable."
    assert move_fraction is None
