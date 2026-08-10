import app.services.fundamentals as fundamentals_module
from app.services.fundamentals import fetch_fundamentals, format_market_cap, market_data_block, resolve_ticker


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
}


def test_market_data_block_renders_full_shape():
    block = market_data_block(FULL_FUNDAMENTALS)
    assert block == (
        "Price: $189.30 | Market Cap: $2.95T\n"
        "P/E (trailing): 31.2 | P/E (forward): 27.8\n"
        "EPS (trailing): $6.07 | Dividend Yield: 0.55%\n"
        "52-Week Range: $164.08 - $237.23"
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
