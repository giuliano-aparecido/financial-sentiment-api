from app.services.ticker import extract_ticker


def test_extracts_ticker_from_natural_language_query():
    # The regression case: text.upper() used to run before matching, so
    # every 3-5 letter word qualified and "WILL" was extracted instead
    # of "AAPL" - this is the UI's own placeholder query.
    assert extract_ticker("Will AAPL go up or down based on recent news and quarterly results?") == "AAPL"


def test_extracts_ticker_mid_sentence():
    assert extract_ticker("What about TSLA earnings this quarter") == "TSLA"


def test_extracts_first_of_multiple_tickers():
    assert extract_ticker("GOOGL vs MSFT which is better") == "GOOGL"


def test_falls_back_to_default_when_no_ticker_present():
    assert extract_ticker("is the market bullish today") == "AAPL"


def test_ignores_common_stopwords():
    assert extract_ticker("WILL the CEO announce an IPO for NVDA") == "NVDA"


def test_falls_back_when_only_stopwords_present():
    assert extract_ticker("WILL the CEO announce an IPO") == "AAPL"
