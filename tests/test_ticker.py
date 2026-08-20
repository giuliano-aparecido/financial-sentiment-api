from app.services.ticker import extract_ticker


def test_extracts_ticker_from_natural_language_query():
    # The regression case: text.upper() used to run before matching, so
    # every 3-5 letter word qualified and "WILL" was extracted instead
    # of "AAPL" - this is the UI's own placeholder query.
    assert extract_ticker("Will AAPL go up or down based on recent news and quarterly results?") == ("AAPL", True)


def test_extracts_ticker_mid_sentence():
    assert extract_ticker("What about TSLA earnings this quarter") == ("TSLA", True)


def test_extracts_first_of_multiple_tickers():
    assert extract_ticker("GOOGL vs MSFT which is better") == ("GOOGL", True)


def test_falls_back_to_default_when_no_ticker_present():
    assert extract_ticker("is the market bullish today") == ("AAPL", False)


def test_ignores_common_stopwords():
    assert extract_ticker("WILL the CEO announce an IPO for NVDA") == ("NVDA", True)


def test_falls_back_when_only_stopwords_present():
    assert extract_ticker("WILL the CEO announce an IPO") == ("AAPL", False)


def test_extracts_cashtag_ticker():
    assert extract_ticker("What's the outlook for $TSLA this quarter?") == ("TSLA", True)


def test_cashtag_is_uppercased():
    assert extract_ticker("thoughts on $aapl earnings?") == ("AAPL", True)


def test_cashtag_takes_priority_over_plain_caps_word():
    # WILL would otherwise be filtered by the stopword list and NVDA found
    # by the fallback heuristic anyway - this asserts the cashtag short-
    # circuits straight to NVDA without even running that fallback.
    assert extract_ticker("Will $NVDA beat GOOGL this quarter") == ("NVDA", True)


def test_falls_back_when_query_names_a_ticker_that_is_purely_lowercase():
    # No cashtag and no all-caps candidate - "aapl" in lowercase doesn't
    # match the fallback heuristic's \b[A-Z]{2,5}\b pattern, so this is
    # correctly a fallback, not a false positive.
    assert extract_ticker("what about aapl") == ("AAPL", False)


def test_cashtag_preserves_explicit_exchange_suffix():
    # Regression: an explicit "$ARYN.SW" used to have ".SW" silently
    # stripped, forcing a dependency on fundamentals.resolve_ticker's
    # live Yahoo search call to add it back - which can fail
    # independently of whether the user already gave the right answer.
    assert extract_ticker("Is $ARYN.SW a good investment now for long term?") == ("ARYN.SW", True)


def test_cashtag_suffix_is_uppercased_with_the_rest():
    assert extract_ticker("thoughts on $nesn.sw") == ("NESN.SW", True)


def test_cashtag_without_suffix_still_works_unchanged():
    assert extract_ticker("what about $NESN") == ("NESN", True)


def test_cashtag_does_not_swallow_trailing_sentence_after_period():
    # A period followed by a SPACE (a real sentence boundary) must not be
    # mistaken for the start of an exchange suffix.
    assert extract_ticker("$AAPL. Great stock overall.") == ("AAPL", True)


def test_cashtag_does_not_swallow_overlong_trailing_text():
    # "something-longer-here" is far more than a 1-3 letter exchange
    # suffix - must fall back to just the base ticker, not a partial
    # garbage match.
    assert extract_ticker("$TSLA.something-longer-here") == ("TSLA", True)
