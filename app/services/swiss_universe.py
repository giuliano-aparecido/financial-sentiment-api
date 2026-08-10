"""
Shared "find Swiss small caps, excluding foreign companies" logic, used by
both swiss_small_cap_crash_rebound.py and swiss_small_cap_today_screener.py.
Free tools only: yfinance (no API key).

Two filters matter for "small caps in Switzerland, excluding foreign
companies":
  1. Market-cap range (MIN/MAX_MARKET_CAP_CHF below) - Yahoo's region='ch'
     screener field means "listed in the CH region", not "domiciled in
     Switzerland", and it includes ETFs/bonds/structured products
     alongside real equities.
  2. `country` == "Switzerland" from each candidate's own info - SIX lists
     several foreign-domiciled companies (confirmed live: Bitcoin Group SE
     is domiciled in Germany, ams-OSRAM AG in Austria, both pass the
     region+market-cap+quoteType==EQUITY filters otherwise). Only this
     per-company `country` field actually distinguishes domestic from
     foreign - the region/exchange filters alone do not.
"""

import time
import yfinance as yf

# Small-cap band in CHF. SIX's own tiers: SMI (~20 largest) and SMIM (next
# ~30) together cover roughly down to CHF ~1-1.5B; below that is broadly
# "small cap" on this exchange. MIN_MARKET_CAP_CHF excludes illiquid micro
# caps unlikely to have reliable daily pricing. Both are just a starting
# point - adjust freely for a stricter/looser definition.
MIN_MARKET_CAP_CHF = 50_000_000
MAX_MARKET_CAP_CHF = 2_000_000_000

# Politeness delay between per-ticker yfinance .info calls - this is an
# unofficial/undocumented API, not a documented rate limit to size against
# (unlike the Gemini API elsewhere in this project's sibling repo), so this
# is a conservative default, not a measured cap.
INFO_REQUEST_DELAY_SECONDS = 0.3

# Passes the market-cap band and Switzerland-domicile checks but isn't a
# normal operating company, so it doesn't belong in a "small cap" universe
# regardless: SNBN.SW is the Swiss National Bank (confirmed live: shows up
# in the market-cap band, mostly canton-held). Add more symbols here as
# other non-operating-company edge cases turn up.
EXCLUDED_TICKERS = {"SNBN.SW"}


def discover_candidates(min_market_cap=MIN_MARKET_CAP_CHF, max_market_cap=MAX_MARKET_CAP_CHF):
    """Screens Yahoo's CH-region universe for equities in the market-cap
    band. Returns (symbol -> quote dict) for quoteType == 'EQUITY' only,
    excluding EXCLUDED_TICKERS - the region+market-cap query alone still
    returns ETFs/structured products/bonds mixed in (confirmed live), so
    quoteType is filtered here rather than trusted from the query. Each
    quote dict is the FULL raw screener response for that symbol (price,
    live change%, volume, 3-month average volume, etc.) - callers needing
    "today" data (see swiss_small_cap_today_screener.py) can read it
    straight off this dict, no extra fetch needed.
    """
    query = yf.EquityQuery(
        "and",
        [
            yf.EquityQuery("eq", ["region", "ch"]),
            yf.EquityQuery("btwn", ["intradaymarketcap", min_market_cap, max_market_cap]),
        ],
    )

    candidates = {}
    offset = 0
    page_size = 250  # Yahoo's documented max per page
    while True:
        page = yf.screen(query, offset=offset, size=page_size, sortField="ticker", sortAsc=True)
        quotes = page.get("quotes", [])
        if not quotes:
            break
        for q in quotes:
            symbol = q.get("symbol")
            if q.get("quoteType") == "EQUITY" and symbol and symbol not in EXCLUDED_TICKERS:
                candidates[symbol] = q
        offset += page_size
        if offset >= page.get("total", 0):
            break
    return candidates


def filter_domestic(candidates, delay_seconds=INFO_REQUEST_DELAY_SECONDS):
    """Keeps only candidates whose own `country` field is Switzerland -
    the one field that actually reflects company domicile rather than
    exchange/listing region (see module docstring). Fetched per-ticker via
    Ticker.info since the screener response doesn't include this field.
    Returns (symbol -> dict) with the original screener `quote` retained
    (for live/"today" fields) alongside domicile-confirmed extras
    (sector, trailing_eps) that only .info has. Fails soft per ticker: a
    fetch error just excludes that ticker with a warning, rather than
    aborting the whole scan.
    """
    domestic = {}
    for symbol, quote in candidates.items():
        try:
            info = yf.Ticker(symbol).info
            country = info.get("country")
            if country == "Switzerland":
                domestic[symbol] = {
                    "name": quote.get("longName") or quote.get("shortName") or symbol,
                    "sector": info.get("sector"),
                    "market_cap": quote.get("marketCap"),
                    "trailing_eps": info.get("trailingEps"),
                    "quote": quote,
                }
            else:
                print(f"  skip {symbol}: domiciled in {country!r}, not Switzerland")
        except Exception as e:
            print(f"  skip {symbol}: info fetch failed ({e!r})")
        time.sleep(delay_seconds)
    return domestic
