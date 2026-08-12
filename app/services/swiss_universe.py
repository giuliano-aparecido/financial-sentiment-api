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

import datetime
import json
import logging
import os
import time
import yfinance as yf
from yfinance.data import YfData

logger = logging.getLogger(__name__)

# Escape hatch, off by default: seeds yfinance's process-wide crumb/cookie
# jar (YfData is a true singleton - yfinance.data.YfData, one per process,
# metaclass=SingletonMeta) from externally-obtained values instead of
# letting yfinance fetch its own. Confirmed live: this server's outbound IP
# is currently blocked specifically at Yahoo's crumb-fetch endpoint
# (yfinance.data.YfData._get_crumb_csrf -> query2.finance.yahoo.com/v1/
# test/getcrumb), which raises YFRateLimitError - and since a fresh process
# has no cached crumb (never persisted to disk, and yfinance never re-
# fetches once it HAS one - no expiry check in _get_crumb_csrf), every
# scan on a freshly-restarted process re-attempts that same blocked fetch
# and fails immediately, even though the actual data endpoints
# (screener/download/quoteSummary) were never even reached to know if
# THEY'D have worked. This is why the scan worked for a long time, then
# broke right after a run of unrelated redeploys: an old process had
# already cached a working crumb from before the block started and never
# needed to ask again; each restart throws that away.
#
# Seeding a crumb+cookie pair captured from a DIFFERENT, unblocked network
# lets this process skip the blocked fetch entirely (verified: a captured
# crumb+cookies pair authenticates real Yahoo calls with no re-fetch - see
# the PR that added this). NOT guaranteed to work here: Yahoo may bind a
# crumb/cookie pair to the IP that requested it, in which case using it
# from a different IP fails too - that's genuinely unknown without trying
# it live. If the exact same YFRateLimitError keeps happening after
# setting these, that's the answer: IP-bound, and this doesn't help. A
# different error afterward means it worked and something else is going
# on. Remove YF_SEED_CRUMB/YF_SEED_COOKIES once Yahoo's block on this IP's
# own crumb-fetch lifts and a real fetch starts working again - this is a
# stop-gap, not a permanent fix (the seeded crumb/cookies will themselves
# eventually expire on Yahoo's side, at an unknown time).
def _seed_yf_session_from_env():
    seed_crumb = os.environ.get("YF_SEED_CRUMB")
    seed_cookies_json = os.environ.get("YF_SEED_COOKIES")
    if not seed_crumb or not seed_cookies_json:
        return
    data = YfData()
    if data._crumb:
        return  # already seeded or already fetched its own this process - don't clobber either
    for name, value in json.loads(seed_cookies_json).items():
        data._session.cookies.set(name, value)
    data._crumb = seed_crumb
    logger.info("Seeded yfinance crumb/cookies from YF_SEED_CRUMB/YF_SEED_COOKIES (Yahoo crumb-fetch workaround)")


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
    _seed_yf_session_from_env()

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


def _ex_dividend_date(info: dict) -> str | None:
    # yfinance's exDividendDate is a Unix timestamp (seconds), not a date
    # string - confirmed live. None for companies with no dividend history
    # (the key is simply absent from .info), which utcfromtimestamp(None)
    # would raise on, so this checks first rather than catching.
    ts = info.get("exDividendDate")
    if not ts:
        return None
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date().isoformat()


def filter_domestic(candidates, delay_seconds=INFO_REQUEST_DELAY_SECONDS):
    """Keeps only candidates whose own `country` field is Switzerland -
    the one field that actually reflects company domicile rather than
    exchange/listing region (see module docstring). Fetched per-ticker via
    Ticker.info since the screener response doesn't include this field.
    Returns (symbol -> dict) with the original screener `quote` retained
    (for live/"today" fields) alongside domicile-confirmed extras that
    only .info has - sector/trailing_eps (used by the valuation-adjacent
    scripts) plus a handful of extra current-snapshot fields (dividend
    yield, ex-dividend date, trailing/forward P/E, beta, 52-week range)
    pulled from this SAME .info call at no extra request cost, for
    scripts that want a fuller company profile (see
    swiss_small_cap_crash_rebound.py's run_scan). Fails soft per ticker: a
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
                    "trailing_pe": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "ex_dividend_date": _ex_dividend_date(info),
                    "beta": info.get("beta"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "quote": quote,
                }
            else:
                print(f"  skip {symbol}: domiciled in {country!r}, not Switzerland")
        except Exception as e:
            print(f"  skip {symbol}: info fetch failed ({e!r})")
        time.sleep(delay_seconds)
    return domestic
