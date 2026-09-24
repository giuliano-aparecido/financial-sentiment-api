import datetime
import logging
import math

import pandas as pd
import yfinance as yf

from app.services.valuation import REIT_SECTORS, SECTOR_MEDIAN_PE

logger = logging.getLogger(__name__)


def _usable(value) -> bool:
    """None and NaN both mean "not usable" - NaN passes yfinance's own
    truthiness check (`not float('nan')` is False), so a plain `if not
    value` guard silently lets NaN through. Confirmed live: a NaN
    yearAgoEps/low/high/growth from earnings_estimate flowed all the way
    to a rendered "Intrinsic Value: $nan" / "undervalued by ~nan%" in the
    model prompt (min(nan, X) and max(nan, X) both return nan, so
    valuation.py's own g1 clamps didn't catch it either)."""
    if value is None:
        return False
    try:
        return not math.isnan(value)
    except TypeError:
        return True


# Yahoo quotes LSE-listed securities (currency "GBp" or "GBX") in pence,
# while every other numeric `.info` field - market cap, EPS, book value,
# dividend rate, revenue, FCF - is already in pounds. Uncorrected, any
# per-share math derived from market_cap/price (see valuation.py's
# _shares_outstanding_approx) is off by ~100x. Other exchanges with a
# similar minor-subunit convention (Johannesburg's ZAc, Tel Aviv's ILA)
# aren't covered yet - not added speculatively, only once confirmed live.
_PENCE_CURRENCIES = frozenset({"GBp", "GBX"})


def _normalize_pence_quote(
    *, currency: str | None, price: float | None, year_low: float | None, year_high: float | None
) -> tuple[str | None, float | None, float | None, float | None]:
    """(currency, price, year_low, year_high) with a pence quote converted
    to pounds and relabeled "GBP" - unchanged for anything else. Case-
    sensitive on purpose: "GBp" (pence) and "GBP" (pounds) differ only in
    the case of that last letter, so folding case here would erase the
    only signal there is.
    """
    if currency not in _PENCE_CURRENCIES:
        return currency, price, year_low, year_high

    def _to_pounds(v: float | None) -> float | None:
        return v / 100 if v is not None else None

    return "GBP", _to_pounds(price), _to_pounds(year_low), _to_pounds(year_high)


def resolve_ticker(ticker: str) -> str:
    """Resolves a bare ticker to the symbol yfinance/Yahoo actually
    recognizes, e.g. "NESN" -> "NESN.SW". app.services.ticker.extract_ticker
    has no exchange-suffix awareness (it just pulls a 2-5 letter token out
    of free text), which works fine for US listings (no suffix needed on
    Yahoo) but 404s for most non-US ones - confirmed live:
    yf.Ticker("NESN").info returns nothing, while yf.Search("NESN").quotes
    ranks "NESN.SW" (Nestle's real Swiss listing) as the top EQUITY match.
    This is exactly why market_data/valuation render "Data unavailable."
    for a company like Nestle while the news block (a free-text Google News
    search - see news.py - which doesn't need a valid symbol at all) still
    works.

    Only meant to be called as a FALLBACK after a direct fetch with the
    original ticker has already failed (see fetch_fundamentals below) -
    calling it unconditionally on every request would add a network round
    trip to the already-working common case for no benefit. Returns the
    ORIGINAL ticker unchanged (not None) on any resolution failure or if no
    EQUITY match is found, so callers can use the result unconditionally -
    a resolution failure just means the downstream fetch fails the same way
    it would have without this fallback.
    """
    try:
        matches = yf.Search(ticker).quotes
    except Exception as e:
        logger.warning("yfinance ticker search failed for %r: %s", ticker, e)
        return ticker
    for match in matches:
        if match.get("quoteType") == "EQUITY" and match.get("symbol"):
            return match["symbol"]
    return ticker


def _fetch_growth_consensus(ticker: str) -> dict:
    """Best-effort near-term consensus growth from yfinance's
    earnings_estimate table (0y/+1y analyst EPS estimates), used by
    valuation.py to derive a per-company g1 (see that module's
    build_scenarios). All-None on any failure or missing data - this is
    deliberately independent of fetch_fundamentals' own try/except, since a
    growth-estimate outage shouldn't fail the whole fundamentals fetch (the
    valuation model already has a generic-growth fallback for exactly this
    case).

    growth_0y/growth_1y are the current-year/next-year consensus EPS growth
    rates (yfinance's own precomputed "growth" column). growth_0y_low/_high
    are derived from the SAME 0y row's low/high analyst estimates vs its
    yearAgoEps - confirmed live against 5 analyst-provided examples to
    approximate a best/worst-case growth spread for companies whose
    consensus isn't already distorted by a rebound/giveback (see
    valuation.py's reliability check for how that distortion is detected).
    """
    empty = {"growth_0y": None, "growth_1y": None, "growth_0y_low": None, "growth_0y_high": None}
    try:
        estimate = yf.Ticker(ticker).earnings_estimate
        row_0y = estimate.loc["0y"]
        row_1y = estimate.loc["+1y"]
        year_ago = row_0y["yearAgoEps"]
        growth_0y = row_0y["growth"]
        growth_1y = row_1y["growth"]
        low = row_0y["low"]
        high = row_0y["high"]
        # year_ago==0 would also divide-by-zero below - `not year_ago`
        # covers None/0/NaN-is-truthy-so-NOT-caught-here, hence the
        # explicit _usable() check alongside it (see that function's
        # comment).
        if not _usable(year_ago) or year_ago == 0 or not _usable(growth_0y) or not _usable(growth_1y):
            return empty
        result = {"growth_0y": growth_0y, "growth_1y": growth_1y, "growth_0y_low": None, "growth_0y_high": None}
        # low/high degrade independently rather than failing the whole
        # result - same fail-soft convention as the rest of this module;
        # build_scenarios' consensus path already handles either being
        # None (see its own comment on high_offset/low_offset mirroring).
        if _usable(low):
            result["growth_0y_low"] = (low - year_ago) / abs(year_ago)
        if _usable(high):
            result["growth_0y_high"] = (high - year_ago) / abs(year_ago)
        return result
    except Exception as e:
        logger.warning("yfinance growth-estimate fetch failed for %s: %s", ticker, e)
        return empty


def _fetch_recent_eps_surprise(ticker: str) -> float | None:
    """Actual-vs-consensus EPS surprise (as a fraction, e.g. 0.94 for a
    94% beat) for the most recently REPORTED quarter, from yfinance's
    earnings_dates table. Used by valuation.py to detect a likely one-
    time/non-operating item in trailing EPS - confirmed live: GOOG's
    trailing EPS was inflated by two consecutive quarters beating
    consensus by +94% and +213% (almost certainly mark-to-market gains
    on its equity investment stakes, a known recurring GAAP-distortion
    pattern for it specifically), not organic operating growth. This is
    the mirror image of the trailing-vs-forward P/E screens in
    valuation.py (which catch a distorted EPS via an anomalously LOW
    P/E) - GOOG's P/E looked completely normal precisely BECAUSE the
    inflated EPS denominator masked it, so neither of those screens
    fired. Comparing actual-vs-consensus EPS for the most recent quarter
    is a more direct signal than a P/E ratio for this specific failure
    mode. None (not necessarily a fetch failure) when no reported row
    with a usable estimate exists yet."""
    try:
        dates = yf.Ticker(ticker).earnings_dates
        reported = dates.dropna(subset=["Reported EPS"])
        if reported.empty:
            return None
        row = reported.iloc[0]
        estimate = row["EPS Estimate"]
        actual = row["Reported EPS"]
        if not _usable(estimate) or not _usable(actual) or estimate == 0:
            return None
        return (actual - estimate) / abs(estimate)
    except Exception as e:
        logger.warning("yfinance earnings-surprise fetch failed for %s: %s", ticker, e)
        return None


def _fetch_price_info(ticker: str) -> dict | None:
    """Ticker.info if it resolves to a real quote with a price, else None -
    factored out so fetch_fundamentals can retry with the same ticker and
    with a resolve_ticker()-corrected symbol without duplicating the
    fetch/price-check logic."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning("yfinance fundamentals fetch failed for %s: %s", ticker, e)
        return None
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    return info if price is not None else None


def fetch_fundamentals(ticker: str) -> dict | None:
    """Fetches current fundamentals for `ticker` via yfinance's .info dict.
    Returns None on any fetch failure or if price itself is missing (the
    one field everything else - market cap, the valuation comparison - is
    anchored to). Individual other fields (P/E, dividend yield, 52-week
    range, EPS, book value) can still be None inside a successful result;
    callers render those as "N/A" rather than failing the whole block.

    A failed first attempt retries the SAME ticker once (a transient/
    degraded yfinance response looks identical to a wrong symbol - see
    the retry's own comment below) before falling through to
    resolve_ticker (e.g. "NESN" -> "NESN.SW") - see that function's
    docstring for why the latter is needed for most non-US listings. The
    resolved symbol (which may equal the original) is used for every
    subsequent call in this function, including _fetch_growth_consensus
    below, and is exposed back as "resolved_ticker" so callers that need
    the SAME symbol used elsewhere (e.g. fetch_earnings, run independently
    and concurrently - see analyze.py - so it can't just reuse this
    result) can be told about the correction; it's not required reading
    for callers that don't care.
    """
    info = _fetch_price_info(ticker)
    if info is None:
        # yfinance can return a real quote's .info with no price on a
        # transient/degraded response, indistinguishable at this point from
        # a genuinely wrong symbol - retry the same ticker once before
        # falling through to resolve_ticker's more expensive Search call,
        # which wouldn't help a flaky response anyway (confirmed in the
        # wild: portfolio-manager-backend's ported copy of this function
        # hit this for UBER, an ordinary NYSE equity - see that repo's
        # yahoo_provider.py history).
        info = _fetch_price_info(ticker)
    resolved_ticker = ticker
    if info is None:
        resolved_ticker = resolve_ticker(ticker)
        if resolved_ticker != ticker:
            info = _fetch_price_info(resolved_ticker)
    if info is None:
        return None

    currency, price, year_low, year_high = _normalize_pence_quote(
        currency=info.get("currency"),
        price=info.get("currentPrice") or info.get("regularMarketPrice"),
        year_low=info.get("fiftyTwoWeekLow"),
        year_high=info.get("fiftyTwoWeekHigh"),
    )
    fundamentals = {
        "resolved_ticker": resolved_ticker,
        "price": price,
        "market_cap": info.get("marketCap"),
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "eps_trailing": info.get("trailingEps"),
        "book_value_per_share": info.get("bookValue"),
        "dividend_yield": info.get("dividendYield"),
        "year_low": year_low,
        "year_high": year_high,
        # Added for the scenario-DCF valuation model (app/services/
        # valuation.py) - all sourced from this SAME .info call, no extra
        # yfinance request. free_cash_flow/total_revenue/dividend_rate are
        # per-company-total or per-share cash-flow-basis candidates;
        # sector/industry/payout_ratio drive which basis gets picked. Each
        # is independently optional - valuation.py's classifier and
        # fail-soft rendering handle any subset being None.
        "free_cash_flow": info.get("freeCashflow"),
        "total_revenue": info.get("totalRevenue"),
        "dividend_rate": info.get("dividendRate") or info.get("trailingAnnualDividendRate"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        # Not used by anything valuation-related below - fetched here from
        # this same .info call purely so news.py's relevance filter
        # (_is_relevant_headline) has a company name to match headlines
        # against, without a second/duplicate yfinance .info fetch.
        "company_name": info.get("shortName") or info.get("longName"),
        "payout_ratio": info.get("payoutRatio"),
        # currency = what the stock TRADES in; financial_currency = what
        # totalRevenue/freeCashflow are REPORTED in - confirmed live these
        # can differ for a company that reports in one currency but is
        # cross-listed on an exchange denominated in another (the same
        # non-US-listing class resolve_ticker exists for). valuation.py's
        # cash_flow_basis_value uses this to avoid dividing a
        # financial-currency total by a trading-currency share count for
        # the fcf/revenue bases - see that function's own comment.
        "currency": currency,
        "financial_currency": info.get("financialCurrency"),
        # Value-screen metric (see value_screen_metrics below) with no
        # existing fetched-or-derivable equivalent elsewhere in this dict -
        # everything else that function needs (ROE, P/S, FCF yield, PEG) is
        # derived from fields already fetched above, deliberately reusing
        # the SAME formulas valuation.py already uses internally (e.g. ROE
        # = eps_trailing / book_value_per_share, matching
        # _sustainable_growth_rate exactly) rather than fetching yfinance's
        # own returnOnEquity/pegRatio fields, which use different
        # methodology and could show a second, disagreeing number for the
        # same concept in the same prompt.
        "operating_margin": info.get("operatingMargins"),
    }
    fundamentals.update(_fetch_growth_consensus(resolved_ticker))
    fundamentals["recent_eps_surprise"] = _fetch_recent_eps_surprise(resolved_ticker)
    return fundamentals


# Calendar days fetched before published_date to guarantee finding a
# prior trading day's close even across a long weekend/holiday run -
# ported verbatim (same name and value) from financial-sentiment-model's
# generate_real_dataset.py PRE_PUBLISH_BUFFER_DAYS.
PRE_PUBLISH_BUFFER_DAYS = 7

# Calendar days fetched AFTER published_date - only day-0 itself is
# needed, but weekends/tz quirks around yfinance's `end` (exclusive) mean
# a 1-day buffer isn't always enough to guarantee day-0 lands inside the
# fetched window.
POST_PUBLISH_BUFFER_DAYS = 3


def _as_of_timestamp(index, as_of_date):
    """pd.Timestamp for as_of_date, localized to match `index`'s own
    tz-awareness - ported verbatim from financial-sentiment-model's
    generate_real_dataset.py (same helper, same tz-mismatch problem:
    yfinance price-history indices are inconsistently tz-aware across
    tickers, and pandas raises TypeError comparing/searchsorting a naive
    Timestamp against a tz-aware DatetimeIndex or vice versa)."""
    ts = pd.Timestamp(as_of_date)
    tz = getattr(index, "tz", None)
    return ts.tz_localize(tz) if tz is not None else ts


def price_move_on_date(ticker: str, published_date: datetime.date | None) -> tuple[str, float | None]:
    """Single-day close-to-close move for `published_date` - the day-0
    close (published_date, or the next trading day if published after
    close/on a weekend) vs. the immediately preceding trading day's close.
    Task A's only price signal (see inference.py's classify_news).
    Replaces the old recent_price_move's "trailing as of now"
    approximation now that news.py (see fetch_live_news_rag) selects and
    returns the ONE headline actually used, and its published_date with
    it - this can finally measure the SAME thing training does
    (financial-sentiment-model's generate_real_dataset.py measure_
    reaction_windows/price_context_block, ported not imported, same
    convention as the DCF valuation math), eliminating the trailing-vs-
    forward mismatch the old function openly documented as a stopgap. See
    that repo's module docstring history item 12 for the SIGN.SW example
    (a same-day >10% crash that mostly reversed within two days) that
    motivated single-day over a multi-day window in the first place.

    Returns (text, move_fraction) - move_fraction is a plain fraction
    (0.056 for +5.6%), matching price_context_block's own convention.
    ("Data unavailable.", None) if published_date is None (news.py
    selected no headline) or on any fetch failure/insufficient history.
    """
    if published_date is None:
        return "Data unavailable.", None

    start_date = published_date - datetime.timedelta(days=PRE_PUBLISH_BUFFER_DAYS)
    end_date = published_date + datetime.timedelta(days=POST_PUBLISH_BUFFER_DAYS)
    try:
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
    except Exception as e:
        logger.warning("yfinance price-history fetch failed for %s: %s", ticker, e)
        return "Data unavailable.", None

    if hist.empty:
        return "Data unavailable.", None

    published_ts = _as_of_timestamp(hist.index, published_date)
    day0_pos = hist.index.searchsorted(published_ts)
    if day0_pos == 0 or day0_pos >= len(hist):
        return "Data unavailable.", None

    prev_close = hist["Close"].iloc[day0_pos - 1]
    day0_close = hist["Close"].iloc[day0_pos]
    if not _usable(prev_close) or not _usable(day0_close) or not prev_close:
        return "Data unavailable.", None

    move_fraction = (day0_close - prev_close) / prev_close
    return f"{ticker} moved {move_fraction * 100:+.1f}% on the day this was published.", move_fraction


# See value_screen_metrics' own comment on peg_ratio for why this exists.
PEG_MIN_GROWTH_FOR_COMPUTATION = 0.02


def value_screen_metrics(fundamentals: dict) -> dict:
    """Derives the value-investing checklist metrics (see
    value-investing-checklist.md) from fields already in `fundamentals` -
    ROE, Price/Sales, FCF yield, and PEG are all computed here rather than
    fetched separately, deliberately reusing the same formulas valuation.py
    already uses internally (see fetch_fundamentals' comment on
    operating_margin for why). Each value is None when its inputs are
    missing/unusable - callers render those as "N/A", same fail-soft
    convention as the rest of this module.
    """
    eps_trailing = fundamentals.get("eps_trailing")
    book_value_per_share = fundamentals.get("book_value_per_share")
    roe = None
    if eps_trailing and book_value_per_share and book_value_per_share > 0:
        roe = eps_trailing / book_value_per_share

    market_cap = fundamentals.get("market_cap")
    total_revenue = fundamentals.get("total_revenue")
    price_to_sales = None
    if market_cap and total_revenue and total_revenue > 0:
        price_to_sales = market_cap / total_revenue

    free_cash_flow = fundamentals.get("free_cash_flow")
    fcf_yield = None
    if free_cash_flow is not None and market_cap and market_cap > 0:
        fcf_yield = free_cash_flow / market_cap

    pe_trailing = fundamentals.get("pe_trailing")
    growth_0y = fundamentals.get("growth_0y")
    # PEG only means anything against POSITIVE expected growth - a negative
    # or zero growth_0y would produce a negative/undefined PEG that reads
    # as "attractively priced" by a naive "lower is better" rule while
    # actually describing a shrinking business, so it's left None (renders
    # "N/A") rather than shown as a number that would mislead. A near-zero
    # (but positive) growth_0y has the same problem the other direction -
    # confirmed live PEG values up to 525 in synthetic data purely from
    # dividing by a growth rate close to 0%, not from any real
    # over/under-valuation signal. PEG_MIN_GROWTH_FOR_COMPUTATION floors
    # how small a growth rate this ratio is computed against at all.
    peg_ratio = None
    if pe_trailing and growth_0y and growth_0y > PEG_MIN_GROWTH_FOR_COMPUTATION:
        peg_ratio = pe_trailing / (growth_0y * 100)

    price = fundamentals.get("price")
    price_to_book = None
    if price and book_value_per_share and book_value_per_share > 0:
        price_to_book = price / book_value_per_share

    sector = fundamentals.get("sector")
    sector_median_pe = SECTOR_MEDIAN_PE.get(sector)

    return {
        "roe": roe,
        "operating_margin": fundamentals.get("operating_margin"),
        "price_to_sales": price_to_sales,
        "fcf_yield": fcf_yield,
        "peg_ratio": peg_ratio,
        "price_to_book": price_to_book,
        "is_reit_sector": sector in REIT_SECTORS,
        "sector_median_pe": sector_median_pe,
    }


def format_market_cap(value: float) -> str:
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    return f"${value / 1e9:.1f}B"


def market_data_block(fundamentals: dict | None) -> str:
    """Renders the 'Current Market Data' prompt block. Byte-format matches
    generate_synthetic_dataset.py/generate_real_dataset.py in
    financial-sentiment-model exactly (see that repo's
    CONTRIBUTING.md 4-way sync rule) - the model is trained on this shape.
    """
    if not fundamentals or fundamentals.get("market_cap") is None:
        return "Data unavailable."

    pe_trailing = fundamentals.get("pe_trailing")
    pe_forward = fundamentals.get("pe_forward")
    eps = fundamentals.get("eps_trailing")
    dividend_yield = fundamentals.get("dividend_yield")
    year_low = fundamentals.get("year_low")
    year_high = fundamentals.get("year_high")

    pe_trailing_str = f"{pe_trailing:.1f}" if pe_trailing else "N/A"
    pe_forward_str = f"{pe_forward:.1f}" if pe_forward else "N/A"
    eps_str = f"${eps:.2f}" if eps is not None else "N/A"
    div_yield_str = f"{dividend_yield:.2f}%" if dividend_yield else "0.00%"
    range_str = f"${year_low:.2f} - ${year_high:.2f}" if (year_low and year_high) else "N/A"

    # Only the Price line's prefix, not EPS/market-cap/range too - a
    # contained fix, not a full currency-formatting rewrite (see
    # valuation.py's cash_flow_basis_value for the more consequential half
    # of the currency problem: FX-mixing in the fcf/revenue bases' actual
    # math, not just display). A reader seeing "Price: CHF 92.50" already
    # gets the signal the whole block is non-USD; USD (the overwhelming
    # common case) renders byte-identical to before.
    currency = fundamentals.get("currency") or "USD"
    price_prefix = "$" if currency == "USD" else f"{currency} "

    screen = value_screen_metrics(fundamentals)
    operating_margin_str = f"{screen['operating_margin'] * 100:.1f}%" if screen["operating_margin"] is not None else "N/A"
    roe_str = f"{screen['roe'] * 100:.1f}%" if screen["roe"] is not None else "N/A"
    price_to_book_str = f"{screen['price_to_book']:.1f}" if screen["price_to_book"] is not None else "N/A"
    price_to_sales_str = f"{screen['price_to_sales']:.1f}" if screen["price_to_sales"] is not None else "N/A"
    fcf_yield_str = f"{screen['fcf_yield'] * 100:.1f}%" if screen["fcf_yield"] is not None else "N/A"
    peg_ratio_str = f"{screen['peg_ratio']:.1f}" if screen["peg_ratio"] is not None else "N/A"
    # Sector name now renders even when no median exists for it (Real
    # Estate, deliberately excluded - see SECTOR_MEDIAN_PE's own comment)
    # rather than a bare "N/A" - the model's prompt has no other reliable
    # signal that a company is a REIT specifically (the "Dividend-based"
    # valuation label alone doesn't say why), and the checklist's REIT
    # Price/Book rule only applies if the sector is actually identifiable.
    sector = fundamentals.get("sector")
    if screen["sector_median_pe"] is not None:
        sector_median_pe_str = f"{screen['sector_median_pe']:.1f} ({sector})"
    elif sector:
        sector_median_pe_str = f"N/A ({sector})"
    else:
        sector_median_pe_str = "N/A"

    return (
        f"Price: {price_prefix}{fundamentals['price']:.2f} | Market Cap: {format_market_cap(fundamentals['market_cap'])}\n"
        f"P/E (trailing): {pe_trailing_str} | P/E (forward): {pe_forward_str}\n"
        f"EPS (trailing): {eps_str} | Dividend Yield: {div_yield_str}\n"
        f"52-Week Range: {range_str}\n"
        f"Operating Margin: {operating_margin_str} | ROE: {roe_str} | Price/Book: {price_to_book_str}\n"
        f"Price/Sales: {price_to_sales_str} | FCF Yield: {fcf_yield_str} | PEG: {peg_ratio_str}\n"
        f"Sector Median P/E: {sector_median_pe_str}"
    )
