import datetime
import logging
import re
import urllib.parse
from dataclasses import dataclass

import feedparser
import httpx

logger = logging.getLogger(__name__)

# Strips a Yahoo-style exchange suffix (".SW", ".DE", ...) before using a
# ticker for full-text search or headline matching - added 2026-08-20,
# real bug found live: once app.services.ticker's cashtag regex started
# preserving an explicit suffix ("$ARYN.SW" -> "ARYN.SW", see that
# module's own history), the SAME suffixed value flowed into this file's
# Google News query and into _is_relevant_headline's ticker match. Google
# News search treats "ARYN.SW" as a literal token no article's text
# actually contains - confirmed live: "ARYN.SW stock earnings financial
# news" returned 0 entries, while the bare "ARYN stock earnings financial
# news" returned 23 and the company name returned 38. The relevance
# filter had the identical problem in reverse - re.escape("ARYN.SW")
# requires the literal ".SW" in the headline text too, so a real,
# relevant headline that only ever writes the bare ticker ("ARYZTA
# (SWX:ARYN)") would have been wrongly rejected. yfinance calls
# (fundamentals.py) still need and use the full suffixed ticker - only
# this module's search/matching needs it stripped.
_EXCHANGE_SUFFIX_RE = re.compile(r"\.[A-Za-z]{1,3}$")

PUBLISHER_FALLBACK = "Google News"

# How many raw RSS entries to consider as candidates before filtering -
# NOT the final count shown to the model (exactly one headline is always
# selected, see fetch_live_news_rag). Needs to be comfortably larger than
# 1 since a real, confirmed-live fraction of raw results are unusable (see
# LOW_QUALITY_PUBLISHERS/_is_low_content_headline below) or simply not
# about this company (_is_relevant_headline) - surveying live Google News
# RSS results for ORCL/TSLA/QCOM (2026-08-19) found a SINGLE publisher
# (MarketBeat) alone accounted for 36-40% of ALL 100 raw results per
# ticker, almost entirely auto-generated 13F-filing spam ("46,643 Shares
# in Oracle Corporation $ORCL Purchased by Trust Co. of Vermont"), not
# news.
CANDIDATE_POOL_SIZE = 30

# Ported from financial-sentiment-model's generate_real_dataset.py (same
# denylist, "ported not imported" - the two repos deliberately stay
# independent, same convention already used for the DCF valuation math -
# see that file's own comment for the live-survey numbers behind this
# list). Homogeneous, near-100% low-content sources are excluded from
# candidacy entirely, distinct from heterogeneous opinion/commentary
# outlets (Motley Fool, Benzinga, 24/7 Wall St.) that mix real reporting
# with opinion pieces and are deliberately left in - the headline-shape
# filter below already catches their worst individual offenders.
LOW_QUALITY_PUBLISHERS = {
    "MarketBeat", "Stocktwits", "GuruFocus", "Trefis", "Simply Wall St.",
    "simplywall.st", "Zacks Investment Research", "StockStory",
    "TIKR.com", "Moomoo", "TradingKey", "Quiver Quantitative",
}

# Ported from financial-sentiment-model's generate_real_dataset.py (same
# pattern, same "ported not imported" convention, same whack-a-mole
# caveat noted there - this is a regex denylist, not a robust classifier,
# and will keep missing new phrasings). Headline SHAPES that report THAT
# a stock moved without saying why ("Intel Stock Trades Up, Here Is Why",
# "Why Tesla Stock Dropped on Tuesday"), plus listicle/opinion bait
# ("Should You Buy Microsoft Stock?") and fund-flow filing spam.
_MOVE_VERB_RE_FRAGMENT = (
    r"(?:ris(?:e|es|ing)|fell|fall(?:s|ing)?|dropp?(?:ed|s|ing)?|"
    r"rall(?:y|ies|ying|ied)|slid(?:e|es|ing)?|climb(?:s|ed|ing)?|"
    r"surg(?:e|es|ed|ing)?|plung(?:e|es|ed|ing)?|jump(?:s|ed|ing)?|"
    r"sank|sink(?:s|ing)?|tumbl(?:e|es|ed|ing)|gain(?:s|ed|ing)?|"
    r"los(?:es|ing)|lost|nosediv(?:e|es|ed|ing)|soar(?:s|ed|ing)?|"
    r"sag(?:s|ged|ging)?|slump(?:s|ed|ing)?|wilt(?:s|ed|ing)?|"
    r"slip(?:s|ped|ping)?|retreat(?:s|ed|ing)?|advanc(?:e|es|ed|ing)?|"
    r"wobbl(?:e|es|ed|ing)|sink|dip(?:s|ped|ping)?|swoon(?:s|ed|ing)?|"
    r"spik(?:e|es|ed|ing)|skid(?:s|ded|ding)?)"
)

# 2026-08-20: two branches added/broadened after the fine-tune's Task A
# eval surfaced these exact phrasings slipping through - "stock" added
# as a subject alongside "shares" in the up/down/higher/lower branch,
# and a new bait pattern for "Is X Still Y After Z" headlines whose
# Z-clause names an actual price move (%, rally, surge, etc.) - kept
# narrow after an early broader version false-positived on genuine
# analysis headlines. See financial-sentiment-model's generate_real_
# dataset.py module docstring for the full history/reasoning (ported
# not imported, same convention as the rest of this filter).
_LOW_CONTENT_HEADLINE_RE = re.compile(
    r"stock (?:is )?trad(?:ing|es) (?:up|down|higher|lower)"
    r"|(?:shares|stock) (?:are|is) (?:up|down|higher|lower) today"
    r"|here.s why|here.s what (?:investors|we|you) (?:need to know|see)"
    r"|what you need to know|laps the stock market|what.s going on with"
    rf"|\bwhy\b.{{0,60}}\b(?:stock|shares?)\b.{{0,30}}\b{_MOVE_VERB_RE_FRAGMENT}\b"
    rf"|\b(?:stock|shares?)\b.{{0,20}}\b(?:is|are)\b.{{0,10}}\b{_MOVE_VERB_RE_FRAGMENT}(?:ing)?\b"
    r"|^Is .+ a Good Stock"
    r"|^Is .{1,60}\bStill\b.{1,25}\bAfter\b.{0,40}(?:\d+%|rally|surge|drop|plunge|rout|gain|dip|slump|rebound)"
    r"|Stock a (?:Good )?Buy\b|^Should You Buy|Buy,? Hold,? (?:or|and) Sell"
    r"|^\d+ (?:Reasons?|Stocks?)|Better Buy|Zacks (?:Investment|Rank)|Trending Stock"
    r"|shares (?:added to|removed from|acquired by|sold by|purchased by)"
    r"|^[\d,]+\+? Shares (?:in|of)|(?:Buys|Purchases?|Sells) Shares (?:in|of)"
    r"|(?:Takes|Makes New) .{0,25}(?:Position|Investment) in|Invests? \$[\d,.]+|13F"
    r"|portfolio.{0,20}(?:quiverquant|according to a)",
    re.IGNORECASE,
)


def _is_low_content_headline(title: str) -> bool:
    return bool(_LOW_CONTENT_HEADLINE_RE.search(title))


# Ported from financial-sentiment-model's generate_real_dataset.py
# (SECTOR_KEYWORDS/SECTOR_KEYWORD_PATTERNS/_is_relevant_headline, same
# "ported not imported" convention) - keep this byte-identical to that
# copy per the redesign plan (docs/two-stage-task-a-redesign-plan.md item
# 2), same as price_context_block's phrasing already had to stay in sync
# across both repos.
SECTOR_KEYWORDS = {
    "Technology": {
        "ai", "artificial intelligence", "chip", "chips", "semiconductor",
        "software", "cloud", "cybersecurity", "data center", "data centers",
    },
    "Communication Services": {
        "streaming", "advertising", "ad revenue", "social media", "telecom",
        "wireless", "broadband", "5g",
    },
    "Consumer Cyclical": {
        "retail sales", "consumer spending", "e-commerce", "auto sales",
        "vehicle sales", "electric vehicle", "tariff", "tariffs",
    },
    "Consumer Defensive": {
        "retail sales", "consumer spending", "grocery", "beverage",
    },
    "Financial Services": {
        "rate hike", "rate cut", "federal reserve", "fed", "banking",
        "interest rates", "credit", "payments", "fintech",
    },
    "Industrials": {
        "aerospace", "defense", "manufacturing", "supply chain", "factory",
        "airline", "aviation",
    },
    "Energy": {
        "oil", "gas", "crude", "opec", "drilling", "refinery", "pipeline",
    },
    "Healthcare": {
        "drug", "fda", "clinical trial", "biotech", "pharma", "vaccine",
    },
}
SECTOR_KEYWORD_PATTERNS = {
    sector: re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    for sector, keywords in SECTOR_KEYWORDS.items()
}


def _is_relevant_headline(ticker: str, name: str, sector, title: str) -> bool:
    """True if `title` plausibly concerns `ticker`'s company or its sector -
    the redesign plan's relevance pre-filter. Checked in order: (1)
    company name substring (case-insensitive), (2) ticker as a standalone,
    case-SENSITIVE token (tickers are conventionally all-caps in real
    headlines - a case-insensitive check on short tickers like V/F/MA/GS
    would false-positive on ordinary English words), (3) sector keyword
    match, if this ticker's sector has an entry. A headline matching none
    of these is dropped before it can be selected."""
    title_lower = title.lower()
    if name and name.lower() in title_lower:
        return True
    if re.search(rf"\b{re.escape(ticker)}\b", title):
        return True
    keyword_pattern = SECTOR_KEYWORD_PATTERNS.get(sector)
    if keyword_pattern is not None and keyword_pattern.search(title):
        return True
    return False


def _parse_title_and_publisher(raw_title: str) -> tuple[str, str]:
    """Google News RSS titles are conventionally "Headline - Publisher" -
    split it out so LOW_QUALITY_PUBLISHERS can filter on it (same
    convention generate_real_dataset.py's fetch_headlines_for_window
    already relies on). Falls back to PUBLISHER_FALLBACK when the
    convention doesn't hold for a given entry, so a missing publisher can
    never accidentally match the denylist."""
    if " - " in raw_title:
        title, _, publisher = raw_title.rpartition(" - ")
        return title, publisher
    return raw_title, PUBLISHER_FALLBACK


def _parse_published_date(entry) -> datetime.date | None:
    """feedparser exposes a pre-parsed `published_parsed` (a UTC
    time.struct_time) whenever it recognizes the entry's date format -
    used instead of hand-parsing the raw `published` display string, and
    needed both to rank candidates by recency and to hand fundamentals.
    price_move_on_date a real date. None if feedparser couldn't parse it
    (entry missing/malformed date) - callers treat that as "no date
    signal", not a fetch failure."""
    parsed = entry.get("published_parsed")
    if not parsed:
        return None
    try:
        return datetime.date(parsed.tm_year, parsed.tm_mon, parsed.tm_mday)
    except (TypeError, ValueError):
        return None


@dataclass
class _Candidate:
    title: str
    publisher: str
    published_display: str
    published_date: datetime.date | None


def _most_recent(candidates: list[_Candidate]) -> _Candidate:
    """Picks the most recently published candidate (user-confirmed
    selection rule when multiple relevant headlines survive filtering -
    see the redesign plan). Falls back to the first candidate in original
    RSS order if none of them have a parseable date (Google News RSS
    already returns results roughly newest-first, so this is a reasonable
    degrade, not an arbitrary pick)."""
    with_date = [c for c in candidates if c.published_date is not None]
    if with_date:
        return max(with_date, key=lambda c: c.published_date)
    return candidates[0]


def fetch_live_news_rag(ticker: str, name: str, sector) -> tuple[str, datetime.date | None]:
    """Returns (live_context text, published_date of the selected
    headline). Selects exactly ONE headline - the most recently published
    candidate that passes quality (LOW_QUALITY_PUBLISHERS/_is_low_
    content_headline) AND relevance (_is_relevant_headline) filtering -
    matching training's actual per-row shape: financial-sentiment-model's
    generate_real_dataset.py has always been single-headline-per-row, but
    this used to join up to 4 headlines into one block backed by a single
    TRAILING price move, a real train/inference mismatch. The single
    selected headline's published_date lets the caller (see
    app.routers.analyze) fetch a price move anchored to the SAME day
    training measures (fundamentals.price_move_on_date), not a trailing-
    as-of-now approximation. Task B receives this same text (for
    traceability - its reasoning should reference the actual news Task A
    reacted to), not a separately fetched multi-headline block.

    ("Data unavailable.", None) if the feed is empty or every entry fails
    to parse, or on any fetch error.
    """
    search_ticker = _EXCHANGE_SUFFIX_RE.sub("", ticker)
    try:
        query = f"{search_ticker} stock earnings financial news"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        # feedparser.parse(url) fetches internally via urllib with no
        # timeout - a hung connection would pin this thread (and the
        # asyncio.to_thread pool slot it came from) forever. Fetch with an
        # explicit timeout ourselves and hand feedparser the bytes instead.
        response = httpx.get(rss_url, timeout=10.0)
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        if not feed.entries:
            return "Data unavailable.", None

        candidates = []
        for entry in feed.entries[:CANDIDATE_POOL_SIZE]:
            raw_title = entry.get("title", "")
            if not raw_title:
                continue
            title, publisher = _parse_title_and_publisher(raw_title)
            published_display = entry.get("published", "")[:16]  # Date string snippet
            candidates.append(_Candidate(title, publisher, published_display, _parse_published_date(entry)))

        if not candidates:
            return "Data unavailable.", None

        quality_filtered = [
            c for c in candidates
            if c.publisher not in LOW_QUALITY_PUBLISHERS and not _is_low_content_headline(c.title)
        ]
        relevant = [c for c in quality_filtered if _is_relevant_headline(search_ticker, name, sector, c.title)]

        if relevant:
            selected = _most_recent(relevant)
        elif quality_filtered:
            # Nothing relevant survived, but well-sourced/well-shaped
            # headlines exist (e.g. a genuinely quiet news week for this
            # specific company) - fall back to the most recent of those
            # rather than "Data unavailable.": some context beats none,
            # same fallback spirit the old quality-only fallback had.
            selected = _most_recent(quality_filtered)
            logger.info(
                "fetch_live_news_rag[%s]: no candidate passed the relevance filter, falling back to most recent quality-filtered entry",
                ticker,
            )
        else:
            # Every candidate in the pool was low-quality - fall back to
            # the single most recent raw entry rather than "Data
            # unavailable.".
            selected = _most_recent(candidates)
            logger.info(
                "fetch_live_news_rag[%s]: no candidate passed quality filtering, falling back to most recent raw entry",
                ticker,
            )

        text = f"- [{selected.published_display}] {selected.title} - {selected.publisher}"
        return text, selected.published_date

    except Exception as e:
        logger.warning("RAG Google News RSS error for %s: %s", ticker, e)
        return "Data unavailable.", None
