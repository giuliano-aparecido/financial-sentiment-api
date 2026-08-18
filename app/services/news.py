import logging
import re
import urllib.parse

import feedparser
import httpx

logger = logging.getLogger(__name__)

PUBLISHER_FALLBACK = "Google News"

# How many raw RSS entries to consider as candidates before filtering -
# NOT the final count shown to the model (see MAX_NEWS_ITEMS below). Needs
# to be comfortably larger than MAX_NEWS_ITEMS since a real, confirmed-
# live fraction of raw results are unusable (see LOW_QUALITY_PUBLISHERS/
# _is_low_content_headline below) - surveying live Google News RSS results
# for ORCL/TSLA/QCOM (2026-08-19) found a SINGLE publisher (MarketBeat)
# alone accounted for 36-40% of ALL 100 raw results per ticker, almost
# entirely auto-generated 13F-filing spam ("46,643 Shares in Oracle
# Corporation $ORCL Purchased by Trust Co. of Vermont"), not news.
CANDIDATE_POOL_SIZE = 30

# Final number of headlines shown to the model - unchanged from the
# previous "top 4" behavior.
MAX_NEWS_ITEMS = 4

# If quality filtering leaves fewer than this many substantive headlines,
# fall back to the best-available RAW entries rather than "Data
# unavailable." - a request landing during a genuinely quiet news week
# for a ticker still deserves SOME context. Deliberately smaller than
# MAX_NEWS_ITEMS rather than padding a bad result up to the normal count -
# see fetch_live_news_rag's own comment.
FALLBACK_ITEMS_WHEN_NONE_PASS_FILTER = 2

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
    r"sag(?:s|ged|ging)?|slump(?:s|ed|ing)?)"
)
_LOW_CONTENT_HEADLINE_RE = re.compile(
    r"stock (?:is )?trad(?:ing|es) (?:up|down|higher|lower)"
    r"|shares (?:are|is) (?:up|down|higher|lower) today"
    r"|here.s why|here.s what (?:investors|we|you) (?:need to know|see)"
    r"|what you need to know|laps the stock market"
    rf"|\bwhy\b.{{0,60}}\b(?:stock|shares?)\b.{{0,30}}\b{_MOVE_VERB_RE_FRAGMENT}\b"
    rf"|\b(?:stock|shares?)\b.{{0,20}}\b(?:is|are)\b.{{0,10}}\b{_MOVE_VERB_RE_FRAGMENT}(?:ing)?\b"
    r"|^Is .+ a Good Stock|Stock a (?:Good )?Buy\b|^Should You Buy|Buy,? Hold,? (?:or|and) Sell"
    r"|^\d+ (?:Reasons?|Stocks?)|Better Buy|Zacks (?:Investment|Rank)|Trending Stock"
    r"|shares (?:added to|removed from|acquired by|sold by|purchased by)"
    r"|^[\d,]+\+? Shares (?:in|of)|(?:Buys|Purchases?|Sells) Shares (?:in|of)"
    r"|(?:Takes|Makes New) .{0,25}(?:Position|Investment) in|Invests? \$[\d,.]+|13F"
    r"|portfolio.{0,20}(?:quiverquant|according to a)",
    re.IGNORECASE,
)


def _is_low_content_headline(title: str) -> bool:
    return bool(_LOW_CONTENT_HEADLINE_RE.search(title))


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


def fetch_live_news_rag(ticker: str) -> str:
    try:
        # Construct clean query for Google News RSS
        query = f"{ticker} stock earnings financial news"
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
            return "Data unavailable."

        # Fetch a larger candidate POOL than what's actually shown, then
        # filter for quality - confirmed live (2026-08-19, user feedback):
        # taking the raw top 4 unfiltered routinely handed the model 2-3
        # headlines that were ALL low-content noise (see LOW_QUALITY_
        # PUBLISHERS/_is_low_content_headline above), leaving nothing
        # substantive to reason about.
        candidates = []
        for entry in feed.entries[:CANDIDATE_POOL_SIZE]:
            raw_title = entry.get("title", "")
            if not raw_title:
                continue
            title, publisher = _parse_title_and_publisher(raw_title)
            published = entry.get("published", "")[:16]  # Date string snippet
            candidates.append((title, publisher, published))

        filtered = [
            c for c in candidates
            if c[1] not in LOW_QUALITY_PUBLISHERS and not _is_low_content_headline(c[0])
        ]

        if filtered:
            selected = filtered[:MAX_NEWS_ITEMS]
        else:
            # Every candidate in the pool was low-quality (a genuinely
            # quiet news week, or an unusually noisy source mix for this
            # ticker) - fall back to a SMALL number of the best-available
            # raw entries rather than "Data unavailable.": some context is
            # better than none, but this deliberately stays well under
            # MAX_NEWS_ITEMS rather than padding a bad result up to the
            # normal count.
            selected = candidates[:FALLBACK_ITEMS_WHEN_NONE_PASS_FILTER]
            if selected:
                logger.info(
                    "fetch_live_news_rag[%s]: no candidate passed quality filtering, falling back to %d raw entries",
                    ticker, len(selected),
                )

        if not selected:
            return "Data unavailable."

        news_items = [f"- [{published}] {title} - {publisher}" for title, publisher, published in selected]
        return "\n".join(news_items)

    except Exception as e:
        logger.warning("RAG Google News RSS error for %s: %s", ticker, e)
        return "Data unavailable."
