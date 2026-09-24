import datetime
import logging
import re
import unicodedata
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
    "simplywall.st", "Zacks", "StockStory", "TIKR", "Moomoo",
    "TradingKey", "Quiver Quantitative",
}


def _publisher_tokens(publisher: str) -> tuple[str, ...]:
    """Lower-cased alphanumeric tokens, split on everything else, so one
    entry covers every spelling Google News uses for an outlet.

    It relabels them between runs - "MarketBeat" one day,
    "marketbeat.com" the next. Confirmed live 2026-09-18 (in
    portfolio-manager-backend, which runs this same filter): with
    exact-string matching, ALL FOUR results in IBM's 7-day window were
    "marketbeat.com" 13F spam, i.e. the exact source this list exists to
    exclude, passing through on spelling alone.
    """
    return tuple(re.findall(r"[a-z0-9]+", publisher.lower()))


_LOW_QUALITY_PUBLISHER_TOKENS = frozenset(_publisher_tokens(p) for p in LOW_QUALITY_PUBLISHERS)


def _is_low_quality_publisher(publisher: str) -> bool:
    """True when `publisher`'s leading tokens are a denylist entry, so a
    trailing domain or qualifier is covered: "marketbeat.com" ->
    ("marketbeat", "com") matches the "MarketBeat" entry, and "Zacks"
    alone now covers "Zacks.com" and "Zacks Investment Research".

    Matching whole TOKENS rather than a raw string prefix is what keeps
    the short stems safe - "TIKR" must not also deny a hypothetical
    "Tikrit Daily", which a plain `startswith` on alphanumerics-only keys
    would. "simplywall.st" stays in the list alongside "Simply Wall St."
    because the two tokenize differently ("simplywall" vs "simply",
    "wall") and neither is a token-prefix of the other.
    """
    tokens = _publisher_tokens(publisher)
    return any(tokens[: len(denied)] == denied for denied in _LOW_QUALITY_PUBLISHER_TOKENS if denied)


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
    # 2026-09-18: `here.s` matches "here's" and "heres" but NOT "Here Is
    # Why", which is how several outlets write it - "IBM Stock Trades Up
    # After Revenue Report, Here Is Why". "should know" added alongside
    # "need to know" for the same reason. Both widen rejection, which is
    # the safe direction for this filter (see the shortcut-learning note
    # in generate_real_dataset.py).
    r"|here(?:.|\s+i)s why|here(?:.|\s+i)s what (?:investors|we|you) (?:need to know|see|should know)"
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
    r"|portfolio.{0,20}(?:quiverquant|according to a)"
    # 2026-09-18, back-ported: more 13F filing-spam shapes, all four
    # observed in a single live IBM window. The existing branches above
    # miss them because a share count sits between the verb and "Shares",
    # or the verb isn't in their list.
    #
    # Every branch here keys on a SHARE COUNT or a dollar figure, never on
    # a word that merely sounds financial. "Bank", "Capital", "Financial",
    # "Trust" and "Management" are also just what financial-sector issuers
    # are called, so gating on those rejects "Bank of America Buys Stake
    # in Fintech Startup" and "Prudential Financial Sells Shares of Its
    # Annuity Unit" - real corporate events, and a denylist match has
    # nothing downstream to rescue it.
    #
    # A percentage is NOT a usable gate either, for the same reason: a
    # company raising its own stake ("Berkshire Hathaway Boosts Stake in
    # Occidental Petroleum by 5%") reads identically to a fund's 13F
    # delta. So "Baird Financial Group Inc. Reduces Position in IBM" - a
    # real observed spam headline with no quantity at all - is a KNOWN,
    # deliberate gap here: nothing in the headline distinguishes it from
    # an issuer doing the same thing. That shape is caught at the
    # publisher tier instead (it came from marketbeat.com), which is the
    # right layer for it - see _is_low_quality_publisher.
    # A "Shares of X ... Acquired by Y" branch with a gap between the two
    # halves was tried and removed: `.{0,60}` also matches ordinary M&A
    # reporting ("Shares of Activision Jumped After the Company Was
    # Acquired by Microsoft"), which is a large, high-value, causally
    # informative category. The pre-existing adjacent form above
    # ("shares acquired by") and the count-led form ("46,643 Shares in
    # ...") already cover the spam without that gap.
    r"|\b(?:Acquires|Buys|Purchases|Sells|Snaps Up|Reduces|Boosts|Trims|Grows)\s+"
    r"(?:its\s+)?(?:stake|position|holdings?)?\s*(?:of|in|by)?\s*[\d,]+\+?\s+shares\b"
    r"|\bHas \$[\d,.]+ (?:Million|Billion) (?:Stock )?(?:Holdings|Position|Stake)\b",
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


# 2026-09-18, back-ported: legal-entity suffixes dropped from a company
# name before it's matched against a headline. `name` reaches this module
# from yfinance's `.info` (shortName/longName, see fundamentals.py), which
# reports the full REGISTERED name - "Nestle S.A.", "Alphabet Inc.",
# "Mondi plc", "The Coca-Cola Company" - while headlines write the plain
# brand. A raw substring test therefore never fires for most non-US names,
# and the headline falls through to the much weaker ticker and sector
# tiers.
_LEGAL_SUFFIX_WORDS = frozenset(
    {
        "inc", "incorporated", "corp", "corporation", "co", "cos", "company",
        "ltd", "limited", "plc", "llc", "lp", "llp", "pte", "sarl", "gmbh",
        "sa", "sas", "nv", "bv", "ag", "se", "spa", "ab", "asa", "oyj", "as", "kgaa",
        "holding", "holdings", "group", "the",
    }
)

# Below this, a needle is too generic to match on: the tier is
# case-INsensitive, so a 1-character needle turns every "v." citation into
# a Visa story. 2 rather than 3 because `\b` now does the anti-substring
# work, and real 2-character brands exist ("3M" -> "3m").
_MIN_NEEDLE_LENGTH = 2

# Single-token cores that are also ordinary English words. Suffix
# stripping goes too far for these: "Target Corporation" -> "target" then
# matches "Analysts Raise Price Target for Nvidia to $200", "Sea Limited"
# -> "sea" matches "Rescues Sailors After Storm at Sea", and "Box Inc." ->
# "box" matches "Cardboard Box Shortage". The name tier short-circuits
# _is_relevant_headline, so nothing downstream catches the mistake - the
# headline is simply served as if it were about this company.
#
# For these the FULL normalized name is required instead, i.e. the
# behaviour that predates suffix stripping, which is safe precisely
# because "target corporation" as a phrase is not ordinary English.
#
# The cost is a real false negative: "Shell reports record profit" no
# longer matches on the name tier. That's the direction to err in. A
# false positive pairs an unrelated headline with this ticker - in the
# API the model then reasons about the wrong company's news, and in
# generate_real_dataset.py it becomes a training row pairing that
# headline with this ticker's price move, i.e. label noise. A false
# negative only costs a better candidate; the ticker tier below still
# catches the very common "Shell (SHEL) reports ..." RSS phrasing.
#
# Whack-a-mole like the rest of this module's denylists - extend it when
# a collision shows up, and the three verified cases are the seed.
_AMBIGUOUS_NAME_CORES = frozenset(
    {
        "target", "sea", "box", "gap", "shell", "ford", "key", "cross",
        "square", "block", "match", "unity", "arrow", "sun", "star",
        "first", "general", "national", "global", "standard", "premier",
        "energy", "power", "health", "service", "systems", "industries",
        "brands", "foods", "express", "motion", "signal", "vision",
        "focus", "edge", "peak", "summit", "pioneer", "eagle", "anchor",
        "compass", "apex", "core", "prime", "elite", "liberty",
        "atlantic", "pacific", "western", "eastern", "northern",
        "southern", "central",
    }
)


def _normalize_for_match(text: str) -> str:
    """Accents folded, lower-cased, periods and commas dropped, whitespace
    collapsed - applied to BOTH the name and the headline so "Nestle S.A."
    and "Nestle SA" compare equal.

    Deliberately leaves "/" alone, unlike "." and ",": a slash in headline
    text is usually a real word separator ("Baidu/Alibaba race for AI
    dominance"), and dropping it would merge the two sides into one token
    and break the `\\b` word-boundary match on either name.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower().replace(".", "").replace(",", "")
    return " ".join(folded.split())


def _company_match_name(name: str) -> str:
    """The brand part of a registered company name - "Nestle S.A." ->
    "nestle". Stripped from BOTH ends: "The Coca-Cola Company" tail-only
    leaves "the coca-cola", which never appears in a headline that writes
    "Coca-Cola". "" when nothing survives, so the caller can fall back.

    A Danish "A/S" suffix (left intact by _normalize_for_match, see its
    own docstring) still matches the "as" entry here.

    Known limitation, deliberately not chased: a name carrying its brand
    AFTER the suffix ("Petroleo Brasileiro S.A. - Petrobras") keeps the
    whole string and won't match a "Petrobras ..." headline on this tier -
    it still has the ticker tier.
    """
    words = _normalize_for_match(name).split()
    while words and words[0].replace("/", "") in _LEGAL_SUFFIX_WORDS:
        words.pop(0)
    while words and words[-1].replace("/", "") in _LEGAL_SUFFIX_WORDS:
        words.pop()
    return " ".join(words)


def _name_needle(name: str) -> str:
    """The string a headline is matched against for this company - the
    brand core, or the whole normalized name when stripping leaves
    nothing. "" when neither is long enough to be worth matching, i.e.
    skip the name tier entirely.

    The length floor is applied to the RESULT, not just the core: the API
    passes the ticker itself as `name` when yfinance has no company name
    (see routers/analyze.py), so the fallback can otherwise be a single
    letter.
    """
    if not name:
        return ""
    core = _company_match_name(name)
    # A multi-word core is specific enough to match on as-is. A
    # single-word one is only safe if it isn't ordinary English.
    if core and (" " in core or core not in _AMBIGUOUS_NAME_CORES):
        needle = core
    else:
        needle = _normalize_for_match(name)
    return needle if len(needle) >= _MIN_NEEDLE_LENGTH else ""


def _is_relevant_headline(ticker: str, name: str, sector, title: str) -> bool:
    """True if `title` plausibly concerns `ticker`'s company or its sector -
    the redesign plan's relevance pre-filter. Checked in order: (1)
    company name, accent-folded and legal-suffix-stripped, matched on a
    word boundary (see _name_needle), (2) ticker as a standalone,
    case-SENSITIVE token (tickers are conventionally all-caps in real
    headlines - a case-insensitive check on short tickers like V/F/MA/GS
    would false-positive on ordinary English words), (3) sector keyword
    match, if this ticker's sector has an entry. A headline matching none
    of these is dropped before it can be selected."""
    needle = _name_needle(name)
    # Word-boundary, not plain substring: "Sea Limited" -> "sea" would
    # otherwise match the "sea" inside "research" and let an unrelated
    # company's story become this ticker's signal headline.
    if needle and re.search(rf"\b{re.escape(needle)}\b", _normalize_for_match(title)):
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

        candidates = _build_candidates(feed)
        if not candidates:
            return "Data unavailable.", None

        selected = _select_candidate(candidates, ticker, search_ticker, name, sector)
        text = f"- [{selected.published_display}] {selected.title} - {selected.publisher}"
        return text, selected.published_date

    except Exception as e:
        logger.warning("RAG Google News RSS error for %s: %s", ticker, e)
        return "Data unavailable.", None


def _build_candidates(feed) -> list[_Candidate]:
    """Builds the candidate pool from a parsed feedparser feed - see
    fetch_live_news_rag's own docstring for the selection pipeline this
    feeds into."""
    candidates = []
    for entry in feed.entries[:CANDIDATE_POOL_SIZE]:
        raw_title = entry.get("title", "")
        if not raw_title:
            continue
        title, publisher = _parse_title_and_publisher(raw_title)
        # Google News RSS's published string is fixed-width (e.g. "Mon,
        # 04 Sep 2026 12:34:56 GMT") - [:16] isolates just the "Mon, 04
        # Sep 2026" date portion, dropping the time.
        published_display = entry.get("published", "")[:16]
        candidates.append(_Candidate(title, publisher, published_display, _parse_published_date(entry)))
    return candidates


def _select_candidate(candidates: list[_Candidate], ticker: str, search_ticker: str, name: str, sector) -> _Candidate:
    """3-tier selection over a non-empty `candidates` list - see
    fetch_live_news_rag's own docstring for the overall pipeline."""
    quality_filtered = [
        c for c in candidates
        if not _is_low_quality_publisher(c.publisher) and not _is_low_content_headline(c.title)
    ]
    relevant = [c for c in quality_filtered if _is_relevant_headline(search_ticker, name, sector, c.title)]

    if relevant:
        return _most_recent(relevant)
    if quality_filtered:
        # Nothing relevant survived, but well-sourced/well-shaped
        # headlines exist (e.g. a genuinely quiet news week for this
        # specific company) - fall back to the most recent of those
        # rather than "Data unavailable.": some context beats none,
        # same fallback spirit the old quality-only fallback had.
        logger.info(
            "fetch_live_news_rag[%s]: no candidate passed the relevance filter, falling back to most recent quality-filtered entry",
            ticker,
        )
        return _most_recent(quality_filtered)
    # Every candidate in the pool was low-quality - fall back to the
    # single most recent raw entry rather than "Data unavailable.".
    logger.info(
        "fetch_live_news_rag[%s]: no candidate passed quality filtering, falling back to most recent raw entry",
        ticker,
    )
    return _most_recent(candidates)
