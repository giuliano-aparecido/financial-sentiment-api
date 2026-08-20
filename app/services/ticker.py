import re

TICKER_STOPWORDS = {
    "I", "A", "AI", "CEO", "CFO", "IPO", "USA", "ETF", "NEWS", "WILL",
    "THE", "AND", "FOR", "ARE", "YOU", "ITS", "GDP", "SEC", "FED",
}

# Cashtag convention (e.g. $AAPL, $aapl, $ARYN.SW, $NESN.SW) - explicit
# user intent, so it's trusted without stopword filtering and takes
# priority over the fallback heuristic below. The optional exchange-
# suffix group (added 2026-08-20, real bug found live: a user explicitly
# wrote "$ARYN.SW" and got the ".SW" silently stripped, forcing a
# dependency on fundamentals.resolve_ticker's live Yahoo search call to
# add it back - which failed when Yahoo's search endpoint ALSO started
# rate-limiting that day, even though the user had already given the
# exact right answer) means a fully-qualified cashtag now needs zero
# extra network calls to resolve correctly - resolve_ticker's search
# fallback still exists for the (still common) bare-cashtag non-US case
# ("$NESN" -> "NESN.SW"), unchanged. The trailing \b after the whole
# optional group means a stray "$AAPL. Great stock" (space after the
# period) or "$AAPL.something-longer" (falls outside the {1,3} length or
# lacks a boundary right after) both correctly fall back to matching
# "AAPL" alone, not swallowing unrelated trailing text - see this
# module's own tests for the specific cases checked before relying on
# this.
CASHTAG_RE = re.compile(r'\$([A-Za-z]{1,5}(?:\.[A-Za-z]{1,3})?)\b')


def extract_ticker(text: str) -> tuple[str, bool]:
    """Returns (ticker, was_explicit). was_explicit is False only when the
    query named no cashtag and no plausible all-caps candidate, so the
    "AAPL" default was used - callers use this to warn the user their
    analysis is for a stock they never actually asked about, rather than
    silently returning an Apple analysis for an unrelated question.
    """
    cashtag = CASHTAG_RE.search(text)
    if cashtag:
        return cashtag.group(1).upper(), True

    # Fallback: match against the original casing, not text.upper() -
    # uppercasing first made every 3-5 letter word in the query a "ticker"
    # (e.g. the UI's own placeholder "Will AAPL go up..." matched "WILL"
    # before AAPL).
    for candidate in re.findall(r'\b[A-Z]{2,5}\b', text):
        if candidate not in TICKER_STOPWORDS:
            return candidate, True

    return "AAPL", False
