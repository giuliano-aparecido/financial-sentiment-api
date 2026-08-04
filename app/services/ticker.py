import re

TICKER_STOPWORDS = {
    "I", "A", "AI", "CEO", "CFO", "IPO", "USA", "ETF", "NEWS", "WILL",
    "THE", "AND", "FOR", "ARE", "YOU", "ITS", "GDP", "SEC", "FED",
}

# Cashtag convention (e.g. $AAPL, $aapl) - explicit user intent, so it's
# trusted without stopword filtering and takes priority over the fallback
# heuristic below.
CASHTAG_RE = re.compile(r'\$([A-Za-z]{1,5})\b')


def extract_ticker(text: str) -> str:
    cashtag = CASHTAG_RE.search(text)
    if cashtag:
        return cashtag.group(1).upper()

    # Fallback: match against the original casing, not text.upper() -
    # uppercasing first made every 3-5 letter word in the query a "ticker"
    # (e.g. the UI's own placeholder "Will AAPL go up..." matched "WILL"
    # before AAPL).
    for candidate in re.findall(r'\b[A-Z]{2,5}\b', text):
        if candidate not in TICKER_STOPWORDS:
            return candidate

    return "AAPL"
