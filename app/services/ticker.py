import re

TICKER_STOPWORDS = {
    "I", "A", "AI", "CEO", "CFO", "IPO", "USA", "ETF", "NEWS", "WILL",
    "THE", "AND", "FOR", "ARE", "YOU", "ITS", "GDP", "SEC", "FED",
}


def extract_ticker(text: str) -> str:
    # Match against the original casing, not text.upper() - uppercasing first
    # made every 3-5 letter word in the query a "ticker" (e.g. the UI's own
    # placeholder "Will AAPL go up..." matched "WILL" before AAPL).
    for candidate in re.findall(r'\b[A-Z]{2,5}\b', text):
        if candidate not in TICKER_STOPWORDS:
            return candidate

    return "AAPL"
