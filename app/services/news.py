import logging
import urllib.parse

import feedparser

logger = logging.getLogger(__name__)


def fetch_live_news_rag(ticker: str) -> str:
    try:
        # Construct clean query for Google News RSS
        query = f"{ticker} stock earnings financial news"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        # Parse XML feed directly (No API key, No IP rate limits)
        feed = feedparser.parse(rss_url)

        if not feed.entries:
            return f"Recent market volatility and financial developments for {ticker}."

        # Extract top 4 news headlines with publication dates
        news_items = []
        for entry in feed.entries[:4]:
            title = entry.get("title", "")
            published = entry.get("published", "")[:16]  # Date string snippet
            news_items.append(f"- [{published}] {title}")

        return "\n".join(news_items)

    except Exception as e:
        logger.warning("RAG Google News RSS error for %s: %s", ticker, e)
        return f"Recent quarterly earnings and news updates for {ticker}."
