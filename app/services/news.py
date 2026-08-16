import logging
import urllib.parse

import feedparser
import httpx

logger = logging.getLogger(__name__)


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

        # Extract top 4 news headlines with publication dates
        news_items = []
        for entry in feed.entries[:4]:
            title = entry.get("title", "")
            published = entry.get("published", "")[:16]  # Date string snippet
            news_items.append(f"- [{published}] {title}")

        return "\n".join(news_items)

    except Exception as e:
        logger.warning("RAG Google News RSS error for %s: %s", ticker, e)
        return "Data unavailable."
