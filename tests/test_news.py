from app.services import news


class FakeResponse:
    def __init__(self, content=b""):
        self.content = content

    def raise_for_status(self):
        pass


class FakeFeed:
    def __init__(self, entries):
        self.entries = entries


def _entry(title, published="Mon, 22 Jun 2026 00:00"):
    return {"title": title, "published": published}


def _install(monkeypatch, entries):
    monkeypatch.setattr(news.httpx, "get", lambda url, timeout=None: FakeResponse())
    monkeypatch.setattr(news.feedparser, "parse", lambda content: FakeFeed(entries))


# --- _parse_title_and_publisher ---


def test_parse_title_and_publisher_splits_on_trailing_dash():
    title, publisher = news._parse_title_and_publisher("Oracle cuts guidance - Reuters")
    assert title == "Oracle cuts guidance"
    assert publisher == "Reuters"


def test_parse_title_and_publisher_falls_back_when_no_dash():
    title, publisher = news._parse_title_and_publisher("Oracle cuts guidance")
    assert title == "Oracle cuts guidance"
    assert publisher == news.PUBLISHER_FALLBACK


# --- _is_low_content_headline ---


def test_low_content_headline_catches_trades_up_here_is_why():
    assert news._is_low_content_headline("Intel (INTC) Stock Trades Up, Here Is Why")


def test_low_content_headline_catches_why_stock_dropped_variant():
    assert news._is_low_content_headline("Why Tesla Stock Dropped on Tuesday")
    assert news._is_low_content_headline("Why is Amazon stock rallying today?")
    assert news._is_low_content_headline("Why Adobe (ADBE) Stock Is Falling Today")


def test_low_content_headline_catches_listicle_bait():
    assert news._is_low_content_headline("Should You Buy Microsoft Stock Before April 29?")
    assert news._is_low_content_headline("Is Oracle Stock a Buy at $245?")


def test_low_content_headline_catches_fund_flow_spam():
    assert news._is_low_content_headline("46,643 Shares in Oracle Corporation $ORCL Purchased by Trust Co. of Vermont")


def test_low_content_headline_leaves_real_news_alone():
    assert not news._is_low_content_headline("Oracle cuts guidance, citing softening cloud demand")
    assert not news._is_low_content_headline("Tesla signs major Arizona power deal")


# --- fetch_live_news_rag ---


def test_filters_out_low_quality_publisher(monkeypatch):
    _install(monkeypatch, [
        _entry("46,643 Shares in Oracle Corporation $ORCL Purchased by Trust Co. - MarketBeat"),
        _entry("Oracle cuts guidance, citing softening cloud demand - Reuters"),
    ])
    result = news.fetch_live_news_rag("ORCL")
    assert "MarketBeat" not in result
    assert "Oracle cuts guidance" in result


def test_filters_out_low_content_headline_from_a_normal_publisher(monkeypatch):
    _install(monkeypatch, [
        _entry("Why Tesla Stock Dropped on Tuesday - CNBC"),
        _entry("Tesla signs major Arizona power deal - CNBC"),
    ])
    result = news.fetch_live_news_rag("TSLA")
    assert "Why Tesla Stock Dropped" not in result
    assert "Tesla signs major Arizona power deal" in result


def test_keeps_at_most_max_news_items_from_filtered_pool(monkeypatch):
    _install(monkeypatch, [_entry(f"Real headline number {i} about the company - Reuters") for i in range(10)])
    result = news.fetch_live_news_rag("AAPL")
    assert len(result.splitlines()) == news.MAX_NEWS_ITEMS


def test_falls_back_to_raw_entries_when_every_candidate_is_low_quality(monkeypatch):
    _install(monkeypatch, [
        _entry("46,643 Shares Purchased by Trust Co. - MarketBeat"),
        _entry("TSLA Looks 2.0% Overvalued on GF Value - GuruFocus"),
        _entry("Why Tesla Stock Dropped on Tuesday - CNBC"),
    ])
    result = news.fetch_live_news_rag("TSLA")
    assert result != "Data unavailable."
    assert len(result.splitlines()) == news.FALLBACK_ITEMS_WHEN_NONE_PASS_FILTER


def test_output_line_format_includes_date_title_and_publisher(monkeypatch):
    _install(monkeypatch, [_entry("Oracle cuts guidance, citing softening cloud demand - Reuters", published="Mon, 22 Jun 2026 00:00")])
    result = news.fetch_live_news_rag("ORCL")
    assert result == "- [Mon, 22 Jun 2026] Oracle cuts guidance, citing softening cloud demand - Reuters"


def test_no_entries_returns_data_unavailable(monkeypatch):
    _install(monkeypatch, [])
    assert news.fetch_live_news_rag("ORCL") == "Data unavailable."


def test_fetch_error_returns_data_unavailable(monkeypatch):
    def _boom(url, timeout=None):
        raise Exception("network error")
    monkeypatch.setattr(news.httpx, "get", _boom)
    assert news.fetch_live_news_rag("ORCL") == "Data unavailable."
