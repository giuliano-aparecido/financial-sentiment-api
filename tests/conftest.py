import pytest

from app.services import news, news_classifier


@pytest.fixture(autouse=True)
def _no_live_news_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(news_classifier, "classify_headlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(news, "_fetch_yahoo_candidates", lambda ticker: [])
