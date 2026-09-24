"""news_classifier - the Gemini screen, with the SDK client faked. No network."""

from types import SimpleNamespace

import pytest

from app import config
from app.services import news_classifier
from app.services.news_classifier import HeadlineAssessment, _Assessment, classify_headlines

HEADLINES = [("Novo Sets Long-Term Targets", "Reuters"), ("Novo Banco reports profit surge", "Reuters")]


def _fake_client(parsed=None, error=None, calls=None):
    class _Models:
        def generate_content(self, model, contents, config):
            if calls is not None:
                calls.append(contents)
            if error:
                raise error
            return SimpleNamespace(parsed=parsed)

    class _Client:
        def __init__(self, api_key, http_options):
            self.models = _Models()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Client


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")


def test_assessments_come_back_in_input_order_with_importance_clamped(configured, monkeypatch):
    parsed = [_Assessment(index=1, relevant=False, importance=0), _Assessment(index=0, relevant=True, importance=9)]
    monkeypatch.setattr(news_classifier.genai, "Client", _fake_client(parsed))

    result = classify_headlines("Novo Nordisk A/S", "NVO", "Healthcare", HEADLINES)

    assert result == [HeadlineAssessment(relevant=True, importance=5), HeadlineAssessment(relevant=False, importance=1)]


def test_prompt_names_the_company_and_numbers_each_headline(configured, monkeypatch):
    calls = []
    parsed = [_Assessment(index=0, relevant=True, importance=3), _Assessment(index=1, relevant=True, importance=3)]
    monkeypatch.setattr(news_classifier.genai, "Client", _fake_client(parsed, calls=calls))

    classify_headlines("Novo Nordisk A/S", "NVO", "Healthcare", HEADLINES)

    assert "Novo Nordisk A/S (ticker NVO)" in calls[0]
    assert "0. Novo Sets Long-Term Targets - Reuters" in calls[0]
    assert "1. Novo Banco reports profit surge - Reuters" in calls[0]


def test_no_api_key_means_no_call(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(news_classifier.genai, "Client", _fake_client(error=AssertionError("must not be called")))

    assert classify_headlines("Novo Nordisk A/S", "NVO", None, HEADLINES) is None


def test_a_failed_call_returns_none(configured, monkeypatch):
    monkeypatch.setattr(news_classifier.genai, "Client", _fake_client(error=RuntimeError("429 RESOURCE_EXHAUSTED")))

    assert classify_headlines("Novo Nordisk A/S", "NVO", None, HEADLINES) is None


def test_a_response_missing_a_headline_returns_none(configured, monkeypatch):
    monkeypatch.setattr(news_classifier.genai, "Client", _fake_client([_Assessment(index=0, relevant=True, importance=4)]))

    assert classify_headlines("Novo Nordisk A/S", "NVO", None, HEADLINES) is None


def test_an_unparseable_response_returns_none(configured, monkeypatch):
    monkeypatch.setattr(news_classifier.genai, "Client", _fake_client(parsed=None))

    assert classify_headlines("Novo Nordisk A/S", "NVO", None, HEADLINES) is None
