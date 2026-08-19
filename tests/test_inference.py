import asyncio
import json

from app.services import inference

MARKET_DATA = "Price: $189.30 | Market Cap: $2.95T\nP/E (trailing): 31.2 | P/E (forward): 27.8\nEPS (trailing): $6.07 | Dividend Yield: 0.55%\n52-Week Range: $164.08 - $237.23"
VALUATION = "Intrinsic Value (Graham Number): $78.40\nvs Current Price: overvalued by ~59%"
EARNINGS = "Last Quarter (2026-06-30): Revenue $85.8B (+4.9% YoY), EPS $1.40 (beat est. $1.35)\nNext Earnings Date: 2026-10-29"

V4_MODEL_OUTPUT = json.dumps({
    "impacted_stocks": [
        {
            "ticker": "AAPL",
            "reasoning": "Strong iPhone demand.",
            "direction": "BULLISH",
            "confidence": 0.87,
            "answer": "Yes, looks like a buy.",
        }
    ]
})

V3_MODEL_OUTPUT = json.dumps({
    "impacted_stocks": [
        {
            "ticker": "AAPL",
            "reasoning": "Strong iPhone demand.",
            "direction": "BULLISH",
            "confidence": 0.87,
        }
    ]
})


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.last_payload = None

    async def post(self, url, headers=None, json=None):
        self.last_payload = json
        return self._response


def _run(coro):
    return asyncio.run(coro)


def _install_fake_client(monkeypatch, generated_text, status_code=200):
    response = FakeResponse(status_code, json_data=[{"generated_text": generated_text}])
    fake = FakeClient(response)
    monkeypatch.setattr(inference, "_client", fake)
    return fake


def test_prompt_includes_all_v4_blocks(monkeypatch):
    fake = _install_fake_client(monkeypatch, V4_MODEL_OUTPUT)
    _run(inference.analyze_with_hf("AAPL", "Is AAPL a buy?", "- news headline", MARKET_DATA, VALUATION, EARNINGS))

    prompt = fake.last_payload["inputs"]
    assert "Current Market Data:\n" + MARKET_DATA in prompt
    assert "Valuation:\n" + VALUATION in prompt
    assert "Recent Earnings:\n" + EARNINGS in prompt
    assert "Recent News & Results:\n- news headline" in prompt
    assert "a direct answer to the user's question" in prompt


def test_payload_requests_512_max_new_tokens(monkeypatch):
    fake = _install_fake_client(monkeypatch, V4_MODEL_OUTPUT)
    _run(inference.analyze_with_hf("AAPL", "", "news", MARKET_DATA, VALUATION, EARNINGS))
    assert fake.last_payload["parameters"]["max_new_tokens"] == 512


def test_successful_v4_response_includes_answer_and_data_blocks(monkeypatch):
    _install_fake_client(monkeypatch, V4_MODEL_OUTPUT)
    result = _run(inference.analyze_with_hf("AAPL", "Is AAPL a buy?", "news", MARKET_DATA, VALUATION, EARNINGS))

    assert result["predicted_direction"] == "BULLISH"
    assert result["answer"] == "Yes, looks like a buy."
    assert result["market_data"] == MARKET_DATA
    assert result["valuation"] == VALUATION
    assert result["earnings"] == EARNINGS


def test_v3_model_output_omits_answer_without_erroring(monkeypatch):
    # A still-served older (pre-v4) model's JSON has no "answer" key at
    # all - this must degrade to an omitted field, not a 500 on an
    # otherwise-successful analysis.
    _install_fake_client(monkeypatch, V3_MODEL_OUTPUT)
    result = _run(inference.analyze_with_hf("AAPL", "Is AAPL a buy?", "news", MARKET_DATA, VALUATION, EARNINGS))

    assert result["predicted_direction"] == "BULLISH"
    assert "answer" not in result
    # Data blocks are independent of the model's own output shape - still present.
    assert result["market_data"] == MARKET_DATA


def test_unparseable_model_output_falls_back_but_keeps_data_blocks(monkeypatch):
    _install_fake_client(monkeypatch, "not json at all, no braces here")
    result = _run(inference.analyze_with_hf("AAPL", "Is AAPL a buy?", "news", MARKET_DATA, VALUATION, EARNINGS))

    assert result["raw_response"] == "not json at all, no braces here"
    assert "predicted_direction" not in result
    assert result["market_data"] == MARKET_DATA
    assert result["valuation"] == VALUATION


def test_ticker_was_explicit_defaults_true_and_propagates_on_success(monkeypatch):
    _install_fake_client(monkeypatch, V4_MODEL_OUTPUT)
    result = _run(inference.analyze_with_hf("AAPL", "Is AAPL a buy?", "news", MARKET_DATA, VALUATION, EARNINGS))
    assert result["ticker_was_explicit"] is True


def test_ticker_was_explicit_false_propagates_on_success(monkeypatch):
    _install_fake_client(monkeypatch, V4_MODEL_OUTPUT)
    result = _run(inference.analyze_with_hf(
        "AAPL", "is the market bullish today", "news", MARKET_DATA, VALUATION, EARNINGS,
        ticker_was_explicit=False,
    ))
    assert result["ticker_was_explicit"] is False


def test_ticker_was_explicit_false_propagates_on_raw_response_fallback(monkeypatch):
    _install_fake_client(monkeypatch, "not json at all, no braces here")
    result = _run(inference.analyze_with_hf(
        "AAPL", "is the market bullish today", "news", MARKET_DATA, VALUATION, EARNINGS,
        ticker_was_explicit=False,
    ))
    assert result["ticker_was_explicit"] is False
    assert result["earnings"] == EARNINGS
