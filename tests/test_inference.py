import asyncio
import json

from app.services import inference

MARKET_DATA = "Price: $189.30 | Market Cap: $2.95T\nP/E (trailing): 31.2 | P/E (forward): 27.8\nEPS (trailing): $6.07 | Dividend Yield: 0.55%\n52-Week Range: $164.08 - $237.23"
VALUATION = "Intrinsic Value (EPS-based): $78.40\nvs Current Price: overvalued by ~59%"
EARNINGS = "Last Quarter (2026-06-30): Revenue $85.8B (+4.9% YoY), EPS $1.40 (beat est. $1.35)\nNext Earnings Date: 2026-10-29"
PRICE_CONTEXT = "AAPL moved -9.2% over the last 3 trading days."
NEWS = "- news headline"


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class FakeClient:
    """Responses are consumed in call order - analyze_two_stage always
    calls Task A (classify_news) before Task B (generate_analysis), so
    responses[0] answers Task A and responses[1] answers Task B."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []

    async def post(self, url, headers=None, json=None):
        self.payloads.append(json)
        return self._responses.pop(0)


def _run(coro):
    return asyncio.run(coro)


def _ok(generated_text):
    return FakeResponse(200, json_data=[{"generated_text": generated_text}])


def _reaction_response(reaction):
    return _ok(json.dumps({"news_reaction": reaction}))


def _analysis_response(reasoning, answer):
    return _ok(json.dumps({"reasoning": reasoning, "answer": answer}))


def _install_fake_client(monkeypatch, responses):
    fake = FakeClient(responses)
    monkeypatch.setattr(inference, "_client", fake)
    return fake


def _run_two_stage(gap_pct=None):
    return inference.analyze_two_stage(
        "AAPL", "Is AAPL a buy?", NEWS, MARKET_DATA, VALUATION, EARNINGS, PRICE_CONTEXT, gap_pct,
    )


# --- classify_news (Task A) ---


def test_task_a_prompt_contains_price_context_and_news_only(monkeypatch):
    fake = _install_fake_client(monkeypatch, [_reaction_response("good")])
    _run(inference.classify_news("AAPL", PRICE_CONTEXT, NEWS))

    prompt = fake.payloads[0]["inputs"]
    assert "Recent Price Move: " + PRICE_CONTEXT in prompt
    assert "Recent News & Results:\n" + NEWS in prompt
    # Task A never sees fundamentals, valuation, earnings, or the user's
    # own question - see _build_reaction_prompt's own comment for why.
    assert "Current Market Data" not in prompt
    assert "Valuation:" not in prompt
    assert "Recent Earnings" not in prompt
    assert "User Question" not in prompt


def test_task_a_requests_48_max_new_tokens(monkeypatch):
    fake = _install_fake_client(monkeypatch, [_reaction_response("good")])
    _run(inference.classify_news("AAPL", PRICE_CONTEXT, NEWS))
    assert fake.payloads[0]["parameters"]["max_new_tokens"] == 48


def test_classify_news_returns_valid_reaction(monkeypatch):
    _install_fake_client(monkeypatch, [_reaction_response("overreaction_down")])
    result = _run(inference.classify_news("AAPL", PRICE_CONTEXT, NEWS))
    assert result == "overreaction_down"


def test_classify_news_returns_none_on_unparseable_output(monkeypatch):
    _install_fake_client(monkeypatch, [_ok("not json at all, no braces here")])
    result = _run(inference.classify_news("AAPL", PRICE_CONTEXT, NEWS))
    assert result is None


def test_classify_news_returns_none_on_unrecognized_class(monkeypatch):
    _install_fake_client(monkeypatch, [_reaction_response("bullish")])  # old vocabulary, no longer valid
    result = _run(inference.classify_news("AAPL", PRICE_CONTEXT, NEWS))
    assert result is None


# --- generate_analysis (Task B) ---


def test_task_b_prompt_contains_all_blocks_and_given_recommendation(monkeypatch):
    fake = _install_fake_client(monkeypatch, [_analysis_response("Reasoning text.", "Yes, looks like a buy.")])
    _run(inference.generate_analysis(
        "AAPL", "Is AAPL a buy?", "overreaction_down", "BUY", MARKET_DATA, VALUATION, EARNINGS, NEWS,
    ))

    prompt = fake.payloads[0]["inputs"]
    assert "Current Market Data:\n" + MARKET_DATA in prompt
    assert "Valuation:\n" + VALUATION in prompt
    assert "Recent Earnings:\n" + EARNINGS in prompt
    assert "Recent News & Results:\n" + NEWS in prompt
    assert "News Reaction: overreaction_down" in prompt
    assert "Recommended Action: BUY" in prompt
    assert "must never advise the opposite" in prompt


def test_task_b_requests_512_max_new_tokens(monkeypatch):
    fake = _install_fake_client(monkeypatch, [_analysis_response("r", "a")])
    _run(inference.generate_analysis("AAPL", "", "good", "BUY", MARKET_DATA, VALUATION, EARNINGS, NEWS))
    assert fake.payloads[0]["parameters"]["max_new_tokens"] == 512


def test_generate_analysis_returns_reasoning_and_answer(monkeypatch):
    _install_fake_client(monkeypatch, [_analysis_response("Solid fundamentals.", "Yes, a reasonable buy.")])
    result = _run(inference.generate_analysis("AAPL", "", "good", "BUY", MARKET_DATA, VALUATION, EARNINGS, NEWS))
    assert result["reasoning"] == "Solid fundamentals."
    assert result["answer"] == "Yes, a reasonable buy."


def test_generate_analysis_falls_back_to_raw_response_on_unparseable_output(monkeypatch):
    _install_fake_client(monkeypatch, [_ok("not json at all, no braces here")])
    result = _run(inference.generate_analysis("AAPL", "", "good", "BUY", MARKET_DATA, VALUATION, EARNINGS, NEWS))
    assert result == {"raw_response": "not json at all, no braces here"}


# --- analyze_two_stage (full orchestration) ---


def test_two_stage_happy_path_returns_recommendation_confidence_reasoning_answer(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("overreaction_down"),
        _analysis_response("The drop looks overdone given the routine news.", "Yes, this looks like a buy."),
    ])
    result = _run(_run_two_stage(gap_pct=-40.0))

    assert result["news_reaction"] == "overreaction_down"
    assert result["recommendation"] == "BUY"  # overreaction_down + undervalued (gap<=-15) per fusion.FUSION_TABLE
    assert 0.0 < result["confidence"] <= 0.95
    assert result["reasoning"] == "The drop looks overdone given the routine news."
    assert result["answer"] == "Yes, this looks like a buy."
    assert result["valuation_gap_pct"] == -40.0
    assert result["market_data"] == MARKET_DATA
    assert result["valuation"] == VALUATION
    assert result["earnings"] == EARNINGS
    assert "news_reaction_fallback" not in result


def test_two_stage_task_a_failure_falls_back_to_neutral_and_flags_it(monkeypatch):
    _install_fake_client(monkeypatch, [
        _ok("not json at all, no braces here"),  # Task A fails to parse
        _analysis_response("Neutral read.", "No strong signal either way."),
    ])
    result = _run(_run_two_stage(gap_pct=-40.0))

    assert result["news_reaction"] == "neutral"
    assert result["news_reaction_fallback"] is True
    # fuse("neutral", -40.0) -> undervalued bucket -> BUY (valuation alone
    # drives the call when news itself is neutral) - the recommendation is
    # still real and deterministic, not lost just because Task A failed.
    assert result["recommendation"] == "BUY"


def test_two_stage_task_b_failure_keeps_recommendation_and_uses_raw_response(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("good"),
        _ok("not json at all, no braces here"),  # Task B fails to parse
    ])
    result = _run(_run_two_stage(gap_pct=None))

    assert result["news_reaction"] == "good"
    assert result["recommendation"] == "HOLD"  # good + no_data gap per fusion.FUSION_TABLE
    assert "confidence" in result
    assert result["raw_response"] == "not json at all, no braces here"
    assert "reasoning" not in result
    assert "answer" not in result


def test_two_stage_result_includes_valuation_gap_pct(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("neutral"),
        _analysis_response("r", "a"),
    ])
    result = _run(_run_two_stage(gap_pct=12.5))
    assert result["valuation_gap_pct"] == 12.5


def test_two_stage_no_gap_data_still_returns_a_recommendation(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("bad"),
        _analysis_response("r", "a"),
    ])
    result = _run(_run_two_stage(gap_pct=None))
    assert result["valuation_gap_pct"] is None
    assert result["recommendation"] == "SELL"  # bad + no_data per fusion.FUSION_TABLE
