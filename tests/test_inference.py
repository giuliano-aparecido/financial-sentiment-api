import asyncio
import json

import pytest

from app.services import inference

MARKET_DATA = "Price: $189.30 | Market Cap: $2.95T\nP/E (trailing): 31.2 | P/E (forward): 27.8\nEPS (trailing): $6.07 | Dividend Yield: 0.55%\n52-Week Range: $164.08 - $237.23"
VALUATION = "Intrinsic Value (EPS-based): $78.40\nvs Current Price: overvalued by ~59%"
EARNINGS = "Last Quarter (2026-06-30): Revenue $85.8B (+4.9% YoY), EPS $1.40 (beat est. $1.35)\nNext Earnings Date: 2026-10-29"
PRICE_CONTEXT = "AAPL moved -9.2% on the day this was published."
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


def _analysis_response(reasoning, answer, recommendation="BUY"):
    return _ok(json.dumps({"recommendation": recommendation, "reasoning": reasoning, "answer": answer}))


def _install_fake_client(monkeypatch, responses):
    fake = FakeClient(responses)
    monkeypatch.setattr(inference, "_client", fake)
    return fake


CONTEXT = inference.MarketContext(
    market_data=MARKET_DATA, valuation=VALUATION, earnings=EARNINGS, live_context=NEWS,
)


def _run_two_stage(gap_pct=None, ticker_was_explicit=True):
    return inference.analyze_two_stage(
        "AAPL", "Is AAPL a buy?", CONTEXT, PRICE_CONTEXT, gap_pct,
        ticker_was_explicit=ticker_was_explicit,
    )


# --- start_client (HF_API_TOKEN presence check) ---


def test_start_client_checks_hf_api_token_presence(monkeypatch):
    calls = []
    monkeypatch.setattr(inference, "require_hf_api_token", lambda: calls.append(1))
    _run(inference.start_client())
    try:
        assert calls == [1]
    finally:
        _run(inference.stop_client())


def test_start_client_propagates_missing_token_error(monkeypatch):
    # Regression: HF_API_TOKEN used to have no presence check at all -
    # _call_model would send `Authorization: Bearer None` and every
    # inference call would fail with the same generic 502 a real backend
    # outage produces. start_client() (run at app startup, see main.py's
    # lifespan) must now fail loudly and specifically instead.
    def _boom():
        raise RuntimeError("HF_TOKEN is not set - required for calling the Hugging Face inference backend.")

    monkeypatch.setattr(inference, "require_hf_api_token", _boom)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _run(inference.start_client())


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


def test_call_model_logs_the_prompt_and_the_raw_output_on_success(monkeypatch, caplog):
    import logging

    _install_fake_client(monkeypatch, [_ok('{"news_reaction": "good"}  <|eot_id|>')])
    with caplog.at_level(logging.INFO, logger="app.services.inference"):
        _run(inference.classify_news("AAPL", PRICE_CONTEXT, NEWS, model="llama"))

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Prompt sent to [llama/reaction] for ticker=AAPL:") for m in messages)
    assert 'Raw output from [llama/reaction] for ticker=AAPL:\n{"news_reaction": "good"}  <|eot_id|>' in messages


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


def test_task_b_prompt_contains_all_blocks_and_no_given_recommendation(monkeypatch):
    fake = _install_fake_client(monkeypatch, [_analysis_response("Reasoning text.", "Yes, looks like a buy.")])
    _run(inference.generate_analysis("AAPL", "Is AAPL a buy?", "overreaction_down", CONTEXT))

    prompt = fake.payloads[0]["inputs"]
    assert "Current Market Data:\n" + MARKET_DATA in prompt
    assert "Valuation:\n" + VALUATION in prompt
    assert "Recent Earnings:\n" + EARNINGS in prompt
    assert "Recent News & Results:\n" + NEWS in prompt
    assert "News Reaction: overreaction_down" in prompt
    # The whole point of this plan: no recommendation is handed to the
    # model - it must decide BUY/SELL/HOLD itself from the inputs above.
    assert "Recommended Action" not in prompt
    assert "Decide the recommended action" in prompt


def test_task_b_requests_512_max_new_tokens(monkeypatch):
    fake = _install_fake_client(monkeypatch, [_analysis_response("r", "a")])
    _run(inference.generate_analysis("AAPL", "", "good", CONTEXT))
    assert fake.payloads[0]["parameters"]["max_new_tokens"] == 512


def test_generate_analysis_returns_recommendation_reasoning_and_answer(monkeypatch):
    _install_fake_client(monkeypatch, [_analysis_response("Solid fundamentals.", "Yes, a reasonable buy.", recommendation="BUY")])
    result = _run(inference.generate_analysis("AAPL", "", "good", CONTEXT))
    assert result["recommendation"] == "BUY"
    assert result["reasoning"] == "Solid fundamentals."
    assert result["answer"] == "Yes, a reasonable buy."


def test_generate_analysis_falls_back_to_raw_response_on_unparseable_output(monkeypatch):
    _install_fake_client(monkeypatch, [_ok("not json at all, no braces here")])
    result = _run(inference.generate_analysis("AAPL", "", "good", CONTEXT))
    assert result == {"raw_response": "not json at all, no braces here"}


def test_generate_analysis_falls_back_to_raw_response_on_unrecognized_recommendation(monkeypatch):
    _install_fake_client(monkeypatch, [_analysis_response("r", "a", recommendation="STRONG_BUY")])  # not a valid class
    result = _run(inference.generate_analysis("AAPL", "", "good", CONTEXT))
    assert "raw_response" in result


# --- analyze_two_stage (full orchestration) ---


def test_two_stage_happy_path_returns_recommendation_reasoning_answer(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("overreaction_down"),
        _analysis_response("The drop looks overdone given the routine news.", "Yes, this looks like a buy.", recommendation="BUY"),
    ])
    result = _run(_run_two_stage(gap_pct=-40.0))

    assert result["news_reaction"] == "overreaction_down"
    assert result["recommendation"] == "BUY"  # Task B's own call
    assert result["reasoning"] == "The drop looks overdone given the routine news."
    assert result["answer"] == "Yes, this looks like a buy."
    assert result["valuation_gap_pct"] == -40.0
    assert result["market_data"] == MARKET_DATA
    assert result["valuation"] == VALUATION


def test_two_stage_recommendation_is_whatever_task_b_says(monkeypatch):
    # fuse() no longer runs at inference at all (see docs/task-b-learned-
    # recommendation-plan.md) - Task B's own recommendation is trusted
    # directly, even for a (news_reaction, gap_pct) pair fusion.FUSION_TABLE
    # would have resolved differently (overreaction_down + undervalued was
    # BUY there) - nothing here overrides or double-checks it anymore.
    _install_fake_client(monkeypatch, [
        _reaction_response("overreaction_down"),
        _analysis_response("r", "a", recommendation="SELL"),
    ])
    result = _run(_run_two_stage(gap_pct=-40.0))

    assert result["recommendation"] == "SELL"


def test_ticker_was_explicit_defaults_true_and_propagates(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("good"),
        _analysis_response("r", "a"),
    ])
    result = _run(_run_two_stage(gap_pct=None))
    assert result["ticker_was_explicit"] is True


def test_ticker_was_explicit_false_propagates_on_task_b_success(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("good"),
        _analysis_response("r", "a"),
    ])
    result = _run(_run_two_stage(gap_pct=None, ticker_was_explicit=False))
    assert result["ticker_was_explicit"] is False


def test_ticker_was_explicit_false_propagates_on_task_b_raw_response_fallback(monkeypatch):
    _install_fake_client(monkeypatch, [
        _reaction_response("good"),
        _ok("not json at all, no braces here"),  # Task B fails to parse
    ])
    result = _run(_run_two_stage(gap_pct=None, ticker_was_explicit=False))
    assert result["ticker_was_explicit"] is False
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
    # Task B still ran off the "neutral" fallback input and produced its
    # own recommendation - Task A failing doesn't lose it, since Task B
    # never depended on Task A's specific class, just SOME valid one.
    assert result["recommendation"] == "BUY"


def test_two_stage_task_b_failure_has_no_recommendation_and_uses_raw_response(monkeypatch):
    # Unlike the old fuse()-backed design, there's no deterministic
    # fallback recommendation anymore - if Task B's own output doesn't
    # parse, there simply isn't one to show for this request.
    _install_fake_client(monkeypatch, [
        _reaction_response("good"),
        _ok("not json at all, no braces here"),  # Task B fails to parse
    ])
    result = _run(_run_two_stage(gap_pct=None))

    assert result["news_reaction"] == "good"
    assert "recommendation" not in result
    assert "confidence" not in result
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


