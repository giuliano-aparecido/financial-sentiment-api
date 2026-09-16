import json
import logging
import re
from dataclasses import dataclass

import httpx
from fastapi import HTTPException

from app.config import DEFAULT_HF_INFERENCE_URL, HF_API_TOKEN, MODEL_ARCHITECTURE, require_hf_api_token
from app.services.fusion import fuse
from app.services.parsing import extract_json_object

logger = logging.getLogger(__name__)

# Strips markdown heading markers and this prompt template's own section-
# marker words from user-supplied text before interpolating it into the
# prompt - confirmed live a user_query containing "### Response:" or a
# counterfeit "Valuation:" line can terminate the real prompt early or
# overwrite the model's view of the fetched data blocks, since user_query
# is interpolated upstream of all of them (market_data/valuation/earnings/
# live_context). Not a full prompt-injection defense (none exists for
# free-text LLM input) - just removes the cheapest, most direct way to
# fake a section boundary. Only Task B's prompt uses this (see
# _build_analysis_prompt) - Task A never interpolates user_query at all.
_QUERY_SANITIZE_HEADING_RE = re.compile(r"#{2,}")
_QUERY_SANITIZE_MARKER_RE = re.compile(r"(?im)^\s*(instruction|input|response)\s*:\s*$")


def _sanitize_user_query(user_query: str) -> str:
    sanitized = _QUERY_SANITIZE_HEADING_RE.sub("", user_query)
    sanitized = _QUERY_SANITIZE_MARKER_RE.sub("", sanitized)
    return sanitized.strip()

# Mutable at runtime via /api/update-inference-url so a Colab/ngrok tunnel can
# repoint this without a redeploy. Single-process, in-memory by design - this
# app runs one worker and doesn't need it to survive a restart.
_hf_inference_url = DEFAULT_HF_INFERENCE_URL

_client: httpx.AsyncClient | None = None


def get_hf_inference_url() -> str:
    return _hf_inference_url


def set_hf_inference_url(url: str) -> None:
    global _hf_inference_url
    _hf_inference_url = url


async def start_client() -> None:
    global _client
    # Fails app startup loudly and specifically if HF_TOKEN is missing,
    # rather than letting every /api/analyze request fail later with the
    # same generic 502 _call_model raises for a real backend outage - see
    # require_hf_api_token's own docstring.
    require_hf_api_token()
    # 45.0 -> 280.0: too short for a scale-to-zero backend (financial-
    # sentiment-model's modal/serve_model.py) - confirmed live, a cold
    # start alone (container boot + model load) measured ~120s, well past
    # the old 45s, and the actual max_new_tokens=512 generation adds more
    # on top. 280s stays under Modal's own 300s per-call cap so a genuine
    # hang there still surfaces as Modal's error rather than this client
    # giving up first with no detail. financial-sentiment-web's proxy
    # (AbortSignal/maxDuration) had to move together with this - a
    # shorter timeout anywhere upstream just relocates where the same
    # request dies.
    _client = httpx.AsyncClient(timeout=280.0)


async def stop_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _client_or_raise() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("HTTP client not started - app lifespan did not run")
    return _client


async def _call_model(prompt: str, max_new_tokens: int, *, ticker: str, task_label: str) -> str:
    """POSTs `prompt` to the configured HF inference endpoint and returns
    the raw generated text - the shared transport core both classify_news
    (Task A) and generate_analysis (Task B) build on. Raises
    HTTPException(502) on any transport failure, non-200 response, or
    unexpected response shape; a PARSE failure of otherwise-successful
    generated text is each caller's own concern, not this function's."""
    # Full prompt, not truncated - unlike the response-body logging below,
    # the whole point here is to let you verify exactly what data/formatting
    # reached the model (e.g. confirming market_data/valuation/earnings are
    # populated and not silently "Data unavailable."), so cutting it short
    # would defeat that. INFO (not DEBUG) so it shows up by default under
    # this app's logging.basicConfig(level=logging.INFO) - no config change
    # needed to see it in Render's log stream.
    logger.info("Prompt sent to [%s/%s] for ticker=%s:\n%s", MODEL_ARCHITECTURE, task_label, ticker, prompt)

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }

    client = _client_or_raise()
    try:
        response = await client.post(_hf_inference_url, headers=headers, json=payload)
    except httpx.RequestError as e:
        logger.warning("HF inference request failed for [%s/%s] at %s: %s", MODEL_ARCHITECTURE, task_label, _hf_inference_url, e)
        raise HTTPException(status_code=502, detail="Failed to reach the inference backend.")

    if response.status_code != 200:
        # response.text can carry HF account/model/quota details (or ngrok
        # internals) - log it server-side, don't hand it to the client.
        logger.warning("Inference error for [%s/%s]: %s %s", MODEL_ARCHITECTURE, task_label, response.status_code, response.text[:1000])
        raise HTTPException(status_code=502, detail="The inference backend returned an error.")

    try:
        res_data = response.json()
        return res_data[0]["generated_text"] if isinstance(res_data, list) else str(res_data)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.warning("Unexpected inference response shape for [%s/%s]: %s - body: %s", MODEL_ARCHITECTURE, task_label, e, response.text[:1000])
        raise HTTPException(status_code=502, detail="The inference backend returned an unexpected response.")


def _clean_model_output(raw_text: str) -> str:
    """Strips special tokens and markdown code fences a raw generation can
    carry, before handing the remainder to extract_json_object - same
    cleaning both tasks' outputs need, factored out rather than
    duplicated."""
    clean = re.sub(r"<\|.*?\|>", "", raw_text)  # Strips <|eot_id|> and other Llama tokens
    clean = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", clean)  # Strips ```json ... ``` code blocks
    return clean.strip()


_VALID_NEWS_REACTIONS = frozenset({"good", "bad", "neutral", "overreaction_down", "overreaction_up"})


def _build_reaction_prompt(ticker: str, price_context: str, live_context: str) -> str:
    # Canonical Task A template - must stay byte-identical to financial-
    # sentiment-model's colab/train/{gpu,tpu}/train_model.py and
    # evaluate_*.py copies, and runpod/{train_model,evaluate_model}.py
    # (see that repo's CONTRIBUTING.md sync rule). The model is trained on
    # exactly this shape; a drift here trains one prompt and serves
    # another. Deliberately narrow: no user_query, no market_data/
    # valuation/earnings - Task A only ever reasons about the news itself
    # plus how the stock has recently moved, never fundamentals or the
    # user's question (see the two-stage pipeline redesign - fusion.py,
    # not this prompt, is what combines a reaction with valuation).
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Classify how the market has reacted to the following news for this stock, given its recent price move, and output JSON containing your classification, in exactly this shape:
{{"news_reaction": "good|bad|neutral|overreaction_down|overreaction_up"}}

news_reaction definitions:
- good: the news is genuinely positive for the stock.
- bad: the news is genuinely negative for the stock.
- neutral: the news is routine/ambiguous, not a real catalyst either way.
- overreaction_down: the price fell more than this news alone would justify - a plausible overreaction to the downside.
- overreaction_up: the price rose more than this news alone would justify - a plausible overreaction to the upside.

### Input:

Target Stock: {ticker}
Recent Price Move: {price_context}

Recent News & Results:
{live_context}

### Response:

"""


@dataclass(frozen=True)
class MarketContext:
    """Bundles the fetched-context blocks threaded through the Task B
    pipeline (_build_analysis_prompt/generate_analysis/analyze_two_stage)
    - already-rendered text from market_data_block/valuation_assessment_
    for/earnings_block/fetch_live_news_rag, not fetched again here."""
    market_data: str
    valuation: str
    earnings: str
    live_context: str


def _build_analysis_prompt(
    ticker: str, user_query: str, news_reaction: str, recommendation: str, context: MarketContext,
) -> str:
    # Canonical Task B template - same sync requirement as
    # _build_reaction_prompt above. news_reaction/recommendation are
    # ALREADY-DECIDED inputs here (fuse()'s output, never guessed by this
    # call) - the instruction explicitly forbids advising the opposite of
    # Recommended Action, mirroring the training data's own framing.
    sanitized_user_query = _sanitize_user_query(user_query)
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

You are given a recommended action for this stock, already determined from valuation and news analysis - your job is to explain it, not decide it. Output JSON containing detailed reasoning and a direct answer to the user's question, in exactly this shape:
{{"reasoning": "...", "answer": "..."}}

Your reasoning and answer must be consistent with the Recommended Action below and must never advise the opposite. Treat News Reaction and Recommended Action as given facts, not conclusions to re-derive.

### Input:

Target Stock: {ticker}
User Question: {sanitized_user_query}
News Reaction: {news_reaction}
Recommended Action: {recommendation}

Current Market Data:
{context.market_data}

Valuation:
{context.valuation}

Recent Earnings:
{context.earnings}

Recent News & Results:
{context.live_context}

### Response:

"""


async def classify_news(ticker: str, price_context: str, live_context: str) -> str | None:
    """Task A: classifies how the market has reacted to `live_context`
    given `price_context` (see fundamentals.price_move_on_date). Returns
    the news_reaction string on success, or None if the model's output
    didn't parse as JSON or named a class outside _VALID_NEWS_REACTIONS -
    callers must normalize a None to a safe default (see analyze_two_
    stage) rather than letting it propagate, since fusion.fuse() raises
    on an unrecognized reaction by design (a real bug should never be
    silently masked by an accidental fallback recommendation)."""
    prompt = _build_reaction_prompt(ticker, price_context, live_context)
    raw_text = await _call_model(prompt, max_new_tokens=48, ticker=ticker, task_label="reaction")
    try:
        json_str = extract_json_object(_clean_model_output(raw_text))
        reaction = json.loads(json_str)["news_reaction"]
        if reaction not in _VALID_NEWS_REACTIONS:
            raise ValueError(f"unrecognized news_reaction: {reaction!r}")
        return reaction
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.info("Task A output for [%s] ticker=%s wasn't a valid news_reaction (%s) - raw: %r",
                    MODEL_ARCHITECTURE, ticker, e, raw_text[:300])
        return None


async def generate_analysis(
    ticker: str, user_query: str, news_reaction: str, recommendation: str, context: MarketContext,
) -> dict:
    """Task B: writes reasoning/answer given an ALREADY-DECIDED news_
    reaction + recommendation (never asked to produce either). Returns
    {"reasoning", "answer", "raw_json"} on success, or {"raw_response":
    ...} if the model's output didn't parse - the caller (analyze_two_
    stage) still has the recommendation/confidence/news_reaction
    regardless, since none of those ever depended on this call
    succeeding (a strict improvement over the old single-call design,
    where a parse failure lost the recommendation entirely)."""
    prompt = _build_analysis_prompt(ticker, user_query, news_reaction, recommendation, context)
    raw_text = await _call_model(prompt, max_new_tokens=512, ticker=ticker, task_label="analysis")
    try:
        json_str = extract_json_object(_clean_model_output(raw_text))
        parsed = json.loads(json_str)
        return {"reasoning": parsed["reasoning"], "answer": parsed["answer"], "raw_json": parsed}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.info("Task B output for [%s] ticker=%s wasn't the expected JSON shape (%s) - falling back to raw_response",
                    MODEL_ARCHITECTURE, ticker, e)
        return {"raw_response": raw_text}


async def analyze_two_stage(
    ticker: str,
    user_query: str,
    context: MarketContext,
    price_context: str,
    gap_pct: float | None,
    ticker_was_explicit: bool = True,
) -> dict:
    """Orchestrates the two-stage pipeline: Task A classifies news_
    reaction -> fusion.fuse() computes the ONLY recommendation this
    service ever produces -> Task B explains it, given that
    recommendation as input. Neither LLM call ever decides BUY/SELL/HOLD
    itself - see fusion.py's own module docstring for why. Replaces the
    old single-call analyze_with_hf. ticker_was_explicit just passes
    through to the result dict (see ticker.extract_ticker) - never
    affects Task A/B or fusion, purely informational for the frontend."""
    news_reaction = await classify_news(ticker, price_context, context.live_context)
    news_reaction_fallback = news_reaction is None
    if news_reaction_fallback:
        # Fusion still yields a valuation-driven recommendation off
        # "neutral" - conservative (never invents a directional read) and
        # keeps the response usable even when Task A's output was
        # unparseable (including the safety case of an old, pre-redesign
        # checkpoint still pointed at by this API, which would emit an
        # entirely different JSON shape here).
        news_reaction = "neutral"

    fusion_result = fuse(news_reaction, gap_pct)

    analysis = await generate_analysis(ticker, user_query, news_reaction, fusion_result.recommendation, context)

    result = {
        "model_architecture": MODEL_ARCHITECTURE,
        "ticker": ticker,
        "ticker_was_explicit": ticker_was_explicit,
        "live_news_retrieved": context.live_context,
        "recommendation": fusion_result.recommendation,
        "confidence": fusion_result.confidence,
        "news_reaction": news_reaction,
        "valuation_gap_pct": gap_pct,
        "market_data": context.market_data,
        "valuation": context.valuation,
        "earnings": context.earnings,
        "raw_json": {"task_a": {"news_reaction": news_reaction}, "task_b": analysis.get("raw_json")},
    }
    if news_reaction_fallback:
        result["news_reaction_fallback"] = True

    if "reasoning" in analysis:
        result["reasoning"] = analysis["reasoning"]
        result["answer"] = analysis["answer"]
    else:
        result["raw_response"] = analysis["raw_response"]

    return result
