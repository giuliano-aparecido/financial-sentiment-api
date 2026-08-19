import json
import logging
import re

import httpx
from fastapi import HTTPException

from app.config import DEFAULT_HF_INFERENCE_URL, HF_API_TOKEN, MODEL_ARCHITECTURE
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
# fake a section boundary.
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


async def analyze_with_hf(
    ticker: str,
    user_query: str,
    live_context: str,
    market_data: str,
    valuation: str,
    earnings: str,
    ticker_was_explicit: bool = True,
) -> dict:
    # Canonical prompt template - must stay byte-identical to
    # financial-sentiment-model's colab/train/gpu/tpu train_model.py and
    # evaluate_*.py copies (see that repo's CONTRIBUTING.md 4-way sync
    # rule). The model is trained on exactly this shape; a drift here
    # trains one prompt and serves another.
    #
    # Rules 2-6 below are new (audit-driven prompt fix, bundled with a
    # retrain so the new instructions and the model's training data agree
    # - a prompt-only change against the OLD checkpoint would just be
    # unverified drift). Confirmed live gaps this closes: rule "1." used
    # to be the ONLY critical rule, and it was about news specifically -
    # nothing told the model how to weigh a large Valuation-block gap
    # against a news headline, so it read as decorative (P1). NEUTRAL and
    # confidence were both used in the instruction with no definition at
    # all (P2). The output JSON schema the parser actually requires
    # (impacted_stocks/direction/...) never appeared anywhere in the
    # prompt, living only in the fine-tune weights - any checkpoint swap
    # silently degraded to the raw_response fallback (P3). Rule 6 puts
    # value-investing-checklist.md's graded P/E bands and the REIT
    # Price/Book override into the prompt itself, not just an external
    # doc the model never sees (P7, lightweight version - the sector name
    # now always renders in market_data's Sector Median P/E line, even
    # when no median exists for that sector, specifically so a Real
    # Estate-sector company is identifiable without a new parameter).
    #
    # Deliberately NOT adding an "As of: <date>" line here (a separate
    # audit finding, P5) - unlike the rules above, that needs a NEW
    # per-row training column (neither dataset generator currently has an
    # as_of_date field), so doing it here alone would silently mismatch
    # inference's prompt shape against every training example's - the
    # exact class of bug this repo has hit before (see git history:
    # fix/training-prompt-whitespace-mismatch). Left for a follow-up that
    # threads the column through both generators AND all 8 training/eval
    # template copies together, not shipped half-done under time pressure.
    sanitized_user_query = _sanitize_user_query(user_query)
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Analyze the following financial data and news and output JSON containing the impacted stock ticker, detailed reasoning, directional sentiment (BULLISH/BEARISH/NEUTRAL), confidence score, and a direct answer to the user's question, in exactly this shape:
{{"impacted_stocks": [{{"ticker": "...", "reasoning": "...", "direction": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "answer": "..."}}]}}

CRITICAL SENTIMENT RULES:

1. Weigh guidance cuts and revenue misses higher than minor operational wins.
2. The Valuation block is a real, structural signal, not decoration - a large over/undervaluation gap should meaningfully shape your direction and confidence, not just recent news. Only let concrete, current news override it when the news describes a specific catalyst (an actual event, not a generic "market volatility" statement) the valuation estimate couldn't have priced in.
3. NEUTRAL means the available signals genuinely conflict or are too weak/routine to support a directional call - not a default for "I'm not sure." Use it when Valuation, Market Data, Earnings, and News don't converge on one direction, or when nothing in the input is materially new.
4. confidence is a 0.0-1.0 score for how strongly the evidence supports your direction, not how certain you are a direction exists at all - a NEUTRAL call can still carry moderate confidence when "no clear signal" is itself well-supported.
5. "Data unavailable." or "Not applicable (...)" in any block means exactly that - treat it as missing information, never invent numbers or events to fill the gap.
6. P/E under 20 (sector-adjusted via Sector Median P/E) suggests undervaluation; 20-30 is roughly neutral; over 30 suggests a richer valuation that needs a real growth story to justify. For a Real Estate-sector company specifically, Price/Book below 1.0 is the more meaningful signal - GAAP depreciation makes P/E unreliable for that sector.

### Input:

Target Stock: {ticker}
User Question: {sanitized_user_query}

Current Market Data:
{market_data}

Valuation:
{valuation}

Recent Earnings:
{earnings}

Recent News & Results:
{live_context}

### Response:

"""

    # Full prompt, not truncated - unlike the response-body logging below,
    # the whole point here is to let you verify exactly what data/formatting
    # reached the model (e.g. confirming market_data/valuation/earnings are
    # populated and not silently "Data unavailable."), so cutting it short
    # would defeat that. INFO (not DEBUG) so it shows up by default under
    # this app's logging.basicConfig(level=logging.INFO) - no config change
    # needed to see it in Render's log stream.
    logger.info("Prompt sent to [%s] for ticker=%s:\n%s", MODEL_ARCHITECTURE, ticker, prompt)

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            # 350 -> 512: the v4 `answer` field adds length beyond what the
            # old 4-field JSON output needed.
            "max_new_tokens": 512,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }

    client = _client_or_raise()
    try:
        response = await client.post(_hf_inference_url, headers=headers, json=payload)
    except httpx.RequestError as e:
        logger.warning("HF inference request failed for [%s] at %s: %s", MODEL_ARCHITECTURE, _hf_inference_url, e)
        raise HTTPException(status_code=502, detail="Failed to reach the inference backend.")

    if response.status_code != 200:
        # response.text can carry HF account/model/quota details (or ngrok
        # internals) - log it server-side, don't hand it to the client.
        logger.warning("Inference error for [%s]: %s %s", MODEL_ARCHITECTURE, response.status_code, response.text[:1000])
        raise HTTPException(status_code=502, detail="The inference backend returned an error.")

    try:
        res_data = response.json()
        raw_model_output = res_data[0]["generated_text"] if isinstance(res_data, list) else str(res_data)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.warning("Unexpected inference response shape for [%s]: %s - body: %s", MODEL_ARCHITECTURE, e, response.text[:1000])
        raise HTTPException(status_code=502, detail="The inference backend returned an unexpected response.")

    try:
        # Clean special tokens, markdown code fences, and whitespace
        clean_output = re.sub(r"<\|.*?\|>", "", raw_model_output)  # Strips <|eot_id|> and other Llama tokens
        clean_output = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", clean_output)  # Strips ```json ... ``` code blocks
        clean_output = clean_output.strip()

        # Extract only the valid JSON substring if there's surrounding text
        json_str = extract_json_object(clean_output)

        analysis_json = json.loads(json_str)
        impact = analysis_json["impacted_stocks"][0]

        result = {
            "model_architecture": MODEL_ARCHITECTURE,
            "ticker": ticker,
            "ticker_was_explicit": ticker_was_explicit,
            "live_news_retrieved": live_context,
            "reasoning": impact["reasoning"],
            "predicted_direction": impact["direction"],
            "confidence": impact["confidence"],
            "raw_json": analysis_json,
            "market_data": market_data,
            "valuation": valuation,
            "earnings": earnings,
        }
        # .get, not impact["answer"] - a still-served older model (pre-v4)
        # won't have this key at all, and that must degrade to an omitted
        # field, not a 500 on an otherwise-successful analysis.
        answer = impact.get("answer")
        if answer is not None:
            result["answer"] = answer
        return result
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.info("Model output for [%s] wasn't the expected JSON shape (%s) - falling back to raw_response", MODEL_ARCHITECTURE, e)
        return {
            "model_architecture": MODEL_ARCHITECTURE,
            "ticker": ticker,
            "ticker_was_explicit": ticker_was_explicit,
            "live_news_retrieved": live_context,
            "raw_response": raw_model_output,
            # Still attached even though the model's own output didn't
            # parse - these were fetched independently by the router and
            # remain valid regardless of what the model returned, so the
            # UI can still show data cards alongside the raw fallback text.
            "market_data": market_data,
            "valuation": valuation,
            "earnings": earnings,
        }
