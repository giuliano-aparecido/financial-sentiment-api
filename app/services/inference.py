import json
import logging
import re

import httpx
from fastapi import HTTPException

from app.config import DEFAULT_HF_INFERENCE_URL, HF_API_TOKEN, MODEL_ARCHITECTURE
from app.services.parsing import extract_json_object

logger = logging.getLogger(__name__)

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
    _client = httpx.AsyncClient(timeout=45.0)


async def stop_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _client_or_raise() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("HTTP client not started - app lifespan did not run")
    return _client


async def analyze_with_hf(ticker: str, live_context: str) -> dict:
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the following financial news and output JSON containing the impacted stock ticker, detailed reasoning, directional sentiment (BULLISH/BEARISH/NEUTRAL), and confidence score.

CRITICAL SENTIMENT RULES:
1. If the company lowered its full-year guidance, issued an earnings warning, or suffered a major price drop due to a revenue miss, the overall sentiment MUST be BEARISH regardless of short-term bounces.
2. Weigh guidance cuts and revenue misses higher than minor operational wins.

### Input:
Target Stock: {ticker}
Recent News & Results:
{live_context}

### Response:
"""

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 350,
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

        return {
            "model_architecture": MODEL_ARCHITECTURE,
            "ticker": ticker,
            "live_news_retrieved": live_context,
            "reasoning": impact["reasoning"],
            "predicted_direction": impact["direction"],
            "confidence": impact["confidence"],
            "raw_json": analysis_json,
        }
    except Exception:
        return {
            "model_architecture": MODEL_ARCHITECTURE,
            "ticker": ticker,
            "live_news_retrieved": live_context,
            "raw_response": raw_model_output,
        }
