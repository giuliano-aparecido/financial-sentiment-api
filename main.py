import asyncio

import os

import re

import json

import httpx

import urllib.parse
import feedparser

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler

from slowapi.errors import RateLimitExceeded

from slowapi.middleware import SlowAPIMiddleware

from slowapi.util import get_remote_address

app = FastAPI(title="Multi-Model Financial RAG Reasoning Engine")

# Applies to every route via default_limits, no per-route decorators needed.
# Keyed by client IP - see the Dockerfile's --proxy-headers flag, without
# which every request behind Render's proxy would share one IP and thus one
# bucket. The real cost here is per-request HF inference + yfinance calls,
# so the limit is deliberately tight - this endpoint is not meant for bursts.
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Added before CORSMiddleware so CORS ends up as the outer layer (Starlette
# wraps middleware in reverse order of addition) - otherwise a 429 response
# would be missing CORS headers and the browser would see an opaque network
# error instead of a readable 429.
app.add_middleware(SlowAPIMiddleware)

# ==============================================================================

# ALLOW CORS FOR VERCEL FRONTEND REQUESTS

# ==============================================================================

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app.add_middleware(

    CORSMiddleware,

    allow_origins=[FRONTEND_ORIGIN],

    allow_credentials=False,

    allow_methods=["POST", "GET"],

    allow_headers=["Content-Type", "X-API-Key"],

)

API_KEY = os.getenv("API_KEY")


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):

    if not API_KEY or x_api_key != API_KEY:

        raise HTTPException(status_code=401, detail="Invalid or missing API key")


HF_API_TOKEN = os.getenv("HF_TOKEN")

MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "apertus").lower()

DEFAULT_MODELS = {

    "llama": "gaparecido/llama-3.2-3b-financial-reasoner",

    "apertus": "gaparecido/apertus-8b-financial-reasoner",

    "qwen": "gaparecido/qwen-2.5-7b-financial-reasoner",

    "mistral": "gaparecido/mistral-7b-financial-reasoner"

}

HF_MODEL_REPO = os.getenv("HF_MODEL_URL", DEFAULT_MODELS.get(MODEL_ARCHITECTURE, DEFAULT_MODELS["apertus"]))

HF_INFERENCE_URL = os.getenv("HF_INFERENCE_URL") or f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"

class QueryRequest(BaseModel):

    user_query: str

class UpdateInferenceURLRequest(BaseModel):

    url: str

TICKER_STOPWORDS = {
    "I", "A", "AI", "CEO", "CFO", "IPO", "USA", "ETF", "NEWS", "WILL",
    "THE", "AND", "FOR", "ARE", "YOU", "ITS", "GDP", "SEC", "FED",
}

def extract_ticker(text: str) -> str:
    # Match against the original casing, not text.upper() - uppercasing first
    # made every 3-5 letter word in the query a "ticker" (e.g. the UI's own
    # placeholder "Will AAPL go up..." matched "WILL" before AAPL).
    for candidate in re.findall(r'\b[A-Z]{2,5}\b', text):
        if candidate not in TICKER_STOPWORDS:
            return candidate

    return "AAPL"

def extract_json_object(text: str) -> str:
    # Balances braces (ignoring ones inside string literals) instead of a
    # greedy regex, so trailing commentary from the model with its own
    # stray braces can't extend the match past the real JSON object.
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]

def fetch_live_news_rag(ticker: str) -> str:
    try:
        # Construct clean query for Google News RSS
        query = f"{ticker} stock earnings financial news"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        # Parse XML feed directly (No API key, No IP rate limits)
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return f"Recent market volatility and financial developments for {ticker}."
            
        # Extract top 4 news headlines with publication dates
        news_items = []
        for entry in feed.entries[:4]:
            title = entry.get("title", "")
            published = entry.get("published", "")[:16]  # Date string snippet
            news_items.append(f"- [{published}] {title}")
            
        return "\n".join(news_items)
        
    except Exception as e:
        print(f"RAG Google News RSS Error: {e}")
        return f"Recent quarterly earnings and news updates for {ticker}."

@app.get("/health")

def health_check():

    return {

        "status": "online",

        "active_architecture": MODEL_ARCHITECTURE,

        "target_model_repo": HF_MODEL_REPO

    }

ALLOWED_INFERENCE_HOST_SUFFIXES = tuple(
    suffix.strip()
    for suffix in os.getenv(
        "ALLOWED_INFERENCE_HOST_SUFFIXES",
        "ngrok-free.app,ngrok.io,ngrok.app,huggingface.cloud,huggingface.co",
    ).split(",")
    if suffix.strip()
)

@app.post("/api/update-inference-url", dependencies=[Depends(verify_api_key)])
async def update_inference_url(req: UpdateInferenceURLRequest, request: Request):
    global HF_INFERENCE_URL

    caller_ip = get_remote_address(request)
    new_url = req.url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(new_url)

    # Anyone holding the API key could otherwise repoint this at an internal
    # address or cloud metadata endpoint - analyze_stock then POSTs the HF
    # bearer token there on every request, so this isn't just a bad-config
    # risk, it's a token-exfiltration one.
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host.endswith(ALLOWED_INFERENCE_HOST_SUFFIXES):
        print(f"update-inference-url REJECTED from {caller_ip}: {new_url!r} (disallowed scheme/host)")
        raise HTTPException(
            status_code=400,
            detail="url must be https and match an allowed host (see ALLOWED_INFERENCE_HOST_SUFFIXES)",
        )

    old_url = HF_INFERENCE_URL
    HF_INFERENCE_URL = new_url
    print(f"update-inference-url OK from {caller_ip}: {old_url!r} -> {new_url!r}")
    return {"status": "ok", "hf_inference_url": HF_INFERENCE_URL}

@app.post("/api/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_stock(req: QueryRequest):
    ticker = extract_ticker(req.user_query)
    # fetch_live_news_rag does a blocking HTTP fetch (feedparser.parse) - off
    # the event loop, so one slow Google News response doesn't stall every
    # other concurrent request on this single-worker process.
    live_context = await asyncio.to_thread(fetch_live_news_rag, ticker)
    
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
            "return_full_text": False
        }
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(HF_INFERENCE_URL, headers=headers, json=payload)
    except httpx.RequestError as e:
        print(f"HF inference request failed for [{MODEL_ARCHITECTURE}] at {HF_INFERENCE_URL}: {e}")
        raise HTTPException(status_code=502, detail="Failed to reach the inference backend.")

    if response.status_code != 200:
        # response.text can carry HF account/model/quota details (or ngrok
        # internals) - log it server-side, don't hand it to the client.
        print(f"Inference error for [{MODEL_ARCHITECTURE}]: {response.status_code} {response.text[:1000]}")
        raise HTTPException(status_code=502, detail="The inference backend returned an error.")

    try:
        res_data = response.json()
        raw_model_output = res_data[0]["generated_text"] if isinstance(res_data, list) else str(res_data)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        print(f"Unexpected inference response shape for [{MODEL_ARCHITECTURE}]: {e} - body: {response.text[:1000]}")
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
            "live_news_retrieved": live_context,  # <-- Ensure this variable is non-empty
            "reasoning": impact["reasoning"],
            "predicted_direction": impact["direction"],
            "confidence": impact["confidence"],
            "raw_json": analysis_json
        }
    except Exception:
        return {
            "model_architecture": MODEL_ARCHITECTURE,
            "ticker": ticker,
            "live_news_retrieved": live_context,
            "raw_response": raw_model_output
        }
