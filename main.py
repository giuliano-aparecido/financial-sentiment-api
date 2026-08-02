import os

import re

import json

import httpx

import yfinance as yf

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

app = FastAPI(title="Multi-Model Financial RAG Reasoning Engine")

# ==============================================================================

# ALLOW CORS FOR VERCEL FRONTEND REQUESTS

# ==============================================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],  # Allows requests from Vercel deployment

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)

HF_API_TOKEN = os.getenv("HF_TOKEN")

MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "apertus").lower()

DEFAULT_MODELS = {

    "llama": "your-username/llama-3.2-3b-financial-reasoner",

    "apertus": "your-username/apertus-8b-financial-reasoner",

    "qwen": "your-username/qwen-2.5-7b-financial-reasoner",

    "mistral": "your-username/mistral-7b-financial-reasoner"

}

HF_MODEL_REPO = os.getenv("HF_MODEL_URL", DEFAULT_MODELS.get(MODEL_ARCHITECTURE, DEFAULT_MODELS["apertus"]))

HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"

class QueryRequest(BaseModel):

    user_query: str

def extract_ticker(text: str) -> str:

    match = re.search(r'\b[A-Z]{3,5}\b', text.upper())

    return match.group(0) if match else "AAPL"

def fetch_live_news_rag(ticker: str) -> str:

    try:

        stock = yf.Ticker(ticker)

        news_items = stock.news[:3]

        if not news_items:

            return f"No recent live news found for {ticker}."

        

        context_str = ""

        for item in news_items:

            title = item.get("title", "")

            summary = item.get("summary", "")

            context_str += f"- {title}: {summary}\n"

        return context_str

    except Exception:

        return f"Recent quarterly and news updates for {ticker}."

@app.get("/health")

def health_check():

    return {

        "status": "online",

        "active_architecture": MODEL_ARCHITECTURE,

        "target_model_repo": HF_MODEL_REPO

    }

@app.post("/api/analyze")

async def analyze_stock(req: QueryRequest):

    ticker = extract_ticker(req.user_query)

    live_context = fetch_live_news_rag(ticker)

    

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Analyze the following financial news and output JSON containing the impacted stock ticker, detailed reasoning, directional sentiment (BULLISH/BEARISH/NEUTRAL), and confidence score.

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

    async with httpx.AsyncClient(timeout=45.0) as client:

        response = await client.post(HF_INFERENCE_URL, headers=headers, json=payload)

        

    if response.status_code != 200:

        raise HTTPException(

            status_code=500, 

            detail=f"Inference error for [{MODEL_ARCHITECTURE}]: {response.text}"

        )

        

    res_data = response.json()

    raw_model_output = res_data[0]["generated_text"] if isinstance(res_data, list) else str(res_data)

    

    try:

        analysis_json = json.loads(raw_model_output)

        impact = analysis_json["impacted_stocks"][0]

        

        return {

            "model_architecture": MODEL_ARCHITECTURE,

            "ticker": ticker,

            "live_news_retrieved": live_context,

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
