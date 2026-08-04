import asyncio

from fastapi import APIRouter, Depends

from app.deps import verify_api_key
from app.models import QueryRequest
from app.services.inference import analyze_with_hf
from app.services.news import fetch_live_news_rag
from app.services.ticker import extract_ticker

router = APIRouter()


@router.post("/api/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_stock(req: QueryRequest):
    ticker = extract_ticker(req.user_query)
    # fetch_live_news_rag does a blocking HTTP fetch (feedparser.parse) - off
    # the event loop, so one slow Google News response doesn't stall every
    # other concurrent request on this single-worker process.
    live_context = await asyncio.to_thread(fetch_live_news_rag, ticker)
    return await analyze_with_hf(ticker, live_context)
