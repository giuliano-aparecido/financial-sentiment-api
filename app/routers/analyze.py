import asyncio

from fastapi import APIRouter, Depends

from app.deps import verify_api_key
from app.models import QueryRequest
from app.services.earnings import earnings_block, fetch_earnings
from app.services.fundamentals import fetch_fundamentals, market_data_block, recent_price_move
from app.services.inference import analyze_two_stage
from app.services.news import fetch_live_news_rag
from app.services.ticker import extract_ticker
from app.services.valuation import valuation_assessment_for

router = APIRouter()


@router.post("/api/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_stock(req: QueryRequest):
    ticker = extract_ticker(req.user_query)
    # All four do blocking I/O (feedparser/yfinance) - off the event loop
    # via to_thread (same reason as before: one slow response shouldn't
    # stall every other concurrent request on this single-worker process),
    # and gathered concurrently so the total wait is the slowest of the
    # four, not their sum. recent_price_move is Task A's only price
    # signal (see inference.classify_news) - a separate yfinance call from
    # fetch_fundamentals since it needs trailing daily history, not a
    # single .info snapshot.
    live_context, fundamentals, earnings_data, (price_context, _move_fraction) = await asyncio.gather(
        asyncio.to_thread(fetch_live_news_rag, ticker),
        asyncio.to_thread(fetch_fundamentals, ticker),
        asyncio.to_thread(fetch_earnings, ticker),
        asyncio.to_thread(recent_price_move, ticker),
    )
    # valuation_assessment_for's scenario-DCF math is pure/in-memory - no
    # to_thread needed, it just consumes fundamentals' already-fetched dict.
    market_data = market_data_block(fundamentals)
    # fundamentals may have resolved a bare extracted ticker to its real
    # Yahoo symbol internally (e.g. "NESN" -> "NESN.SW" - see
    # fundamentals.resolve_ticker) - use that resolved form here too, so
    # CURATED_SCENARIOS lookups and the audit log line up with what was
    # actually fetched, not the raw extracted text.
    valuation_ticker = fundamentals.get("resolved_ticker", ticker) if fundamentals else ticker
    valuation, gap_pct = valuation_assessment_for(fundamentals, ticker=valuation_ticker)
    earnings = earnings_block(earnings_data)
    return await analyze_two_stage(
        ticker, req.user_query, live_context, market_data, valuation, earnings, price_context, gap_pct,
    )
