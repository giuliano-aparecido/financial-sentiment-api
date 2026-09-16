import asyncio

from fastapi import APIRouter, Depends, Query

from app.deps import verify_api_key
from app.models import QueryRequest
from app.services.earnings import earnings_block, fetch_earnings
from app.services.fundamentals import fetch_fundamentals, market_data_block, price_move_on_date
from app.services.inference import MarketContext, analyze_two_stage
from app.services.news import fetch_live_news_rag
from app.services.ticker import extract_ticker
from app.services.valuation import valuation_assessment_for

router = APIRouter()


@router.post("/api/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_stock(req: QueryRequest, model: str = Query("llama", description="Model to use: 'llama' or 'apertus'")):
    ticker, ticker_was_explicit = extract_ticker(req.user_query)
    # fetch_fundamentals and fetch_earnings are independent of everything
    # else, so they still run concurrently. fetch_live_news_rag can't join
    # that gather anymore, though: its relevance filter needs a company
    # name/sector to check headlines against (see app.services.news.
    # _is_relevant_headline), and the cheapest source for those is
    # fundamentals' own .info fetch, not a second yfinance call - so news
    # now runs AFTER fundamentals resolves. price_move_on_date, in turn,
    # needs the SPECIFIC headline news.py selected (its published_date),
    # so it runs after that. This trades some of the old 4-way parallelism
    # for the correctness fix the redesign plan calls for (train and
    # inference measuring the same single-day, same-headline price move) -
    # see financial-sentiment-model's docs/two-stage-task-a-redesign-
    # plan.md. All four calls still do blocking I/O (feedparser/yfinance)
    # off the event loop via to_thread.
    fundamentals, earnings_data = await asyncio.gather(
        asyncio.to_thread(fetch_fundamentals, ticker),
        asyncio.to_thread(fetch_earnings, ticker),
    )
    # fundamentals may have resolved a bare extracted ticker to its real
    # Yahoo symbol internally (e.g. "NESN" -> "NESN.SW" - see
    # fundamentals.resolve_ticker) - use that resolved form for every
    # subsequent lookup too, so CURATED_SCENARIOS lookups, the news
    # search, the price lookup, and the audit log all line up with what
    # was actually fetched, not the raw extracted text.
    valuation_ticker = fundamentals.get("resolved_ticker", ticker) if fundamentals else ticker
    company_name = (fundamentals.get("company_name") or ticker) if fundamentals else ticker
    sector = fundamentals.get("sector") if fundamentals else None

    live_context, published_date = await asyncio.to_thread(fetch_live_news_rag, valuation_ticker, company_name, sector)
    price_context, _move_fraction = await asyncio.to_thread(price_move_on_date, valuation_ticker, published_date)

    # valuation_assessment_for's scenario-DCF math is pure/in-memory - no
    # to_thread needed, it just consumes fundamentals' already-fetched dict.
    market_data = market_data_block(fundamentals)
    valuation, gap_pct = valuation_assessment_for(fundamentals, ticker=valuation_ticker)
    earnings = earnings_block(earnings_data)
    context = MarketContext(market_data=market_data, valuation=valuation, earnings=earnings, live_context=live_context)
    return await analyze_two_stage(
        ticker, req.user_query, context, price_context, gap_pct,
        ticker_was_explicit=ticker_was_explicit,
        model=model,
    )
