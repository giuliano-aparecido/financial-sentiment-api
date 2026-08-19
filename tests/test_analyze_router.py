from app.routers import analyze as analyze_router

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


def _patch_services(monkeypatch, fundamentals=None, earnings_data=None, price_move=("Data unavailable.", None)):
    monkeypatch.setattr(analyze_router, "fetch_live_news_rag", lambda ticker: f"news for {ticker}")
    monkeypatch.setattr(analyze_router, "fetch_fundamentals", lambda ticker: fundamentals)
    monkeypatch.setattr(analyze_router, "fetch_earnings", lambda ticker: earnings_data)
    monkeypatch.setattr(analyze_router, "recent_price_move", lambda ticker: price_move)

    captured = {}

    async def fake_analyze_two_stage(
        ticker, user_query, live_context, market_data, valuation, earnings, price_context, gap_pct,
        ticker_was_explicit=True,
    ):
        captured.update(
            ticker=ticker, user_query=user_query, live_context=live_context,
            market_data=market_data, valuation=valuation, earnings=earnings,
            price_context=price_context, gap_pct=gap_pct,
            ticker_was_explicit=ticker_was_explicit,
        )
        return {
            "recommendation": "HOLD", "news_reaction": "neutral",
            "market_data": market_data, "valuation": valuation, "earnings": earnings,
        }

    monkeypatch.setattr(analyze_router, "analyze_two_stage", fake_analyze_two_stage)
    return captured


def test_analyze_route_wires_fetched_blocks_into_inference_call(client, monkeypatch):
    fundamentals = {
        "price": 189.30, "market_cap": 2.95e12, "pe_trailing": 31.2, "pe_forward": 27.8,
        "eps_trailing": 6.07, "book_value_per_share": 20.0, "dividend_yield": 0.55,
        "year_low": 164.08, "year_high": 237.23,
        "free_cash_flow": None, "total_revenue": None, "dividend_rate": None,
        "sector": "Technology", "industry": "Consumer Electronics",
        "payout_ratio": 0.12,
    }
    earnings_data = {
        "last_quarter_date": "2026-06-30", "revenue": 85.8e9, "yoy_growth_pct": 4.9,
        "eps_actual": 1.40, "eps_estimate": 1.35, "next_earnings_date": "2026-10-29",
    }
    captured = _patch_services(
        monkeypatch, fundamentals=fundamentals, earnings_data=earnings_data,
        price_move=("AAPL moved -3.4% over the last 3 trading days.", -0.034),
    )

    response = client.post(
        "/api/analyze",
        json={"user_query": "Is $AAPL a buy?"},
        headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert captured["ticker"] == "AAPL"
    assert captured["live_context"] == "news for AAPL"
    assert "Price: $189.30" in captured["market_data"]
    assert "EPS-based" in captured["valuation"]
    assert "Revenue $85.8B" in captured["earnings"]
    assert captured["price_context"] == "AAPL moved -3.4% over the last 3 trading days."
    # A real, meaningful over/undervaluation gap should be a nonzero
    # number - the router must actually thread valuation_assessment_for's
    # second return value through to analyze_two_stage, not drop it.
    assert isinstance(captured["gap_pct"], float)


def test_analyze_route_degrades_gracefully_when_fundamentals_and_earnings_fail(client, monkeypatch):
    # fetch_fundamentals/fetch_earnings return None on a total fetch
    # failure (see their own docstrings) - the route must still succeed,
    # not 500, with the blocks rendering as "Data unavailable.".
    captured = _patch_services(monkeypatch, fundamentals=None, earnings_data=None)

    response = client.post(
        "/api/analyze",
        json={"user_query": "Is $AAPL a buy?"},
        headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert captured["market_data"] == "Data unavailable."
    assert captured["valuation"] == "Data unavailable."
    assert captured["earnings"] == "Data unavailable."
    assert captured["gap_pct"] is None


def test_analyze_route_flags_ticker_as_not_explicit_when_query_names_none(client, monkeypatch):
    captured = _patch_services(monkeypatch, fundamentals=None, earnings_data=None)

    response = client.post(
        "/api/analyze",
        json={"user_query": "is the market bullish today"},
        headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert captured["ticker"] == "AAPL"
    assert captured["ticker_was_explicit"] is False


def test_analyze_route_flags_ticker_as_explicit_when_query_names_one(client, monkeypatch):
    captured = _patch_services(monkeypatch, fundamentals=None, earnings_data=None)

    response = client.post(
        "/api/analyze",
        json={"user_query": "Is $AAPL a buy?"},
        headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert captured["ticker_was_explicit"] is True
