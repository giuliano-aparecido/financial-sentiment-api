from app.routers import analyze as analyze_router

VALID_KEY = "test-api-key"  # matches conftest.py's API_KEY env var


def _patch_services(monkeypatch, fundamentals=None, earnings_data=None, price_move=("Data unavailable.", None)):
    monkeypatch.setattr(analyze_router, "fetch_live_news_rag", lambda ticker, name, sector: (f"news for {ticker}", None))
    monkeypatch.setattr(analyze_router, "fetch_fundamentals", lambda ticker: fundamentals)
    monkeypatch.setattr(analyze_router, "fetch_earnings", lambda ticker: earnings_data)
    monkeypatch.setattr(analyze_router, "price_move_on_date", lambda ticker, published_date: price_move)

    captured = {}

    async def fake_analyze_two_stage(
        ticker, user_query, context, price_context, gap_pct,
        ticker_was_explicit=True, model="llama",
    ):
        captured.update(
            ticker=ticker, user_query=user_query, live_context=context.live_context,
            market_data=context.market_data, valuation=context.valuation, earnings=context.earnings,
            price_context=price_context, gap_pct=gap_pct,
            ticker_was_explicit=ticker_was_explicit, model=model,
        )
        return {
            "recommendation": "HOLD", "news_reaction": "neutral",
            "market_data": context.market_data, "valuation": context.valuation, "earnings": context.earnings,
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
        "payout_ratio": 0.12, "company_name": "Apple Inc.",
    }
    earnings_data = {
        "last_quarter_date": "2026-06-30", "revenue": 85.8e9, "yoy_growth_pct": 4.9,
        "eps_actual": 1.40, "eps_estimate": 1.35, "next_earnings_date": "2026-10-29",
    }
    captured = _patch_services(
        monkeypatch, fundamentals=fundamentals, earnings_data=earnings_data,
        price_move=("AAPL moved -1.4% on the day this was published.", -0.014),
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
    assert captured["price_context"] == "AAPL moved -1.4% on the day this was published."
    # A real, meaningful over/undervaluation gap should be a nonzero
    # number - the router must actually thread valuation_assessment_for's
    # second return value through to analyze_two_stage, not drop it.
    assert isinstance(captured["gap_pct"], float)


def test_analyze_route_passes_resolved_ticker_name_and_sector_to_news_fetch(client, monkeypatch):
    fundamentals = {
        "price": 92.5, "resolved_ticker": "NESN.SW", "sector": "Consumer Defensive",
        "company_name": "Nestle SA",
    }
    news_calls = []

    def fake_fetch_live_news_rag(ticker, name, sector):
        news_calls.append((ticker, name, sector))
        return f"news for {ticker}", None

    monkeypatch.setattr(analyze_router, "fetch_live_news_rag", fake_fetch_live_news_rag)
    monkeypatch.setattr(analyze_router, "fetch_fundamentals", lambda ticker: fundamentals)
    monkeypatch.setattr(analyze_router, "fetch_earnings", lambda ticker: None)
    monkeypatch.setattr(analyze_router, "price_move_on_date", lambda ticker, published_date: ("Data unavailable.", None))

    async def fake_analyze_two_stage(*args, **kwargs):
        return {"recommendation": "HOLD", "news_reaction": "neutral"}

    monkeypatch.setattr(analyze_router, "analyze_two_stage", fake_analyze_two_stage)

    response = client.post(
        "/api/analyze",
        json={"user_query": "Is $NESN a buy?"},
        headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    # Uses the RESOLVED ticker (NESN.SW), not the raw extracted one
    # (NESN), and threads the company name/sector through so news.py's
    # relevance filter has something to check headlines against.
    assert news_calls == [("NESN.SW", "Nestle SA", "Consumer Defensive")]


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


# --- ?model= selection ---


def test_analyze_route_defaults_to_llama_when_no_model_given(client, monkeypatch):
    captured = _patch_services(monkeypatch, fundamentals=None, earnings_data=None)

    response = client.post("/api/analyze", json={"user_query": "Is $AAPL a buy?"}, headers={"X-API-Key": VALID_KEY})

    assert response.status_code == 200
    assert captured["model"] == "llama"


def test_analyze_route_forwards_model_query_param_normalized(client, monkeypatch):
    from app.services import inference

    monkeypatch.setattr(inference, "_inference_urls", {"llama": "https://l.modal.run", "apertus": "https://a.modal.run"})
    captured = _patch_services(monkeypatch, fundamentals=None, earnings_data=None)

    response = client.post(
        "/api/analyze?model=Apertus", json={"user_query": "Is $AAPL a buy?"}, headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert captured["model"] == "apertus"


def test_analyze_route_rejects_unknown_model_before_fetching_anything(client, monkeypatch):
    fetches = []
    monkeypatch.setattr(analyze_router, "fetch_fundamentals", lambda ticker: fetches.append(ticker))

    response = client.post(
        "/api/analyze?model=gpt9", json={"user_query": "Is $AAPL a buy?"}, headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 400
    assert "gpt9" in response.json()["detail"]
    assert fetches == []


def test_analyze_route_accepts_any_model_registered_at_runtime(client, monkeypatch):
    from app.services import inference

    monkeypatch.setattr(inference, "_inference_urls", {"llama": "https://old.modal.run", "kim": "https://kim.modal.run"})
    captured = _patch_services(monkeypatch, fundamentals=None, earnings_data=None)

    response = client.post(
        "/api/analyze?model=kim", json={"user_query": "Is $AAPL a buy?"}, headers={"X-API-Key": VALID_KEY},
    )

    assert response.status_code == 200
    assert captured["model"] == "kim"
