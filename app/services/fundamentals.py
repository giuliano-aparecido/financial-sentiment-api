import logging

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_fundamentals(ticker: str) -> dict | None:
    """Fetches current fundamentals for `ticker` via yfinance's .info dict.
    Returns None on any fetch failure or if price itself is missing (the
    one field everything else - market cap, the valuation comparison - is
    anchored to). Individual other fields (P/E, dividend yield, 52-week
    range, EPS, book value) can still be None inside a successful result;
    callers render those as "N/A" rather than failing the whole block.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning("yfinance fundamentals fetch failed for %s: %s", ticker, e)
        return None

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        return None

    return {
        "price": price,
        "market_cap": info.get("marketCap"),
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "eps_trailing": info.get("trailingEps"),
        "book_value_per_share": info.get("bookValue"),
        "dividend_yield": info.get("dividendYield"),
        "year_low": info.get("fiftyTwoWeekLow"),
        "year_high": info.get("fiftyTwoWeekHigh"),
        # Added for the scenario-DCF valuation model (app/services/
        # valuation.py) - all sourced from this SAME .info call, no extra
        # yfinance request. free_cash_flow/total_revenue/dividend_rate are
        # per-company-total or per-share cash-flow-basis candidates;
        # sector/industry/payout_ratio drive which basis gets picked. Each
        # is independently optional - valuation.py's classifier and
        # fail-soft rendering handle any subset being None.
        "free_cash_flow": info.get("freeCashflow"),
        "total_revenue": info.get("totalRevenue"),
        "dividend_rate": info.get("dividendRate") or info.get("trailingAnnualDividendRate"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "payout_ratio": info.get("payoutRatio"),
    }


def format_market_cap(value: float) -> str:
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    return f"${value / 1e9:.1f}B"


def market_data_block(fundamentals: dict | None) -> str:
    """Renders the 'Current Market Data' prompt block. Byte-format matches
    generate_synthetic_dataset.py/generate_real_dataset.py in
    financial-sentiment-model-colab exactly (see that repo's
    CONTRIBUTING.md 4-way sync rule) - the model is trained on this shape.
    """
    if not fundamentals or fundamentals.get("market_cap") is None:
        return "Data unavailable."

    pe_trailing = fundamentals.get("pe_trailing")
    pe_forward = fundamentals.get("pe_forward")
    eps = fundamentals.get("eps_trailing")
    dividend_yield = fundamentals.get("dividend_yield")
    year_low = fundamentals.get("year_low")
    year_high = fundamentals.get("year_high")

    pe_trailing_str = f"{pe_trailing:.1f}" if pe_trailing else "N/A"
    pe_forward_str = f"{pe_forward:.1f}" if pe_forward else "N/A"
    eps_str = f"${eps:.2f}" if eps is not None else "N/A"
    div_yield_str = f"{dividend_yield:.2f}%" if dividend_yield else "0.00%"
    range_str = f"${year_low:.2f} - ${year_high:.2f}" if (year_low and year_high) else "N/A"

    return (
        f"Price: ${fundamentals['price']:.2f} | Market Cap: {format_market_cap(fundamentals['market_cap'])}\n"
        f"P/E (trailing): {pe_trailing_str} | P/E (forward): {pe_forward_str}\n"
        f"EPS (trailing): {eps_str} | Dividend Yield: {div_yield_str}\n"
        f"52-Week Range: {range_str}"
    )
