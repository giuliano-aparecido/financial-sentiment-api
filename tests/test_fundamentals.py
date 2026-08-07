from app.services.fundamentals import format_market_cap, market_data_block


def test_format_market_cap_uses_trillions_above_1e12():
    assert format_market_cap(2.95e12) == "$2.95T"


def test_format_market_cap_uses_billions_below_1e12():
    assert format_market_cap(13.8e9) == "$13.8B"


FULL_FUNDAMENTALS = {
    "price": 189.30,
    "market_cap": 2.95e12,
    "pe_trailing": 31.2,
    "pe_forward": 27.8,
    "eps_trailing": 6.07,
    "dividend_yield": 0.55,
    "year_low": 164.08,
    "year_high": 237.23,
}


def test_market_data_block_renders_full_shape():
    block = market_data_block(FULL_FUNDAMENTALS)
    assert block == (
        "Price: $189.30 | Market Cap: $2.95T\n"
        "P/E (trailing): 31.2 | P/E (forward): 27.8\n"
        "EPS (trailing): $6.07 | Dividend Yield: 0.55%\n"
        "52-Week Range: $164.08 - $237.23"
    )


def test_market_data_block_falls_back_to_na_for_missing_optional_fields():
    partial = {**FULL_FUNDAMENTALS, "pe_forward": None, "year_low": None}
    block = market_data_block(partial)
    assert "P/E (forward): N/A" in block
    assert "52-Week Range: N/A" in block


def test_market_data_block_defaults_dividend_yield_to_zero_when_missing():
    partial = {**FULL_FUNDAMENTALS, "dividend_yield": None}
    assert "Dividend Yield: 0.00%" in market_data_block(partial)


def test_market_data_block_data_unavailable_when_none():
    assert market_data_block(None) == "Data unavailable."


def test_market_data_block_data_unavailable_when_market_cap_missing():
    assert market_data_block({"price": 189.30, "market_cap": None}) == "Data unavailable."
