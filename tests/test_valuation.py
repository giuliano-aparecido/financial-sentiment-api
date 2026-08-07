from app.services.valuation import graham_number, valuation_block, valuation_block_for


def test_graham_number_computes_sqrt_of_22_5_times_eps_times_bvps():
    # 22.5 * 6.07 * 20.0 = 2731.5, sqrt ~= 52.264
    assert round(graham_number(6.07, 20.0), 2) == 52.26


def test_graham_number_none_for_negative_eps():
    assert graham_number(-1.0, 20.0) is None


def test_graham_number_none_for_negative_book_value():
    assert graham_number(6.07, -5.0) is None


def test_graham_number_none_for_zero_eps():
    assert graham_number(0.0, 20.0) is None


def test_graham_number_none_for_missing_inputs():
    assert graham_number(None, 20.0) is None
    assert graham_number(6.07, None) is None


def test_valuation_block_reports_overvalued_when_price_above_intrinsic():
    block = valuation_block(189.30, 52.26)
    assert "Intrinsic Value (Graham Number): $52.26" in block
    assert "overvalued by ~262%" in block


def test_valuation_block_reports_undervalued_when_price_below_intrinsic():
    block = valuation_block(40.0, 52.26)
    assert "undervalued" in block
    assert "overvalued" not in block


def test_valuation_block_not_applicable_when_intrinsic_is_none():
    assert valuation_block(189.30, None) == "Not applicable (negative or missing EPS/book value)."


def test_valuation_block_data_unavailable_when_price_missing():
    assert valuation_block(None, 52.26) == "Data unavailable."


def test_valuation_block_for_computes_from_fundamentals_dict():
    fundamentals = {"price": 189.30, "eps_trailing": 6.07, "book_value_per_share": 20.0}
    block = valuation_block_for(fundamentals)
    assert "Intrinsic Value (Graham Number): $52.26" in block


def test_valuation_block_for_not_applicable_on_negative_eps():
    fundamentals = {"price": 189.30, "eps_trailing": -2.0, "book_value_per_share": 20.0}
    assert valuation_block_for(fundamentals) == "Not applicable (negative or missing EPS/book value)."


def test_valuation_block_for_data_unavailable_when_fundamentals_missing():
    assert valuation_block_for(None) == "Data unavailable."
    assert valuation_block_for({"price": None}) == "Data unavailable."
