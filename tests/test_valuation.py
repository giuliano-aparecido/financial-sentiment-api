from app.services.valuation import (
    cash_flow_basis_value,
    classify_valuation_basis,
    intrinsic_value,
    scenario_dcf_value,
    valuation_block,
    valuation_block_for,
)

# --- classify_valuation_basis ---


def test_classify_unprofitable_company_uses_revenue_basis():
    assert classify_valuation_basis(eps_trailing=-1.0, payout_ratio=None, sector="Technology", free_cash_flow=None) == "revenue"


def test_classify_missing_eps_uses_revenue_basis():
    assert classify_valuation_basis(eps_trailing=None, payout_ratio=0.5, sector="Energy", free_cash_flow=1e9) == "revenue"


def test_classify_zero_eps_uses_revenue_basis():
    assert classify_valuation_basis(eps_trailing=0.0, payout_ratio=None, sector="Technology", free_cash_flow=None) == "revenue"


def test_classify_high_payout_uses_dividends_basis():
    # Matches KO/XOM-style mature payers (real payout ratios ~0.62/~0.68).
    assert classify_valuation_basis(eps_trailing=3.5, payout_ratio=0.62, sector="Consumer Defensive", free_cash_flow=1e9) == "dividends"


def test_classify_payout_exactly_at_threshold_uses_dividends_basis():
    assert classify_valuation_basis(eps_trailing=3.5, payout_ratio=0.40, sector="Technology", free_cash_flow=None) == "dividends"


def test_classify_asset_heavy_sector_with_positive_fcf_uses_fcf_basis():
    assert classify_valuation_basis(eps_trailing=5.0, payout_ratio=0.10, sector="Industrials", free_cash_flow=2e9) == "fcf"


def test_classify_asset_heavy_sector_without_fcf_falls_back_to_eps():
    # Confirmed live: yfinance's freeCashflow is None for banks (Financial
    # Services isn't in ASSET_HEAVY_SECTORS anyway, but this also covers an
    # asset-heavy company mid capex-spike with no usable FCF figure).
    assert classify_valuation_basis(eps_trailing=5.0, payout_ratio=0.10, sector="Industrials", free_cash_flow=None) == "eps"


def test_classify_asset_heavy_sector_with_negative_fcf_falls_back_to_eps():
    assert classify_valuation_basis(eps_trailing=5.0, payout_ratio=0.10, sector="Energy", free_cash_flow=-5e8) == "eps"


def test_classify_profitable_low_payout_non_asset_heavy_uses_eps_basis():
    # Matches AAPL-style tech (real payout ratio ~0.12).
    assert classify_valuation_basis(eps_trailing=8.71, payout_ratio=0.12, sector="Technology", free_cash_flow=1e11) == "eps"


def test_classify_bank_uses_eps_basis_via_fallback():
    # Financial Services is never in ASSET_HEAVY_SECTORS, and yfinance's
    # freeCashflow is None for banks in practice (confirmed live for JPM) -
    # both independently route here to "eps".
    assert classify_valuation_basis(eps_trailing=15.0, payout_ratio=0.26, sector="Financial Services", free_cash_flow=None) == "eps"


# --- cash_flow_basis_value ---


def test_cash_flow_basis_value_eps_is_already_per_share():
    assert cash_flow_basis_value("eps", {"eps_trailing": 8.71}) == 8.71


def test_cash_flow_basis_value_dividends_is_already_per_share():
    assert cash_flow_basis_value("dividends", {"dividend_rate": 2.12}) == 2.12


def test_cash_flow_basis_value_revenue_divides_by_shares_approx():
    # shares approx = market_cap / price = 3.0e12 / 300.0 = 1.0e10
    fundamentals = {"total_revenue": 4.0e11, "market_cap": 3.0e12, "price": 300.0}
    assert cash_flow_basis_value("revenue", fundamentals) == 40.0


def test_cash_flow_basis_value_fcf_divides_by_shares_approx():
    fundamentals = {"free_cash_flow": 1.0e11, "market_cap": 3.0e12, "price": 300.0}
    assert cash_flow_basis_value("fcf", fundamentals) == 10.0


def test_cash_flow_basis_value_revenue_none_when_shares_not_derivable():
    assert cash_flow_basis_value("revenue", {"total_revenue": 4.0e11, "market_cap": None, "price": 300.0}) is None


def test_cash_flow_basis_value_fcf_none_when_fcf_missing():
    assert cash_flow_basis_value("fcf", {"free_cash_flow": None, "market_cap": 3.0e12, "price": 300.0}) is None


# --- scenario_dcf_value ---
# Reference value below computed via an independent Python loop (not this
# module) for cf0=10.0, g1=g2=0.08 (the "normal" scenario's growth - equal
# across both stages in this model, unlike an earlier rejected fading-
# growth variant), exit_multiple=15.0, r=0.10: year-10 projected cash flow
# is 10 x 1.08^10 ~= 21.589; PV of years 1-10 plus the exit-multiple
# terminal value (21.589 x 15, discounted back 10 years) sums to
# ~215.379972.


def test_scenario_dcf_value_matches_independent_reference_calc():
    pv = scenario_dcf_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=15.0, discount_rate=0.10)
    assert round(pv, 4) == 215.3800


def test_scenario_dcf_value_higher_growth_produces_higher_pv():
    low = scenario_dcf_value(cf0=10.0, g1=0.04, g2=0.04, exit_multiple=15.0, discount_rate=0.10)
    high = scenario_dcf_value(cf0=10.0, g1=0.20, g2=0.20, exit_multiple=15.0, discount_rate=0.10)
    assert high > low


def test_scenario_dcf_value_higher_exit_multiple_produces_higher_pv():
    low = scenario_dcf_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=10.0, discount_rate=0.10)
    high = scenario_dcf_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=30.0, discount_rate=0.10)
    assert high > low


def test_scenario_dcf_value_higher_discount_rate_produces_lower_pv():
    low_rate = scenario_dcf_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=15.0, discount_rate=0.08)
    high_rate = scenario_dcf_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=15.0, discount_rate=0.15)
    assert high_rate < low_rate


# --- intrinsic_value ---
# Reference: same independent loop as above, run once per SCENARIOS entry
# (normal: g1=g2=0.08, exit=15x; best: g1=g2=0.10, exit=30x; worst:
# g1=g2=0.04, exit=10x; discount_rate=0.10 fixed across all three) and
# weighted 0.60/0.20/0.20: normal ~= 215.379972, best = 400.0 exactly
# (10 x 1.10^10 x 30, discounted back 10 years at 10% - the discount
# factor and growth factor cancel exactly since g=r=0.10, leaving
# cf0 x exit_multiple = 10 x 30 = 300 plus the smaller annual-cashflow PV
# terms), worst ~= 131.482128 -> weighted ~= 235.524409.


def test_intrinsic_value_matches_independent_reference_calc():
    iv = intrinsic_value(cf0=10.0)
    assert round(iv, 4) == 235.5244


def test_intrinsic_value_none_for_missing_cf0():
    assert intrinsic_value(cf0=None) is None


def test_intrinsic_value_none_for_zero_cf0():
    assert intrinsic_value(cf0=0.0) is None


def test_intrinsic_value_none_for_negative_cf0():
    assert intrinsic_value(cf0=-5.0) is None


# --- valuation_block ---


def test_valuation_block_reports_overvalued_when_price_above_intrinsic():
    block = valuation_block(price=300.0, intrinsic=235.5244, basis="eps")
    assert "Intrinsic Value (EPS-based): $235.52" in block
    assert "overvalued by ~27%" in block


def test_valuation_block_reports_undervalued_when_price_below_intrinsic():
    block = valuation_block(price=100.0, intrinsic=235.5244, basis="eps")
    assert "undervalued" in block
    assert "overvalued" not in block


def test_valuation_block_shows_basis_label():
    assert "FCF-based" in valuation_block(price=100.0, intrinsic=50.0, basis="fcf")
    assert "Dividend-based" in valuation_block(price=100.0, intrinsic=50.0, basis="dividends")
    assert "Revenue-based" in valuation_block(price=100.0, intrinsic=50.0, basis="revenue")


def test_valuation_block_not_applicable_when_intrinsic_is_none():
    assert valuation_block(price=189.30, intrinsic=None, basis="eps") == (
        "Not applicable (insufficient data for the eps-based valuation basis)."
    )


def test_valuation_block_data_unavailable_when_price_missing():
    assert valuation_block(price=None, intrinsic=52.26, basis="eps") == "Data unavailable."


# --- valuation_block_for ---


def test_valuation_block_for_computes_from_fundamentals_dict():
    fundamentals = {
        "price": 300.0,
        "eps_trailing": 10.0,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "free_cash_flow": None,
        "market_cap": 3.0e12,
    }
    block = valuation_block_for(fundamentals)
    assert "Intrinsic Value (EPS-based): $235.52" in block


def test_valuation_block_for_uses_dividends_basis_for_high_payout_company():
    fundamentals = {
        "price": 80.0,
        "eps_trailing": 3.5,
        "payout_ratio": 0.62,
        "sector": "Consumer Defensive",
        "dividend_rate": 2.12,
        "market_cap": 3.5e11,
    }
    assert "Dividend-based" in valuation_block_for(fundamentals)


def test_valuation_block_for_uses_revenue_basis_for_unprofitable_company():
    fundamentals = {
        "price": 5.0,
        "eps_trailing": -1.2,
        "payout_ratio": None,
        "sector": "Consumer Defensive",
        "total_revenue": 5.0e8,
        "market_cap": 1.0e9,
    }
    assert "Revenue-based" in valuation_block_for(fundamentals)


def test_valuation_block_for_not_applicable_when_basis_input_missing():
    # EPS basis chosen (profitable, low payout, not asset-heavy), but no
    # eps_trailing means cash_flow_basis_value has nothing to work with.
    fundamentals = {
        "price": 189.30,
        "eps_trailing": None,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "total_revenue": None,
        "market_cap": 1.0e12,
    }
    assert valuation_block_for(fundamentals) == (
        "Not applicable (insufficient data for the revenue-based valuation basis)."
    )


def test_valuation_block_for_data_unavailable_when_fundamentals_missing():
    assert valuation_block_for(None) == "Data unavailable."
    assert valuation_block_for({"price": None}) == "Data unavailable."
