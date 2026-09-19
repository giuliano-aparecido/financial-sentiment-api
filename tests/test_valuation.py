import pytest

from app.services.valuation import (
    CURATED_SCENARIOS,
    DIVIDEND_PAYOUT_THRESHOLD,
    G1_CAP,
    G1_FLOOR,
    G2_DIVIDENDS_FLOOR,
    VALUATION_PCT_DISPLAY_CAP,
    _sustainable_growth_rate,
    build_scenarios,
    cash_flow_basis_value,
    classify_valuation_basis,
    intrinsic_value,
    scenario_dcf_value,
    scenario_terminal_value,
    valuation_assessment_for,
    valuation_block,
    valuation_block_for,
)

# --- classify_valuation_basis ---


def test_classify_reit_uses_dividends_basis_regardless_of_low_payout_ratio():
    # Confirmed live: Aedifica (a real REIT) has payout_ratio=0.34 - below
    # DIVIDEND_PAYOUT_THRESHOLD - because GAAP payout ratio is computed
    # against depreciation-depressed GAAP earnings, not the real cash REITs
    # actually distribute. The Real Estate sector override must win even
    # when payout_ratio alone would say "eps".
    assert classify_valuation_basis({"eps_trailing": 11.49, "payout_ratio": 0.34, "sector": "Real Estate", "free_cash_flow": None}) == "dividends"


def test_classify_reit_uses_dividends_basis_even_when_technically_unprofitable():
    # GAAP depreciation can push a healthy REIT's reported EPS to zero or
    # negative - the sector override must win before the profitability
    # check, not after it (a REIT should never fall to "revenue").
    assert classify_valuation_basis({"eps_trailing": -0.5, "payout_ratio": None, "sector": "Real Estate", "free_cash_flow": None}) == "dividends"


def test_classify_unprofitable_company_uses_revenue_basis():
    assert classify_valuation_basis({"eps_trailing": -1.0, "payout_ratio": None, "sector": "Technology", "free_cash_flow": None}) == "revenue"


def test_classify_missing_eps_uses_revenue_basis():
    assert classify_valuation_basis({"eps_trailing": None, "payout_ratio": 0.5, "sector": "Energy", "free_cash_flow": 1e9}) == "revenue"


def test_classify_zero_eps_uses_revenue_basis():
    assert classify_valuation_basis({"eps_trailing": 0.0, "payout_ratio": None, "sector": "Technology", "free_cash_flow": None}) == "revenue"


def test_classify_high_payout_uses_dividends_basis():
    # Matches KO/VZ/MO-style mature payers (real payout ratios ~0.62-0.89).
    # XOM is deliberately NOT such an example despite a plausible-looking
    # payout ratio (~0.525) - see CURATED_SCENARIOS_BASIS's override in
    # valuation_block_for, which wins for curated tickers regardless of
    # what this classifier alone would say.
    assert classify_valuation_basis({"eps_trailing": 3.5, "payout_ratio": 0.62, "sector": "Consumer Defensive", "free_cash_flow": 1e9}) == "dividends"


def test_classify_payout_exactly_at_threshold_uses_dividends_basis():
    assert classify_valuation_basis({"eps_trailing": 3.5, "payout_ratio": DIVIDEND_PAYOUT_THRESHOLD, "sector": "Technology", "free_cash_flow": None}) == "dividends"


def test_classify_payout_above_ceiling_falls_through_to_fcf_not_dividends():
    # Confirmed live: DSM-Firmenich, mid its 2023 merger, showed
    # payout_ratio=1.79 - paying out more than its entire trailing
    # earnings. That's a transiently earnings-crushed asset-heavy
    # chemicals company holding its dividend flat, not a genuine
    # Kinder-Morgan-style cash-cow payout policy - it should fall through
    # to the asset-heavy/fcf check, not get caught by the payout check.
    assert classify_valuation_basis({"eps_trailing": 1.4, "payout_ratio": 1.79, "sector": "Basic Materials", "free_cash_flow": 5e8}) == "fcf"


def test_classify_payout_above_ceiling_falls_through_to_eps_when_fcf_unusable():
    # Same DSM-style distorted payout ratio, but this time FCF is also
    # currently negative (also confirmed live for DSM-Firmenich) - must
    # land on "eps", not get stuck on "dividends" via the excluded payout
    # check, and not misfire into "fcf" with an unusable negative figure.
    assert classify_valuation_basis({"eps_trailing": 1.4, "payout_ratio": 1.79, "sector": "Basic Materials", "free_cash_flow": -1.4e8}) == "eps"


def test_classify_payout_at_ceiling_still_uses_dividends_basis():
    assert classify_valuation_basis({"eps_trailing": 3.5, "payout_ratio": 1.20, "sector": "Technology", "free_cash_flow": None}) == "dividends"


def test_classify_asset_heavy_sector_with_positive_fcf_uses_fcf_basis():
    assert classify_valuation_basis({"eps_trailing": 5.0, "payout_ratio": 0.10, "sector": "Industrials", "free_cash_flow": 2e9}) == "fcf"


def test_classify_asset_heavy_sector_without_fcf_falls_back_to_eps():
    # Confirmed live: yfinance's freeCashflow is None for banks (Financial
    # Services isn't in ASSET_HEAVY_SECTORS anyway, but this also covers an
    # asset-heavy company mid capex-spike with no usable FCF figure).
    assert classify_valuation_basis({"eps_trailing": 5.0, "payout_ratio": 0.10, "sector": "Industrials", "free_cash_flow": None}) == "eps"


def test_classify_asset_heavy_sector_with_negative_fcf_falls_back_to_eps():
    assert classify_valuation_basis({"eps_trailing": 5.0, "payout_ratio": 0.10, "sector": "Energy", "free_cash_flow": -5e8}) == "eps"


def test_classify_profitable_low_payout_non_asset_heavy_uses_eps_basis():
    # Matches AAPL-style tech (real payout ratio ~0.12).
    assert classify_valuation_basis({"eps_trailing": 8.71, "payout_ratio": 0.12, "sector": "Technology", "free_cash_flow": 1e11}) == "eps"


def test_classify_bank_uses_eps_basis_via_fallback():
    # Financial Services is never in ASSET_HEAVY_SECTORS, and yfinance's
    # freeCashflow is None for banks in practice (confirmed live for JPM) -
    # both independently route here to "eps".
    assert classify_valuation_basis({"eps_trailing": 15.0, "payout_ratio": 0.26, "sector": "Financial Services", "free_cash_flow": None}) == "eps"


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


# --- scenario_dcf_value (full-sum: interim years + terminal, "dividends" basis) ---
# Reference value below computed via an independent Python loop (not this
# module) for cf0=10.0, g1=g2=0.08, exit_multiple=15.0, r=0.10: year-10
# projected cash flow is 10 x 1.08^10 ~= 21.589; PV of years 1-10 plus the
# exit-multiple terminal value (21.589 x 15, discounted back 10 years) sums
# to ~215.379972.


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


# --- scenario_terminal_value (terminal-only, "eps"/"fcf"/"revenue" bases) ---
# Reference value below computed independently for cf0=10.0, g1=g2=0.08,
# exit_multiple=15.0, r=0.10: year-10 cash flow 10 x 1.08^10 ~= 21.589,
# terminal value 21.589 x 15 ~= 323.84, discounted back 10 years at 10% ->
# ~124.85. Deliberately smaller than scenario_dcf_value's 215.38 for the
# same inputs - that gap IS the interim-year summation scenario_dcf_value
# adds and scenario_terminal_value doesn't (see module history point 3b).


def test_scenario_terminal_value_matches_independent_reference_calc():
    pv = scenario_terminal_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=15.0, discount_rate=0.10)
    assert round(pv, 2) == 124.85


def test_scenario_terminal_value_is_lower_than_full_sum_for_same_inputs():
    terminal_only = scenario_terminal_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=15.0, discount_rate=0.10)
    full_sum = scenario_dcf_value(cf0=10.0, g1=0.08, g2=0.08, exit_multiple=15.0, discount_rate=0.10)
    assert terminal_only < full_sum


def test_scenario_terminal_value_higher_growth_produces_higher_pv():
    low = scenario_terminal_value(cf0=10.0, g1=0.04, g2=0.04, exit_multiple=15.0, discount_rate=0.10)
    high = scenario_terminal_value(cf0=10.0, g1=0.20, g2=0.20, exit_multiple=15.0, discount_rate=0.10)
    assert high > low


# --- build_scenarios ---


def test_build_scenarios_uses_curated_table_verbatim_for_known_ticker():
    fundamentals = {"growth_0y": 999.0, "growth_1y": 999.0}  # would-be consensus, ignored
    scenarios = build_scenarios("NVDA", fundamentals, basis="eps")
    assert scenarios["normal"]["g1"] == CURATED_SCENARIOS["NVDA"]["normal"]["g1"]
    assert scenarios["normal"]["g2"] == CURATED_SCENARIOS["NVDA"]["normal"]["g2"]
    assert scenarios["normal"]["exit_multiple"] == CURATED_SCENARIOS["NVDA"]["normal"]["exit_multiple"]


def test_build_scenarios_curated_tickers_use_equal_probability():
    scenarios = build_scenarios("NVDA", {}, basis="eps")
    assert scenarios["normal"]["probability"] == 1 / 3
    assert scenarios["best"]["probability"] == 1 / 3
    assert scenarios["worst"]["probability"] == 1 / 3


def test_build_scenarios_ignores_curated_table_when_basis_does_not_match():
    # Regression test: classify_valuation_basis's result for a real ticker
    # can legitimately differ call to call (e.g. a payout ratio that lands
    # in the dividends band this time) - confirmed live that applying
    # AAPL's EPS-calibrated growth/exit-multiple assumptions to a
    # "dividends" cf0 (its much smaller dividend rate) produces a number
    # with no relationship to the analyst's actual target, not just a less
    # accurate one. AAPL is calibrated for "eps" (CURATED_SCENARIOS_BASIS),
    # so a "dividends" call must NOT use its curated table.
    fundamentals = {"growth_0y": None, "growth_1y": None}
    scenarios = build_scenarios("AAPL", fundamentals, basis="dividends")
    assert scenarios["normal"]["g1"] != CURATED_SCENARIOS["AAPL"]["normal"]["g1"]
    assert scenarios["normal"]["g1"] == 0.08  # falls through to G1_FALLBACK


def test_build_scenarios_pep_curated_only_applies_to_dividends_basis():
    # PEP is the one CURATED_SCENARIOS ticker actually calibrated for
    # "dividends", not "eps" - the inverse case from the AAPL test above.
    scenarios = build_scenarios("PEP", {}, basis="eps")
    assert scenarios["normal"]["g1"] != CURATED_SCENARIOS["PEP"]["normal"]["g1"]


def test_build_scenarios_falls_back_to_generic_for_unknown_ticker_without_consensus():
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", {}, basis="eps")
    assert scenarios["normal"]["g1"] == 0.08
    # g2 = min(GROWTH_BASIS_G2's flat default, this scenario's own g1) -
    # here g1 (0.08, G1_FALLBACK's normal) is BELOW the flat 0.10 default,
    # so g2 matches g1 rather than the flat ceiling. Confirmed live across
    # all 6 CURATED_SCENARIOS tickers' 18 g1/g2 pairs that g2 <= g1 always
    # holds - see GROWTH_BASIS_G2's comment.
    assert scenarios["normal"]["g2"] == 0.08
    assert scenarios["normal"]["exit_multiple"] == 20.0


def test_build_scenarios_g2_matches_flat_default_when_g1_exceeds_it():
    # The flat default still applies as a CEILING when g1 is comfortably
    # above it - e.g. a strong sustainable-growth-rate case.
    scenarios = build_scenarios(
        "SOME_UNKNOWN_TICKER",
        {"eps_trailing": 5.0, "book_value_per_share": 20.0, "payout_ratio": 0.0},
        basis="eps",
    )
    assert scenarios["normal"]["g1"] > 0.10
    assert scenarios["normal"]["g2"] == 0.10


def test_build_scenarios_derives_normal_g1_from_same_direction_consensus():
    # MSFT-style: 0y=13.87%, +1y=19.30%, both positive -> average = 16.585%.
    fundamentals = {"growth_0y": 0.1387, "growth_1y": 0.1930}
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert round(scenarios["normal"]["g1"], 4) == round((0.1387 + 0.1930) / 2, 4)
    # g2/exit_multiple stay generic even when g1 is derived.
    assert scenarios["normal"]["g2"] == 0.10
    assert scenarios["normal"]["exit_multiple"] == 20.0


def test_build_scenarios_derives_best_worst_g1_as_offset_from_blended_normal():
    # best/worst are OFFSETS from the SAME 2-year-blended "normal" value,
    # by the analyst range's own spread around growth_0y - not a direct
    # substitution of growth_0y_high/low (see build_scenarios' own comment
    # for why: that used to let "best" end up WORSE than "normal" when
    # growth_1y differed a lot from growth_0y, since best/worst never saw
    # growth_1y at all under the old derivation).
    fundamentals = {
        "growth_0y": 0.1387,
        "growth_1y": 0.1930,
        "growth_0y_high": 0.25,
        "growth_0y_low": 0.05,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    blended_normal = (0.1387 + 0.1930) / 2
    assert scenarios["normal"]["g1"] == blended_normal
    assert scenarios["best"]["g1"] == pytest.approx(blended_normal + (0.25 - 0.1387))
    assert scenarios["worst"]["g1"] == pytest.approx(blended_normal - (0.1387 - 0.05))
    # The property that actually matters: best >= normal >= worst always,
    # not just for this specific example.
    assert scenarios["best"]["g1"] >= scenarios["normal"]["g1"] >= scenarios["worst"]["g1"]


def test_build_scenarios_best_worst_g1_ordering_holds_even_when_growth_1y_diverges():
    # Regression test for the QCOM case that motivated this fix: this
    # year's estimate is much worse than next year's, so the OLD
    # growth_0y_high/low-direct derivation put "best" (using only this
    # year's still-bad range) below "normal" (which partly saw next
    # year's recovery via the blend) - inverted scenario ordering.
    fundamentals = {
        "growth_0y": -0.1253, "growth_1y": -0.0305,
        "growth_0y_high": -0.1180, "growth_0y_low": -0.1322,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert scenarios["best"]["g1"] >= scenarios["normal"]["g1"] >= scenarios["worst"]["g1"]


def test_build_scenarios_caps_derived_g1_at_g1_cap():
    # Regression test for a confirmed-live DCF blowup: an uncapped g1
    # compounds over 5 years then multiplies by up to 25x, barely dented by
    # discounting - a real (not just theoretical) aggressive consensus
    # growth estimate could blow the resulting intrinsic value out to
    # multiples of the current price. growth_0y_high here (0.90) is well
    # above G1_CAP.
    fundamentals = {
        "growth_0y": 0.50, "growth_1y": 0.50,
        "growth_0y_high": 0.90, "growth_0y_low": 0.30,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert scenarios["best"]["g1"] == G1_CAP


def test_build_scenarios_floors_derived_g1_at_g1_floor():
    # Symmetric case to the G1_CAP test above - confirmed live this was a
    # real gap, not just a theoretical one (see G1_FLOOR's own comment):
    # QCOM's real consensus derives a g1 around -8%, comfortably above
    # this floor on its own, but a worse real case is entirely plausible
    # and previously had no downside bound at all.
    fundamentals = {
        "growth_0y": -0.50, "growth_1y": -0.50,
        "growth_0y_high": -0.30, "growth_0y_low": -0.90,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert scenarios["worst"]["g1"] == G1_FLOOR


def test_build_scenarios_dividends_g2_floored_even_when_g1_deeply_negative():
    # The actual QCOM case (see G2_DIVIDENDS_FLOOR's own comment): g2 = g1
    # for the dividends basis, but g1 alone being negative (a plausible,
    # real one-bad-year claim) shouldn't justify projecting that SAME
    # decline rate for a SECOND five-year stage (a much stronger, decade-
    # long claim a temporary dip doesn't support).
    fundamentals = {
        "growth_0y": -0.1253, "growth_1y": -0.0305,
        "growth_0y_high": -0.1180, "growth_0y_low": -0.1322,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="dividends")
    for tier in ("normal", "best", "worst"):
        assert scenarios[tier]["g1"] < G2_DIVIDENDS_FLOOR  # the underlying claim is genuinely worse...
        assert scenarios[tier]["g2"] == G2_DIVIDENDS_FLOOR  # ...but g2 doesn't inherit it uncapped


def test_build_scenarios_dividends_g2_not_floored_when_g1_milder():
    # G2_DIVIDENDS_FLOOR shouldn't kick in for a mild, plausible decline -
    # only the extreme case needs bounding.
    fundamentals = {
        "growth_0y": -0.01, "growth_1y": -0.01,
        "growth_0y_high": 0.0, "growth_0y_low": -0.02,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="dividends")
    assert scenarios["normal"]["g2"] == scenarios["normal"]["g1"]
    assert scenarios["normal"]["g2"] > G2_DIVIDENDS_FLOOR


def test_build_scenarios_does_not_cap_g1_below_the_cap():
    # The cap must not clamp DOWN a legitimately high-but-under-the-cap
    # estimate - only values that actually exceed it.
    fundamentals = {
        "growth_0y": 0.10, "growth_1y": 0.10,
        "growth_0y_high": 0.20, "growth_0y_low": 0.05,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert scenarios["best"]["g1"] == 0.20


def test_build_scenarios_curated_tickers_bypass_g1_cap():
    # CURATED_SCENARIOS entries are hand-vetted against a real analyst's
    # own DCF (see module docstring) - the cap must not touch them, even
    # though NVDA's own curated "best" g1 (0.30) is close to G1_CAP (0.40).
    scenarios = build_scenarios("NVDA", {}, basis="eps")
    assert scenarios["normal"]["g1"] == CURATED_SCENARIOS["NVDA"]["normal"]["g1"]
    assert scenarios["best"]["g1"] == CURATED_SCENARIOS["NVDA"]["best"]["g1"]


def test_sustainable_growth_rate_is_roe_times_retention():
    # JPM-style example: EPS $19, book value/share $105 -> ROE 18.1%,
    # payout ratio 27% -> retention 73%.
    fundamentals = {"eps_trailing": 19.0, "book_value_per_share": 105.0, "payout_ratio": 0.27}
    roe = 19.0 / 105.0
    assert round(_sustainable_growth_rate(fundamentals), 4) == round(roe * (1 - 0.27), 4)


def test_sustainable_growth_rate_defaults_payout_to_zero_when_missing():
    # No payout_ratio -> assume full reinvestment (100% retention), not
    # "unusable" - a real non-dividend-payer has no payout_ratio at all.
    fundamentals = {"eps_trailing": 10.0, "book_value_per_share": 50.0}
    assert _sustainable_growth_rate(fundamentals) == 10.0 / 50.0


def test_sustainable_growth_rate_none_when_unprofitable():
    fundamentals = {"eps_trailing": -2.0, "book_value_per_share": 50.0, "payout_ratio": 0.0}
    assert _sustainable_growth_rate(fundamentals) is None


def test_sustainable_growth_rate_none_when_book_value_missing():
    fundamentals = {"eps_trailing": 10.0, "book_value_per_share": None, "payout_ratio": 0.0}
    assert _sustainable_growth_rate(fundamentals) is None


def test_build_scenarios_uses_sustainable_growth_rate_when_no_consensus():
    fundamentals = {"eps_trailing": 19.0, "book_value_per_share": 105.0, "payout_ratio": 0.27}
    expected_g1 = _sustainable_growth_rate(fundamentals)
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert round(scenarios["normal"]["g1"], 4) == round(expected_g1, 4)
    assert round(scenarios["best"]["g1"], 4) == round(expected_g1 + 0.02, 4)
    assert round(scenarios["worst"]["g1"], 4) == round(expected_g1 - 0.04, 4)


def test_build_scenarios_consensus_still_wins_over_sustainable_growth_rate():
    # Real analyst consensus data (when reliable) is the most trustworthy
    # source - it must override the fundamentals-derived formula, not the
    # other way around.
    fundamentals = {
        "eps_trailing": 19.0, "book_value_per_share": 105.0, "payout_ratio": 0.27,
        "growth_0y": 0.20, "growth_1y": 0.22,
    }
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert round(scenarios["normal"]["g1"], 4) == round((0.20 + 0.22) / 2, 4)


def test_build_scenarios_falls_back_to_flat_default_when_neither_source_usable():
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", {}, basis="eps")
    assert scenarios["normal"]["g1"] == 0.08


def test_build_scenarios_falls_back_to_generic_when_consensus_reverses_direction():
    # XOM-style: 0y strongly positive (rebound), +1y negative (giveback) -
    # a distorted base year, not a real trend. Must NOT be averaged.
    fundamentals = {"growth_0y": 0.6567, "growth_1y": -0.0862}
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="dividends")
    assert scenarios["normal"]["g1"] == 0.08


def test_build_scenarios_falls_back_to_generic_when_consensus_missing():
    fundamentals = {"growth_0y": None, "growth_1y": None}
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="eps")
    assert scenarios["normal"]["g1"] == 0.08


def test_build_scenarios_works_without_ticker():
    scenarios = build_scenarios(None, {}, basis="eps")
    assert scenarios["normal"]["g1"] == 0.08


def test_build_scenarios_dividends_basis_g2_matches_g1_not_a_fixed_fade():
    # PEP/XOM both showed g2 == g1 for normal/best under the dividends
    # basis (a mature payer is already near its steady-state rate) - the
    # generic fallback should reproduce that shape, not the growth-basis
    # 10% fade ceiling.
    fundamentals = {"growth_0y": 0.05, "growth_1y": 0.03}
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", fundamentals, basis="dividends")
    assert scenarios["normal"]["g2"] == scenarios["normal"]["g1"]
    assert scenarios["best"]["g2"] == scenarios["best"]["g1"]
    assert scenarios["worst"]["g2"] == scenarios["worst"]["g1"]


def test_build_scenarios_worst_exit_multiple_lower_for_asset_heavy_sector():
    # Confirmed live: XOM (Energy, cyclical) got 12.0x vs PEP (Consumer
    # Defensive) at 15.0x for the same worst-case scenario shape.
    cyclical = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Energy"}, basis="dividends")
    non_cyclical = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Consumer Defensive"}, basis="dividends")
    assert cyclical["worst"]["exit_multiple"] < non_cyclical["worst"]["exit_multiple"]


def test_build_scenarios_normal_exit_multiple_sector_anchored_for_eps_and_fcf():
    # Confirmed live: a flat 20x normal exit multiple for EVERY sector
    # contradicted SECTOR_MEDIAN_PE shown in the same prompt for
    # lower-multiple sectors (Energy's median is 12.0) - min(flat
    # default, sector median) pulls normal/best DOWN for those sectors
    # while leaving Technology (median 28.0, above the flat default)
    # exactly at the old flat calibration, matching CURATED_SCENARIOS.
    cyclical = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Energy"}, basis="eps")
    non_cyclical = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Technology"}, basis="eps")
    assert cyclical["normal"]["exit_multiple"] == 12.0
    assert cyclical["best"]["exit_multiple"] == 17.0  # same +5 absolute spread as the flat default
    assert non_cyclical["normal"]["exit_multiple"] == 20.0
    assert non_cyclical["best"]["exit_multiple"] == 25.0


def test_build_scenarios_normal_exit_multiple_floored_at_own_trailing_pe():
    # Regression test: confirmed live investigating XOM right after it
    # left CURATED_SCENARIOS - a flat Energy sector median (12.0) capped
    # its exit multiple BELOW its own real trailing P/E (20.7x), i.e.
    # assuming the market prices it MORE cheaply in 10 years than it
    # already does today. A company genuinely trading above its sector's
    # median multiple shouldn't get pulled down to the sector's generic
    # level - max(sector_median, own_pe) keeps the sector floor only for
    # names actually trading at or below it. Uses 16.0 (strictly between
    # Energy's 12.0 median and the flat 20.0 ceiling) so the floor's
    # effect is visible without also hitting NORMAL_EXIT_MULTIPLE's cap -
    # see the "extreme_pe" test below for that boundary specifically.
    above_sector_median = build_scenarios(
        "SOME_UNKNOWN_TICKER", {"sector": "Energy", "pe_trailing": 16.0}, basis="eps",
    )
    assert above_sector_median["normal"]["exit_multiple"] == 16.0
    assert above_sector_median["best"]["exit_multiple"] == 21.0  # same +5 absolute spread


def test_build_scenarios_own_trailing_pe_does_not_lower_the_sector_median_floor():
    # A company trading BELOW its sector median still gets the sector
    # median, not pulled down further to its own (lower) P/E - the floor
    # only ever raises the multiple, never lowers it below what the
    # sector-anchored logic already produces.
    below_sector_median = build_scenarios(
        "SOME_UNKNOWN_TICKER", {"sector": "Energy", "pe_trailing": 8.0}, basis="eps",
    )
    assert below_sector_median["normal"]["exit_multiple"] == 12.0


def test_build_scenarios_own_trailing_pe_does_not_exceed_flat_normal_exit_multiple_cap():
    # The own-P/E floor still respects the overall NORMAL_EXIT_MULTIPLE
    # ceiling (20.0) - an extremely high real P/E shouldn't push the exit
    # multiple past what's calibrated for Technology's own curated names.
    extreme_pe = build_scenarios(
        "SOME_UNKNOWN_TICKER", {"sector": "Energy", "pe_trailing": 300.0}, basis="eps",
    )
    assert extreme_pe["normal"]["exit_multiple"] == 20.0


def test_build_scenarios_revenue_basis_uses_ps_style_exit_multiples_not_earnings_style():
    # Regression test: revenue-basis exit multiples used to reuse
    # NORMAL_EXIT_MULTIPLE/BEST_EXIT_MULTIPLE (20x/25x) - P/E-grade
    # multiples never confirmed for a revenue metric - which produced
    # nonsense intrinsic values (confirmed live: FLUT came back "94%
    # undervalued" at $1747.78 vs a ~$99 price). A P/S-style multiple must
    # be far lower than the eps/fcf/dividends multiples for the same
    # scenario tier.
    scenarios = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Technology"}, basis="revenue")
    eps_scenarios = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Technology"}, basis="eps")
    assert scenarios["normal"]["exit_multiple"] < eps_scenarios["normal"]["exit_multiple"]
    assert scenarios["best"]["exit_multiple"] < eps_scenarios["best"]["exit_multiple"]
    assert scenarios["worst"]["exit_multiple"] < eps_scenarios["worst"]["exit_multiple"]


def test_build_scenarios_revenue_basis_exit_multiples_unaffected_by_sector():
    # Unlike eps/fcf/dividends, the revenue-basis multiples are flat -
    # no asset-heavy-sector split (see REVENUE_*_EXIT_MULTIPLE's comment).
    cyclical = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Energy"}, basis="revenue")
    non_cyclical = build_scenarios("SOME_UNKNOWN_TICKER", {"sector": "Technology"}, basis="revenue")
    assert cyclical["worst"]["exit_multiple"] == non_cyclical["worst"]["exit_multiple"]


# --- intrinsic_value ---
# Reference values below are the curated-scenario results for the real
# tickers they were sourced from, confirmed live against the analyst's own
# target prices (see valuation.py module history, point 3): NVDA ~11.8% off
# ($270.71 vs $242.24), MSFT ~2.2% off ($397.20 vs $405.97) - both large
# improvements over the old flat model's 40-70%+ errors, not exact matches
# (the analyst's own per-company judgment isn't fully recoverable - see
# build_scenarios).


def test_intrinsic_value_matches_curated_nvda_reference_calc():
    scenarios = build_scenarios("NVDA", {}, basis="eps")
    iv = intrinsic_value(cf0=6.53, basis="eps", scenarios=scenarios)
    assert round(iv, 2) == 270.71


def test_intrinsic_value_matches_curated_pep_reference_calc():
    scenarios = build_scenarios("PEP", {}, basis="dividends")
    iv = intrinsic_value(cf0=5.86, basis="dividends", scenarios=scenarios)
    assert round(iv, 2) == 102.83


def test_intrinsic_value_matches_curated_aapl_reference_calc():
    # Analyst's own target: $128 at trailing EPS $8.26. Reproduces within
    # 2.6% ($124.65), same tolerance band as this model's other curated
    # tickers (documented 2-12% in the module docstring).
    scenarios = build_scenarios("AAPL", {}, basis="eps")
    iv = intrinsic_value(cf0=8.26, basis="eps", scenarios=scenarios)
    assert round(iv, 2) == 124.65


def test_intrinsic_value_none_for_missing_cf0():
    scenarios = build_scenarios(None, {}, basis="eps")
    assert intrinsic_value(cf0=None, basis="eps", scenarios=scenarios) is None


def test_intrinsic_value_none_for_zero_cf0():
    scenarios = build_scenarios(None, {}, basis="eps")
    assert intrinsic_value(cf0=0.0, basis="eps", scenarios=scenarios) is None


def test_intrinsic_value_none_for_negative_cf0():
    scenarios = build_scenarios(None, {}, basis="eps")
    assert intrinsic_value(cf0=-5.0, basis="eps", scenarios=scenarios) is None


def test_intrinsic_value_revenue_basis_far_lower_than_eps_basis_for_same_cf0():
    # Confirms the exit-multiple fix actually lowers the final weighted
    # intrinsic value, not just the individual scenario constants - same
    # per-share cash-flow figure, revenue basis must land well below eps
    # basis now that it no longer shares eps's 20-25x multiples.
    revenue_scenarios = build_scenarios(None, {}, basis="revenue")
    eps_scenarios = build_scenarios(None, {}, basis="eps")
    revenue_iv = intrinsic_value(cf0=76.9, basis="revenue", scenarios=revenue_scenarios)
    eps_iv = intrinsic_value(cf0=76.9, basis="eps", scenarios=eps_scenarios)
    assert revenue_iv < eps_iv


# --- valuation_block ---


def test_valuation_block_reports_overvalued_when_price_above_intrinsic():
    block = valuation_block(price=300.0, intrinsic=235.5244, basis="eps")
    assert "Intrinsic Value (EPS-based): $235.52" in block
    assert "overvalued by ~27%" in block


def test_valuation_block_reports_undervalued_when_price_below_intrinsic():
    block = valuation_block(price=100.0, intrinsic=235.5244, basis="eps")
    assert "undervalued" in block
    assert "overvalued" not in block


def test_valuation_block_caps_extreme_gap_at_display_cap():
    # Backstop for any path to an extreme gap that G1_CAP alone doesn't
    # reach (e.g. an unusual exit-multiple/cf0 combination) - confirmed
    # live via synthetic data reusing this exact formula: gaps up to 760%
    # before either cap existed. Shows ">" (not "~") once actually
    # capped - confirmed live "~150%" with no marker read as a specific,
    # calm estimate while silently understating a genuinely extreme gap
    # (see valuation_block's own comment for the QCOM case this fixes).
    block = valuation_block(price=1000.0, intrinsic=10.0, basis="eps")  # raw gap: 9900%
    assert f">{VALUATION_PCT_DISPLAY_CAP:.0f}%" in block


def test_valuation_block_does_not_cap_gap_below_the_display_cap():
    block = valuation_block(price=300.0, intrinsic=235.5244, basis="eps")  # raw gap: ~27%
    assert "overvalued by ~27%" in block


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
        "eps_trailing": 6.53,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "free_cash_flow": None,
        "market_cap": 3.0e12,
    }
    block = valuation_block_for(fundamentals, ticker="NVDA")
    assert "Intrinsic Value (EPS-based): $270.71" in block


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


def test_valuation_block_for_curated_ticker_overrides_generic_classification():
    # Regression test: XOM is curated for "eps", but its real profile
    # (Energy sector, positive free_cash_flow, payout_ratio below
    # DIVIDEND_PAYOUT_THRESHOLD) makes classify_valuation_basis alone land
    # on "fcf" - confirmed live this exact fundamentals shape produced
    # classified=fcf, using_curated=False after DIVIDEND_PAYOUT_THRESHOLD
    # was raised to fix QCOM. valuation_block_for must override the
    # classifier's output with CURATED_SCENARIOS_BASIS for curated tickers,
    # not just check the two happen to agree.
    fundamentals = {
        "price": 160.10,
        "eps_trailing": 7.65,
        "payout_ratio": 0.525,
        "sector": "Energy",
        "free_cash_flow": 3.0e10,
        "market_cap": 4.5e11,
    }
    assert classify_valuation_basis(fundamentals) == "fcf"
    assert "EPS-based" in valuation_block_for(fundamentals, ticker="XOM")


def test_valuation_block_for_not_applicable_when_eps_distorted_by_earnings_surprise():
    # Regression test: GOOG's trailing EPS was inflated by two
    # consecutive quarters beating consensus by +94%/+213% (almost
    # certainly mark-to-market gains on its equity investment stakes,
    # not organic growth) - confirmed live this basis produced a wildly
    # wrong intrinsic value (+227% vs a real analyst's target).
    # Deliberately renders "Not applicable" rather than falling back to
    # another basis - confirmed live AMZN (same distortion class, a
    # 214% surprise) has an ALSO-distorted fcf fallback this same
    # quarter (a heavy capex cycle crushing free cash flow), so the
    # fallback chain's "some other basis is probably clean" assumption
    # doesn't hold for this failure mode specifically.
    fundamentals = {
        "price": 343.54,
        "eps_trailing": 19.93,
        "pe_trailing": 17.24,
        "pe_forward": 23.30,
        "payout_ratio": 0.0426,
        "sector": "Communication Services",
        "free_cash_flow": 2.27e10,
        "market_cap": 4.2e12,
        "recent_eps_surprise": 2.13,
    }
    assert valuation_block_for(fundamentals, ticker="GOOG") == (
        "Not applicable (insufficient data for the eps-based valuation basis)."
    )


def test_valuation_block_for_ignores_earnings_surprise_for_curated_tickers():
    # A human already verified curated tickers against real analyst work
    # - that judgment should win even if this specific ticker's trailing
    # EPS also happens to show a large earnings surprise.
    fundamentals = {
        "price": 165.79,
        "eps_trailing": 7.65,
        "pe_trailing": 21.7,
        "pe_forward": 20.0,
        "payout_ratio": 0.0,
        "sector": "Technology",
        "free_cash_flow": 1.0e10,
        "market_cap": 1.7e11,
        "recent_eps_surprise": 2.13,
    }
    assert "EPS-based" in valuation_block_for(fundamentals, ticker="NVDA")


def test_valuation_block_for_normal_surprise_does_not_trigger_not_applicable():
    fundamentals = {
        "price": 165.79,
        "eps_trailing": 8.75,
        "pe_trailing": 18.9,
        "pe_forward": 16.3,
        "payout_ratio": 0.41,
        "sector": "Technology",
        "free_cash_flow": 1.0e10,
        "market_cap": 1.7e11,
        "recent_eps_surprise": 0.25,
    }
    assert "EPS-based" in valuation_block_for(fundamentals, ticker="QCOM")


# --- valuation_assessment_for (feeds the informational valuation_gap_pct
# response field, see inference.py's analyze_two_stage) ---


def test_valuation_assessment_for_returns_same_block_as_valuation_block_for():
    fundamentals = {
        "price": 300.0,
        "eps_trailing": 6.53,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "free_cash_flow": None,
        "market_cap": 3.0e12,
    }
    block, gap_pct = valuation_assessment_for(fundamentals, ticker="NVDA")
    assert block == valuation_block_for(fundamentals, ticker="NVDA")
    assert gap_pct is not None


def test_valuation_assessment_for_gap_sign_positive_when_overvalued():
    # price 300 vs intrinsic $270.71 (see the byte-identical block test
    # above) - price above intrinsic is overvalued, a positive gap per
    # this function's own convention (gap_pct = (price-intrinsic)/intrinsic*100).
    fundamentals = {
        "price": 300.0,
        "eps_trailing": 6.53,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "free_cash_flow": None,
        "market_cap": 3.0e12,
    }
    _block, gap_pct = valuation_assessment_for(fundamentals, ticker="NVDA")
    assert gap_pct > 0
    assert gap_pct == pytest.approx(300.0 / 270.71 * 100 - 100, rel=0.01)


def test_valuation_assessment_for_gap_sign_negative_when_undervalued():
    fundamentals = {
        "price": 100.0,
        "eps_trailing": 20.0,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "free_cash_flow": None,
        "market_cap": 3.0e11,
    }
    block, gap_pct = valuation_assessment_for(fundamentals, ticker="TESTX")
    assert "undervalued" in block
    assert gap_pct < 0


def test_valuation_assessment_for_gap_is_none_when_data_unavailable():
    assert valuation_assessment_for(None) == ("Data unavailable.", None)
    assert valuation_assessment_for({"price": None}) == ("Data unavailable.", None)


def test_valuation_assessment_for_gap_is_none_when_not_applicable():
    fundamentals = {
        "price": 189.30,
        "eps_trailing": None,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "total_revenue": None,
        "market_cap": 1.0e12,
    }
    block, gap_pct = valuation_assessment_for(fundamentals)
    assert block == "Not applicable (insufficient data for the revenue-based valuation basis)."
    assert gap_pct is None


def test_valuation_assessment_for_gap_is_none_when_eps_distorted_by_earnings_surprise():
    fundamentals = {
        "price": 216.15,
        "eps_trailing": 8.75,
        "payout_ratio": 0.0,
        "sector": "Technology",
        "free_cash_flow": 1.0e10,
        "market_cap": 1.7e11,
        "recent_eps_surprise": 2.13,
    }
    block, gap_pct = valuation_assessment_for(fundamentals, ticker="GOOG")
    assert block.startswith("Not applicable")
    assert gap_pct is None


def test_valuation_assessment_for_gap_uncapped_past_display_cap():
    # The rendered block text caps display at VALUATION_PCT_DISPLAY_CAP
    # (150%) - the raw gap_pct returned alongside it (surfaced to callers
    # as the informational `valuation_gap_pct` response field, see
    # inference.py's analyze_two_stage) must NOT be capped the same way:
    # capping the number itself (not just the display) would understate
    # how decisive the signal really is.
    fundamentals = {
        "price": 1000.0,
        "eps_trailing": 6.53,
        "payout_ratio": 0.10,
        "sector": "Technology",
        "free_cash_flow": None,
        "market_cap": 3.0e12,
    }
    block, gap_pct = valuation_assessment_for(fundamentals, ticker="NVDA")
    assert ">150%" in block or ">" in block  # display is capped
    assert gap_pct > VALUATION_PCT_DISPLAY_CAP  # the raw number is not
