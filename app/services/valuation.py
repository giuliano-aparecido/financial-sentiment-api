# Scenario-weighted, 2-stage DCF-style intrinsic-value model. Replaces an
# earlier Graham Number implementation (sqrt(22.5 x EPS x book value/share))
# that was confirmed live to be badly broken for asset-light, buyback-heavy
# companies - it showed Apple at "725% overvalued", Tesla at "1321%
# overvalued", purely because Graham Number treats book value as a proxy for
# a company's worth, which fails hard when most of a company's value is
# intangible (brand, ecosystem, IP) rather than balance-sheet assets.
#
# This model classifies each company into one of four valuation bases - the
# metric actually being valued differs by business type, since a single
# metric can't meaningfully value both a bank and a pre-profit growth
# company - and projects THAT metric across two growth stages plus an
# exit-multiple terminal value, discounted back, averaged across three
# scenarios (Normal/Best/Worst). All four bases are treated as per-share
# EQUITY cash flows (not enterprise value), specifically so no net-debt
# bridge is needed - a documented simplification, not a hidden one,
# consistent with this project's existing "Not applicable"/"Data
# unavailable" fail-soft pattern.
#
# History of rejected/replaced approaches, for the record:
#   1. Graham's own growth-adjusted revision (V = EPS x (8.5+2g) x 4.4/Y)
#      using yfinance's raw earningsGrowth field - made things WORSE
#      (Tesla -> 4374%), because that field is a noisy single-quarter YoY
#      number (confirmed live: NVDA showed 214%, XOM 112%), not the
#      smoothed long-term rate the formula assumes.
#   2. A self-invented per-basis growth-tier table (different g1/g2/exit
#      multiple per FCF/EPS/Dividend/Revenue basis) with a CAPM-derived,
#      per-company discount rate from beta - fixed most tickers but left
#      Tesla and Nvidia badly broken (Tesla "overvalued" by 1583-2289%
#      across variants tried), because high-beta names got an inflated
#      discount rate on top of already-conservative growth assumptions,
#      double-punishing exactly the names that needed the opposite.
#   3. A flat, universal SCENARIOS table (same g1/g2/exit_multiple for
#      EVERY company, same 60/20/20 probability weights) - resolved most of
#      variant 2's outliers, but was confirmed live against a real
#      investor-analyst's own per-company DCF assumptions (5 tickers:
#      NVDA/MSFT/PEP/NFLX/XOM) to be wrong in two structural ways, not just
#      mistuned constants:
#        a. Probability weights should be equal (1/3 each), not 60/20/20.
#        b. For "eps"/"fcf"/"revenue" bases, summing all 10 years of
#           projected cash flow AND adding a terminal value double-counts -
#           that projected cash flow isn't actually paid to the
#           shareholder each year (unlike a dividend), so only the
#           discounted terminal (eventual sale) price should count. This
#           was the single biggest source of error (NVDA/MSFT/NFLX were all
#           40-70%+ too high under the old full-sum formula; matched within
#           2-12% once switched to terminal-only - see
#           scenario_terminal_value below).
#      Growth/exit-multiple were also confirmed to genuinely vary by
#      company (not universal) - see CURATED_SCENARIOS and build_scenarios
#      below for how per-company inputs are now sourced.
#
# Deterministic, code-only math - never LLM-generated - matching
# financial-sentiment-model-colab's training-data generators (see that
# repo's CONTRIBUTING.md 4-way sync rule for why the RENDERED BLOCK FORMAT
# needs to stay in step with them; the formula/constants below are not yet
# ported there - see this repo's PR history for the staged-rollout plan).

import logging

logger = logging.getLogger(__name__)

STAGE_1_YEARS = 5
STAGE_2_YEARS = 5

# Sector strings match yfinance's Ticker.info["sector"] values exactly.
ASSET_HEAVY_SECTORS = {"Energy", "Industrials", "Basic Materials", "Utilities"}
DIVIDEND_PAYOUT_THRESHOLD = 0.40  # payout_ratio >= this -> treated as a mature dividend payer

# REITs are legally required to distribute ~90% of TAXABLE income as
# dividends, but yfinance's payoutRatio is computed against GAAP earnings,
# which real-estate accounting depresses with large non-cash depreciation
# charges - a REIT can be distributing effectively all its real cash flow
# while showing a deceptively low GAAP payout ratio (confirmed live:
# Aedifica, a real REIT, showed payout_ratio=0.34 - below
# DIVIDEND_PAYOUT_THRESHOLD - which routed it to "eps" instead of
# "dividends"). Same reasoning means GAAP EPS itself is unreliable for this
# sector (depreciation can push it to near-zero or negative even for a
# healthy REIT), so this is checked as an unconditional sector override,
# ahead of the profitability check below - not just an addition to the
# payout-ratio check.
REIT_SECTORS = {"Real Estate"}

BASIS_LABELS = {
    "revenue": "Revenue-based",
    "eps": "EPS-based",
    "fcf": "FCF-based",
    "dividends": "Dividend-based",
}

# Flat for every company (not risk-adjusted per company) - deliberately
# simpler than a CAPM/beta-derived rate, and specifically what resolved the
# high-beta-name double-punishment problem described in the module history
# above. Unlike g1/g2/exit_multiple below, this one constant was NOT
# contradicted by the live analyst comparison, so it's kept as-is.
DISCOUNT_RATE = 0.10

# Confirmed live (see module history, point 3a): the analyst's three
# scenarios were weighted equally, not 60/20/20.
SCENARIO_PROBABILITY = 1 / 3

# Exact per-company scenario assumptions from a real investor analyst's own
# DCF, keyed by ticker - used verbatim (bypassing build_scenarios' derived/
# generic logic below) whenever the incoming ticker matches. g1 = years 1-5
# growth, g2 = years 6-10 growth, exit_multiple applied to year-10's
# projected cash flow. Confirmed these reproduce the analyst's own target
# price within 2-12% once combined with equal weighting and the
# terminal-only formula (see scenario_terminal_value) for the non-dividend
# bases - NVDA/MSFT/NFLX/PEP were all 40-70%+ off under the old flat model.
CURATED_SCENARIOS = {
    "NVDA": {
        "normal": {"g1": 0.30, "g2": 0.10, "exit_multiple": 20.0},
        "best": {"g1": 0.30, "g2": 0.15, "exit_multiple": 25.0},
        "worst": {"g1": 0.05, "g2": 0.05, "exit_multiple": 10.0},
    },
    "MSFT": {
        "normal": {"g1": 0.15, "g2": 0.10, "exit_multiple": 20.0},
        "best": {"g1": 0.20, "g2": 0.10, "exit_multiple": 25.0},
        "worst": {"g1": 0.05, "g2": 0.05, "exit_multiple": 12.0},
    },
    "PEP": {
        "normal": {"g1": 0.03, "g2": 0.03, "exit_multiple": 20.0},
        "best": {"g1": 0.05, "g2": 0.05, "exit_multiple": 25.0},
        "worst": {"g1": 0.03, "g2": -0.05, "exit_multiple": 15.0},
    },
    "NFLX": {
        "normal": {"g1": 0.12, "g2": 0.10, "exit_multiple": 20.0},
        "best": {"g1": 0.15, "g2": 0.12, "exit_multiple": 25.0},
        "worst": {"g1": 0.08, "g2": 0.06, "exit_multiple": 15.0},
    },
    "XOM": {
        "normal": {"g1": 0.04, "g2": 0.04, "exit_multiple": 20.0},
        "best": {"g1": 0.06, "g2": 0.06, "exit_multiple": 30.0},
        "worst": {"g1": 0.03, "g2": 0.03, "exit_multiple": 12.0},
    },
}

# Fallback for any ticker not in CURATED_SCENARIOS. exit_multiple/g2 are
# used as-is regardless of consensus data availability - confirmed live
# that the analyst's exit multiples cluster tightly around 20x (normal) and
# 25x (best) across unrelated companies (4-5 of 5 examples each), so those
# look like genuine fixed defaults rather than per-company judgment. g2
# (years 6-10 growth), by contrast, showed NO consistent pattern even
# within the SAME company across scenarios (e.g. NVDA's best case fades to
# 15%, not the 10% its normal case fades to) - so it's kept as a flat
# generic default rather than pretending to derive it. g1 here is only the
# FALLBACK for when a per-ticker consensus growth estimate isn't available
# or isn't trustworthy (see build_scenarios) - reuses this model's
# pre-existing "average company" growth assumptions rather than inventing
# new numbers.
GENERIC_SCENARIOS = {
    "normal": {"g1": 0.08, "g2": 0.08, "exit_multiple": 20.0},
    "best": {"g1": 0.10, "g2": 0.10, "exit_multiple": 25.0},
    "worst": {"g1": 0.04, "g2": 0.04, "exit_multiple": 12.0},
}


def classify_valuation_basis(
    eps_trailing: float | None,
    payout_ratio: float | None,
    sector: str | None,
    free_cash_flow: float | None,
) -> str:
    """Picks which metric to value, since a single metric can't meaningfully
    value both a bank and a pre-profit growth company. Evaluated in order:

    1. Real Estate sector -> "dividends" unconditionally, BEFORE the
       profitability check below - see REIT_SECTORS' comment for why
       REITs need a sector override rather than relying on payout_ratio or
       eps_trailing, both of which GAAP real-estate depreciation makes
       unreliable for this sector specifically (confirmed live: Aedifica,
       a real REIT, would otherwise have been misrouted).
    2. Unprofitable or unknown profitability -> "revenue" (can't project
       earnings/FCF/dividends that don't exist yet - matches early-stage
       growth companies like Beyond Meat).
    3. High payout ratio (pays out a large share of earnings) -> "dividends"
       (mature cash-cow-style payers - matches Kinder Morgan/Coca-Cola
       Europacific-style examples).
    4. Asset-heavy sector AND a real positive FCF figure -> "fcf" (matches
       industrial/asset-heavy examples). The FCF check isn't redundant with
       the sector check - confirmed live that yfinance's freeCashflow is
       None for banks (they don't have a meaningful FCF in the standard
       sense), and it can also be negative/None for an asset-heavy company
       mid capex-spike - both fall through to EPS rather than crashing or
       producing a nonsense basis.
    5. Otherwise -> "eps" (profitable, low payout, not asset-heavy - most
       tech/platform/growth names, and the fallback for financials, whose
       sector is never in ASSET_HEAVY_SECTORS)."""
    if sector in REIT_SECTORS:
        return "dividends"
    if eps_trailing is None or eps_trailing <= 0:
        return "revenue"
    if payout_ratio is not None and payout_ratio >= DIVIDEND_PAYOUT_THRESHOLD:
        return "dividends"
    if sector in ASSET_HEAVY_SECTORS and free_cash_flow is not None and free_cash_flow > 0:
        return "fcf"
    return "eps"


def _shares_outstanding_approx(market_cap: float | None, price: float | None) -> float | None:
    # market_cap = price x shares_outstanding by yfinance's own construction,
    # so this is an exact derivation, not an estimate - avoids fetching a
    # separate sharesOutstanding field.
    if not market_cap or not price:
        return None
    return market_cap / price


def cash_flow_basis_value(basis: str, fundamentals: dict) -> float | None:
    """Extracts the per-share cash-flow figure for the classified basis.
    "eps" and "dividends" are already per-share in yfinance's data; "fcf"
    and "revenue" are company totals divided down via the shares
    approximation above. None (not a fetch failure - a "this basis's input
    isn't usable right now" signal) propagates to intrinsic_value below,
    which renders it as "Not applicable", the same fail-soft convention
    the old Graham Number implementation used for negative EPS/book value.
    """
    if basis == "eps":
        return fundamentals.get("eps_trailing")
    if basis == "dividends":
        return fundamentals.get("dividend_rate")

    shares = _shares_outstanding_approx(fundamentals.get("market_cap"), fundamentals.get("price"))
    if not shares:
        return None
    if basis == "revenue":
        revenue = fundamentals.get("total_revenue")
        return revenue / shares if revenue else None
    if basis == "fcf":
        fcf = fundamentals.get("free_cash_flow")
        return fcf / shares if fcf else None
    return None


def build_scenarios(ticker: str | None, fundamentals: dict) -> dict[str, dict]:
    """Assembles this company's normal/best/worst g1/g2/exit_multiple/
    probability. Curated tickers (CURATED_SCENARIOS) use the analyst's
    exact numbers verbatim. Everyone else starts from GENERIC_SCENARIOS and,
    if a same-direction 0y/+1y consensus growth estimate is available (see
    fundamentals.fetch_fundamentals' growth_0y/growth_1y/growth_0y_low/
    growth_0y_high), overrides just g1 per scenario - g2/exit_multiple stay
    generic regardless (see GENERIC_SCENARIOS' comment for why).

    "Same-direction" is the reliability gate: confirmed live that when 0y
    and +1y consensus growth point in OPPOSITE directions (e.g. XOM's
    +65.7% this year / -8.6% next year), that's not a real growth trend -
    it's a rebound-then-giveback around a distorted (commodity-cycle,
    one-off) base year, and no combination of those two numbers recovers
    the analyst's actual 4% long-run assumption. Falling back to the
    generic g1 in that case is a deliberate "don't know" rather than a
    confidently wrong derived number - same fail-soft philosophy as the
    rest of this module.
    """
    if ticker and ticker in CURATED_SCENARIOS:
        return {
            name: {**scenario, "probability": SCENARIO_PROBABILITY}
            for name, scenario in CURATED_SCENARIOS[ticker].items()
        }

    scenarios = {
        name: {**scenario, "probability": SCENARIO_PROBABILITY} for name, scenario in GENERIC_SCENARIOS.items()
    }

    growth_0y = fundamentals.get("growth_0y")
    growth_1y = fundamentals.get("growth_1y")
    consensus_reliable = growth_0y is not None and growth_1y is not None and (growth_0y >= 0) == (growth_1y >= 0)
    if not consensus_reliable:
        return scenarios

    scenarios["normal"]["g1"] = (growth_0y + growth_1y) / 2
    growth_0y_high = fundamentals.get("growth_0y_high")
    growth_0y_low = fundamentals.get("growth_0y_low")
    if growth_0y_high is not None:
        scenarios["best"]["g1"] = growth_0y_high
    if growth_0y_low is not None:
        scenarios["worst"]["g1"] = growth_0y_low
    return scenarios


def scenario_dcf_value(cf0: float, g1: float, g2: float, exit_multiple: float, discount_rate: float) -> float:
    """Present value of ONE scenario, summing every projected year's cash
    flow PLUS the discounted terminal value: cf0 compounds at g1 for
    STAGE_1_YEARS, then at g2 for STAGE_2_YEARS, each year's cash flow
    discounted back at discount_rate; the terminal value (final year's cash
    flow x exit_multiple) is discounted back from the same final year.
    Used only for the "dividends" basis (see scenario_present_values) -
    dividends are real cash actually paid to the shareholder every year, so
    summing the interim stream is correct there, unlike EPS/FCF/revenue
    (see scenario_terminal_value). cf0 must be positive - callers
    (intrinsic_value) are responsible for that check, matching this
    module's existing convention of validating inputs at the boundary
    rather than inside the pure math."""
    pv = 0.0
    cf = cf0
    for year in range(1, STAGE_1_YEARS + 1):
        cf *= 1 + g1
        pv += cf / (1 + discount_rate) ** year
    for year in range(STAGE_1_YEARS + 1, STAGE_1_YEARS + STAGE_2_YEARS + 1):
        cf *= 1 + g2
        pv += cf / (1 + discount_rate) ** year
    terminal_value = cf * exit_multiple
    pv += terminal_value / (1 + discount_rate) ** (STAGE_1_YEARS + STAGE_2_YEARS)
    return pv


def scenario_terminal_value(cf0: float, g1: float, g2: float, exit_multiple: float, discount_rate: float) -> float:
    """Present value of ONE scenario counting ONLY the discounted terminal
    value - no interim-year summation. Used for "eps"/"fcf"/"revenue"
    bases: projected EPS/FCF/revenue isn't cash actually paid to the
    shareholder each year (unlike a dividend), so a shareholder's real
    return comes from eventually selling at the projected year-10 price,
    not from "receiving" ten years of paper earnings on top of that sale.
    Summing both (scenario_dcf_value's approach) double-counts, which was
    confirmed live to be the single largest source of error in the
    previous flat model - see module history, point 3b."""
    future_cf = cf0 * (1 + g1) ** STAGE_1_YEARS * (1 + g2) ** STAGE_2_YEARS
    terminal_value = future_cf * exit_multiple
    return terminal_value / (1 + discount_rate) ** (STAGE_1_YEARS + STAGE_2_YEARS)


def _scenario_pv(basis: str, cf0: float, g1: float, g2: float, exit_multiple: float, discount_rate: float) -> float:
    if basis == "dividends":
        return scenario_dcf_value(cf0, g1, g2, exit_multiple, discount_rate)
    return scenario_terminal_value(cf0, g1, g2, exit_multiple, discount_rate)


def scenario_present_values(cf0: float, basis: str, scenarios: dict[str, dict]) -> dict[str, float]:
    """PV per named scenario in `scenarios` (see build_scenarios), using the
    basis-appropriate formula (see _scenario_pv). Factored out of
    intrinsic_value so the DCF formula is applied exactly once per scenario
    in exactly one place - both intrinsic_value's probability-weighting and
    valuation_block_for's logging build on this same dict rather than
    recomputing or duplicating it."""
    return {
        name: _scenario_pv(basis, cf0, scenario["g1"], scenario["g2"], scenario["exit_multiple"], DISCOUNT_RATE)
        for name, scenario in scenarios.items()
    }


def intrinsic_value(cf0: float | None, basis: str, scenarios: dict[str, dict]) -> float | None:
    """Probability-weighted intrinsic value across `scenarios` (see
    build_scenarios), using the basis-appropriate DCF formula (see
    _scenario_pv). None (not a fetch failure) when cf0 is missing or
    non-positive - the classified basis's own metric isn't usable for this
    company right now (e.g. a company just barely flipped profitable
    enough to avoid the "revenue" fallback but has near-zero EPS), same
    "Not applicable" semantics as the old Graham Number's negative-EPS
    case."""
    if cf0 is None or cf0 <= 0:
        return None

    pvs = scenario_present_values(cf0, basis, scenarios)
    return sum(scenario["probability"] * pvs[name] for name, scenario in scenarios.items())


def valuation_block(price: float | None, intrinsic: float | None, basis: str) -> str:
    """Renders the 'Valuation' prompt block from an already-computed
    intrinsic value (see intrinsic_value). intrinsic=None means the
    classified basis's inputs aren't usable for this company right now - a
    real, expected outcome, not a fetch failure, so it renders as 'Not
    applicable', not 'Data unavailable.'. price=None (fundamentals fetch
    failed entirely) is the actual unavailable case. The basis label (e.g.
    "FCF-based") is always shown so a reader - human or model - can learn
    that the metric being valued differs across companies, not just the
    number.
    """
    label = BASIS_LABELS[basis]
    if intrinsic is None:
        return f"Not applicable (insufficient data for the {label.lower()} valuation basis)."
    if price is None:
        return "Data unavailable."

    pct = (price - intrinsic) / intrinsic * 100
    verdict = "overvalued" if pct >= 0 else "undervalued"
    return (
        f"Intrinsic Value ({label}): ${intrinsic:.2f}\n"
        f"vs Current Price: {verdict} by ~{abs(pct):.0f}%"
    )


def _log_valuation_computation(ticker, fundamentals, basis, cf0, scenarios, intrinsic, block):
    # INFO (not DEBUG) so it shows up by default under this app's existing
    # logging.basicConfig(level=logging.INFO) (see main.py), matching
    # inference.py's prompt-logging convention - no config change needed to
    # see this in Render's log stream. Logs the full "recipe" (which
    # classification inputs drove the basis choice, the actual cf0 used,
    # whether this ticker's scenarios are curated or derived/generic, every
    # scenario's growth/exit-multiple/PV, and the final weighted result) so
    # the formula's behavior can be audited per-ticker after the fact, not
    # just the one-line rendered block. Fires even when cf0/intrinsic end
    # up None (the "Not applicable" case) - that's exactly when knowing WHY
    # (which classification inputs were missing/unusable) is most useful,
    # not less.
    pvs = scenario_present_values(cf0, basis, scenarios) if cf0 is not None and cf0 > 0 else {}
    source = "curated" if ticker and ticker in CURATED_SCENARIOS else "derived/generic"
    scenario_summary = "; ".join(
        f"{name}(p={s['probability']:.0%}, g1={s['g1']:.1%}, g2={s['g2']:.1%}, exit={s['exit_multiple']:.1f}x)"
        + (f" -> PV=${pvs[name]:,.2f}" if name in pvs else "")
        for name, s in scenarios.items()
    )
    logger.info(
        "Valuation[%s]: classification inputs eps_trailing=%s, payout_ratio=%s, "
        "sector=%r, free_cash_flow=%s -> basis=%s; cf0=%s; discount_rate=%.0f%%; "
        "scenarios(%s): %s; intrinsic_value=%s; price=%s; block=%r",
        ticker,
        fundamentals.get("eps_trailing"), fundamentals.get("payout_ratio"),
        fundamentals.get("sector"), fundamentals.get("free_cash_flow"), basis,
        cf0, DISCOUNT_RATE * 100, source, scenario_summary, intrinsic, fundamentals.get("price"),
        block,
    )


def valuation_block_for(fundamentals: dict | None, ticker: str | None = None) -> str:
    """Convenience wrapper for callers holding a fundamentals.fetch_
    fundamentals() result - classifies the valuation basis, builds this
    company's scenarios (see build_scenarios), computes the intrinsic
    value, and renders the block in one call. A missing/failed fundamentals
    fetch (or a missing price specifically) renders as 'Data unavailable.'
    before classification is even attempted, since none of its inputs
    would be trustworthy either. `ticker` is optional - without it,
    build_scenarios can never match CURATED_SCENARIOS and always falls
    back to derived/generic, and the audit log below just omits the label;
    callers without it (e.g. existing tests) still work.
    """
    if not fundamentals or fundamentals.get("price") is None:
        return "Data unavailable."

    basis = classify_valuation_basis(
        fundamentals.get("eps_trailing"),
        fundamentals.get("payout_ratio"),
        fundamentals.get("sector"),
        fundamentals.get("free_cash_flow"),
    )
    cf0 = cash_flow_basis_value(basis, fundamentals)
    scenarios = build_scenarios(ticker, fundamentals)
    intrinsic = intrinsic_value(cf0, basis, scenarios)
    block = valuation_block(fundamentals["price"], intrinsic, basis)
    _log_valuation_computation(ticker, fundamentals, basis, cf0, scenarios, intrinsic, block)
    return block
