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
# SCENARIOS/DISCOUNT_RATE below match a specific, given reference
# calculation (universal across all four bases - the scenario assumptions
# don't vary by which metric is being projected, only the metric's own
# value does). Two earlier, self-invented alternatives were tried and
# rejected first, for the historical record:
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
# The constants below use a flat discount rate and wider Best/Worst exit-
# multiple spread instead, which resolved most of variant 2's outliers.
#
# Deterministic, code-only math - never LLM-generated - matching
# financial-sentiment-model-colab's training-data generators (see that
# repo's CONTRIBUTING.md 4-way sync rule for why the RENDERED BLOCK FORMAT
# needs to stay in step with them; the formula/constants below are not yet
# ported there - see this repo's PR history for the staged-rollout plan).

STAGE_1_YEARS = 5
STAGE_2_YEARS = 5

# Sector strings match yfinance's Ticker.info["sector"] values exactly.
ASSET_HEAVY_SECTORS = {"Energy", "Industrials", "Basic Materials", "Utilities"}
DIVIDEND_PAYOUT_THRESHOLD = 0.40  # payout_ratio >= this -> treated as a mature dividend payer

BASIS_LABELS = {
    "revenue": "Revenue-based",
    "eps": "EPS-based",
    "fcf": "FCF-based",
    "dividends": "Dividend-based",
}

# Flat for every company (not risk-adjusted per company) - deliberately
# simpler than a CAPM/beta-derived rate, and specifically what resolved the
# high-beta-name double-punishment problem described above.
DISCOUNT_RATE = 0.10

# g1 = years 1-5 growth, g2 = years 6-10 growth (equal within each scenario
# here - growth doesn't fade between stages in this model, unlike an
# earlier rejected variant), exit_multiple applied to year-10's projected
# cash flow for the terminal value. Universal across all four valuation
# bases - the scenario assumptions represent market-wide bull/base/bear
# conditions, not a per-metric-type judgment.
SCENARIOS = {
    "normal": {"probability": 0.60, "g1": 0.08, "g2": 0.08, "exit_multiple": 15.0},
    "best": {"probability": 0.20, "g1": 0.10, "g2": 0.10, "exit_multiple": 30.0},
    "worst": {"probability": 0.20, "g1": 0.04, "g2": 0.04, "exit_multiple": 10.0},
}


def classify_valuation_basis(
    eps_trailing: float | None,
    payout_ratio: float | None,
    sector: str | None,
    free_cash_flow: float | None,
) -> str:
    """Picks which metric to value, since a single metric can't meaningfully
    value both a bank and a pre-profit growth company. Evaluated in order:

    1. Unprofitable or unknown profitability -> "revenue" (can't project
       earnings/FCF/dividends that don't exist yet - matches early-stage
       growth companies like Beyond Meat).
    2. High payout ratio (pays out a large share of earnings) -> "dividends"
       (mature cash-cow/REIT-style payers - matches Coca-Cola/Exxon-style
       examples).
    3. Asset-heavy sector AND a real positive FCF figure -> "fcf" (matches
       industrial/asset-heavy examples). The FCF check isn't redundant with
       the sector check - confirmed live that yfinance's freeCashflow is
       None for banks (they don't have a meaningful FCF in the standard
       sense), and it can also be negative/None for an asset-heavy company
       mid capex-spike - both fall through to EPS rather than crashing or
       producing a nonsense basis.
    4. Otherwise -> "eps" (profitable, low payout, not asset-heavy - most
       tech/platform/growth names, and the fallback for financials, whose
       sector is never in ASSET_HEAVY_SECTORS)."""
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


def scenario_dcf_value(cf0: float, g1: float, g2: float, exit_multiple: float, discount_rate: float) -> float:
    """Present value of ONE scenario: cf0 compounds at g1 for
    STAGE_1_YEARS, then at g2 for STAGE_2_YEARS, each year's cash flow
    discounted back at discount_rate; the terminal value (final year's cash
    flow x exit_multiple) is discounted back from the same final year.
    cf0 must be positive - callers (intrinsic_value) are responsible for
    that check, matching this module's existing convention of validating
    inputs at the boundary rather than inside the pure math."""
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


def intrinsic_value(cf0: float | None) -> float | None:
    """Probability-weighted intrinsic value across SCENARIOS, applied
    directly to cf0 - the scenario assumptions are universal across all
    four valuation bases (see module docstring), so this takes no basis
    argument, unlike an earlier per-basis-tiered version. None (not a fetch
    failure) when cf0 is missing or non-positive - the classified basis's
    own metric isn't usable for this company right now (e.g. a company
    just barely flipped profitable enough to avoid the "revenue" fallback
    but has near-zero EPS), same "Not applicable" semantics as the old
    Graham Number's negative-EPS case."""
    if cf0 is None or cf0 <= 0:
        return None

    weighted_total = 0.0
    for scenario in SCENARIOS.values():
        pv = scenario_dcf_value(cf0, scenario["g1"], scenario["g2"], scenario["exit_multiple"], DISCOUNT_RATE)
        weighted_total += scenario["probability"] * pv
    return weighted_total


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


def valuation_block_for(fundamentals: dict | None) -> str:
    """Convenience wrapper for callers holding a fundamentals.fetch_
    fundamentals() result - classifies the valuation basis, computes the
    intrinsic value, and renders the block in one call. A missing/failed
    fundamentals fetch (or a missing price specifically) renders as 'Data
    unavailable.' before classification is even attempted, since none of
    its inputs would be trustworthy either.
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
    intrinsic = intrinsic_value(cf0)
    return valuation_block(fundamentals["price"], intrinsic, basis)
