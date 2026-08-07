import math


def graham_number(eps: float | None, book_value_per_share: float | None) -> float | None:
    """Benjamin Graham's conservative intrinsic-value estimate:
    sqrt(22.5 x EPS x book value/share). None for missing or non-positive
    inputs (a loss-making company or negative book value has no defined
    Graham Number - no sqrt of a negative). Deterministic, code-only math -
    never LLM-generated - matching financial-sentiment-model-colab's
    training-data generators (see that repo's CONTRIBUTING.md 4-way sync
    rule for why this needs to stay in step with them).
    """
    if eps is None or book_value_per_share is None:
        return None
    if eps <= 0 or book_value_per_share <= 0:
        return None
    return math.sqrt(22.5 * eps * book_value_per_share)


def valuation_block(price: float | None, intrinsic: float | None) -> str:
    """Renders the 'Valuation' prompt block from an already-computed
    intrinsic value (see graham_number). intrinsic=None means the Graham
    Number itself isn't defined for this company right now (negative/
    missing EPS or book value) - a real, expected outcome, not a fetch
    failure, so it renders as 'Not applicable', not 'Data unavailable.'.
    price=None (fundamentals fetch failed entirely) is the actual
    unavailable case.
    """
    if intrinsic is None:
        return "Not applicable (negative or missing EPS/book value)."
    if price is None:
        return "Data unavailable."

    pct = (price - intrinsic) / intrinsic * 100
    verdict = "overvalued" if pct >= 0 else "undervalued"
    return (
        f"Intrinsic Value (Graham Number): ${intrinsic:.2f}\n"
        f"vs Current Price: {verdict} by ~{abs(pct):.0f}%"
    )


def valuation_block_for(fundamentals: dict | None) -> str:
    """Convenience wrapper for callers holding a fundamentals.fetch_
    fundamentals() result - computes the Graham Number from it and renders
    the block in one call. A missing/failed fundamentals fetch (or a
    missing price specifically) renders as 'Data unavailable.' before
    graham_number is even consulted, since none of its inputs would be
    trustworthy either.
    """
    if not fundamentals or fundamentals.get("price") is None:
        return "Data unavailable."
    intrinsic = graham_number(fundamentals.get("eps_trailing"), fundamentals.get("book_value_per_share"))
    return valuation_block(fundamentals["price"], intrinsic)
