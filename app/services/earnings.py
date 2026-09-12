import datetime
import logging
import math

import yfinance as yf

from app.services.fundamentals import format_market_cap, resolve_ticker

logger = logging.getLogger(__name__)


def _fetch_quarterly_statements(ticker: str):
    """(income, earnings_dates) if the ticker resolves to real quarterly
    data, else None - factored out so fetch_earnings can retry once with a
    resolve_ticker()-corrected symbol, same pattern as
    fundamentals._fetch_price_info (see that function's docstring for why
    this retry is needed for most non-US listings)."""
    try:
        t = yf.Ticker(ticker)
        income = t.quarterly_income_stmt
        earnings_dates = t.earnings_dates
    except Exception as e:
        logger.warning("yfinance earnings fetch failed for %s: %s", ticker, e)
        return None
    if income is None or income.empty or "Total Revenue" not in income.index:
        return None
    return income, earnings_dates


def fetch_earnings(ticker: str) -> dict | None:
    """Fetches the most recent reported quarter's revenue/EPS (with YoY
    growth and beat/miss/in-line vs estimate, where available) and the next
    scheduled earnings date. Returns a plain dict of already-extracted
    primitive values, not a raw yfinance/pandas object - so
    earnings_block() below can be unit-tested with canned dicts, matching
    this project's existing test style, with no live call or pandas
    dependency needed in the test file. Returns None on total fetch
    failure; individual missing pieces (e.g. no YoY comparator available,
    no scheduled next date) are represented as None inside the dict rather
    than failing the whole fetch.

    If the bare ticker doesn't resolve, retries once via resolve_ticker
    (e.g. "NESN" -> "NESN.SW") - this runs independently of
    fundamentals.fetch_fundamentals' own resolution (the two are fetched
    concurrently via asyncio.gather in analyze.py, so neither can reuse the
    other's result), which just means the resolve_ticker lookup happens
    twice on the fallback path - an acceptable cost since both fetches
    still run in parallel threads, not one blocking the other.
    """
    statements = _fetch_quarterly_statements(ticker)
    if statements is None:
        resolved_ticker = resolve_ticker(ticker)
        if resolved_ticker != ticker:
            statements = _fetch_quarterly_statements(resolved_ticker)
    if statements is None:
        return None
    income, earnings_dates = statements

    revenue_info = _extract_revenue_and_growth(income)
    if revenue_info is None:
        return None
    last_period, revenue, yoy_growth_pct = revenue_info

    eps_actual, eps_estimate, next_earnings_date = _extract_eps_fields(earnings_dates)

    return {
        "last_quarter_date": last_period.date().isoformat(),
        "revenue": revenue,
        "yoy_growth_pct": yoy_growth_pct,
        "eps_actual": eps_actual,
        "eps_estimate": eps_estimate,
        "next_earnings_date": next_earnings_date,
    }


def _extract_revenue_and_growth(income):
    """(last_period, revenue, yoy_growth_pct) for the most recent reported
    quarter in `income` (see fetch_earnings), or None if there's no usable
    revenue row at all. yoy_growth_pct is None when no quarter ~365 days
    earlier exists to compare against."""
    revenue_row = income.loc["Total Revenue"].dropna()
    if revenue_row.empty:
        return None

    last_period = revenue_row.index.max()
    revenue = float(revenue_row[last_period])

    yoy_growth_pct = None
    prior_year_period = last_period - datetime.timedelta(days=365)
    candidates = revenue_row[(revenue_row.index - prior_year_period).map(lambda d: abs(d.days)) < 20]
    if not candidates.empty:
        prior_revenue = float(candidates.iloc[0])
        if prior_revenue:
            yoy_growth_pct = (revenue - prior_revenue) / prior_revenue * 100

    return last_period, revenue, yoy_growth_pct


def _extract_eps_fields(earnings_dates):
    """(eps_actual, eps_estimate, next_earnings_date) from `earnings_dates`
    (see fetch_earnings) - eps_actual/eps_estimate come from the most
    recent PAST row with a reported EPS; next_earnings_date is the
    earliest FUTURE row's date, independent of whether that row has EPS
    data yet (it never would, being in the future)."""
    eps_actual, eps_estimate, next_earnings_date = None, None, None
    if earnings_dates is None or earnings_dates.empty:
        return eps_actual, eps_estimate, next_earnings_date

    now = datetime.datetime.now(datetime.timezone.utc)
    if earnings_dates.index.tz is None:
        now = now.replace(tzinfo=None)

    if "Reported EPS" in earnings_dates.columns:
        reported = earnings_dates.dropna(subset=["Reported EPS"])
        past = reported[reported.index < now]
        if not past.empty:
            row = past.sort_index(ascending=False).iloc[0]
            eps_actual = row.get("Reported EPS")
            eps_estimate = row.get("EPS Estimate")

    future = earnings_dates[earnings_dates.index > now]
    if not future.empty:
        next_earnings_date = future.sort_index().index.min().date().isoformat()

    return eps_actual, eps_estimate, next_earnings_date


def earnings_block(earnings: dict | None) -> str:
    """Renders the 'Recent Earnings' prompt block. Byte-format matches
    financial-sentiment-model's dataset generators exactly (see that
    repo's CONTRIBUTING.md 4-way sync rule) - the model is trained on this
    shape.
    """
    if not earnings or earnings.get("revenue") is None:
        return "Data unavailable."

    yoy = earnings.get("yoy_growth_pct")
    yoy_str = f" ({'+' if yoy >= 0 else ''}{yoy:.1f}% YoY)" if yoy is not None else ""

    eps_note = ""
    actual, estimate = earnings.get("eps_actual"), earnings.get("eps_estimate")
    if actual is not None and estimate:
        # abs_tol=0.005 (half a cent) rather than exact equality - actual/estimate
        # are floats from yfinance, and values that display as the same 2-decimal
        # EPS shouldn't be reported as a "beat"/"miss" over float noise.
        if math.isclose(actual, estimate, abs_tol=0.005):
            eps_note = f", EPS ${actual:.2f} (in line with est. ${estimate:.2f})"
        elif actual > estimate:
            eps_note = f", EPS ${actual:.2f} (beat est. ${estimate:.2f})"
        else:
            eps_note = f", EPS ${actual:.2f} (missed est. ${estimate:.2f})"

    next_line = ""
    if earnings.get("next_earnings_date"):
        next_line = f"\nNext Earnings Date: {earnings['next_earnings_date']}"

    return (
        f"Last Quarter ({earnings['last_quarter_date']}): "
        f"Revenue {format_market_cap(earnings['revenue'])}{yoy_str}{eps_note}"
        f"{next_line}"
    )
