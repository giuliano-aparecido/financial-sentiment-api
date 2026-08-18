"""
Swiss "indicator of volatility" scanner.

For each SIX Swiss Exchange-listed, Switzerland-domiciled company (same
universe as swiss_crash_rebound.py - market cap > CHF 500M, SMI's 20
largest excluded - see swiss_universe.py's MIN/MAX_MARKET_CAP_CHF and
SMI_TICKERS), counts how many trading days in the last LOOKBACK_MONTHS
closed down >= `threshold_pct` and how many closed up >= `threshold_pct`
(close-to-close, same day-over-day % change swiss_crash_rebound.py
already computes). Unlike that scan, this isn't looking for a specific
loss-then-rebound EVENT - it's a blunter "how often does this stock move
by at least this much in either direction" frequency count, one row per
company (not one row per event), added at the user's explicit request as
a third, independently-triggerable table alongside crash-rebound and
today's-big-losers.

`threshold_pct` is user-selectable in the frontend (2%/3%/5% - see
ALLOWED_THRESHOLD_PCTS) rather than fixed, unlike DROP_THRESHOLD_PCT/
GAIN_THRESHOLD_PCT in swiss_crash_rebound.py which are fixed constants -
this scan's whole point is comparing frequency across different bar
heights, so the threshold has to be a request-time parameter, not a
module constant.

Deliberately does its OWN independent universe discovery (discover_
candidates()/filter_domestic()) rather than reusing the `domestic` dict
the crash-rebound/today-screener scan already computed - see
research_job.py's start_indicator_scan for why: this table has its own
independent refresh button/threshold selector, and sharing a cached
`domestic` across scan types would either (a) let this table's discovery
go stale between the OTHER two tables' own refreshes, or (b) require
invalidating a cache today_screener's own tests already depend on
re-discovering fresh on every Refresh (see that module's docstring on
"refreshed on demand via the button, or naturally on the next day").
Simpler and safer to pay the ~30-60s/~150-live-call discovery cost again
on this table's OWN refresh, same bounded per-click cost the OTHER
table's own Refresh already has, than to risk that guarantee.

Reuses swiss_crash_rebound.py's download_ohlcv_chunked (identical
chunking/pacing, not a second copy - see that function's own docstring)
for the batch 12-month price-history download.
"""

import pandas as pd

from app.services.swiss_crash_rebound import download_ohlcv_chunked

# --- Config ---

LOOKBACK_MONTHS = 12
# Same 1-month buffer reasoning as swiss_crash_rebound.py's HISTORY_PERIOD
# - the first in-window day still needs a valid previous-close.
HISTORY_PERIOD = "13mo"

# The only threshold values the frontend's selector offers - validated
# server-side too (not just trusted from the request) since an arbitrary
# caller-supplied float would still run a real ~150-call live scan either
# way; there's no reason to accept a value the UI never actually offers.
ALLOWED_THRESHOLD_PCTS = (2.0, 3.0, 5.0)


def find_volatility_days(symbols, domestic, lookback_months, history_period, threshold_pct):
    """Returns a DataFrame with one row per company that had AT LEAST ONE
    qualifying loss day AND at least one qualifying gain day in the
    lookback window - a company with only losses (or only gains) is
    omitted, not shown with a 0 in one column (changed 2026-08-19 at the
    user's explicit request: "It should show only companies with loss
    and gains, if loss or gain is 0 then it should not be in the table" -
    previously any company with EITHER at least one qualifying day of
    EITHER kind was included, which is a looser bar than what was
    actually wanted). Columns: ticker, name, sector, market_cap,
    loss_days (count of days closing down >= threshold_pct - i.e. a
    LOSS of at least threshold_pct), gain_days (count of days closing up
    >= threshold_pct), total_days (loss_days + gain_days). Sorted by
    total_days descending - the most volatile names first. NaN skipped
    naturally: a NaN day (a ticker's own first available trading day,
    see swiss_crash_rebound.py's identically-shaped bug/fix for the full
    story) compares False against both `<=` and `>=` here, the same way
    it silently passed the OLD unguarded check there - but here that's
    actually safe and correct as-is, since neither comparison being
    True just correctly excludes that one day from both counts, it
    doesn't fabricate a false match the way a single unguarded `>`
    comparison did in that other module.
    """
    if not symbols:
        return pd.DataFrame()

    data = download_ohlcv_chunked(symbols, history_period)

    cutoff = pd.Timestamp.today(tz=data.index.tz) - pd.DateOffset(months=lookback_months)

    rows = []
    for symbol in symbols:
        try:
            close = data[symbol]["Close"].dropna()
        except KeyError:
            continue
        if close.empty:
            continue

        pct_change = close.pct_change() * 100
        windowed = pct_change[pct_change.index >= cutoff]

        loss_days = int((windowed <= -threshold_pct).sum())
        gain_days = int((windowed >= threshold_pct).sum())
        if loss_days == 0 or gain_days == 0:
            continue

        rows.append({
            "ticker": symbol,
            "name": domestic[symbol]["name"],
            "sector": domestic[symbol]["sector"],
            "market_cap": domestic[symbol]["market_cap"],
            "loss_days": loss_days,
            "gain_days": gain_days,
            "total_days": loss_days + gain_days,
        })

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    return results.sort_values("total_days", ascending=False).reset_index(drop=True)


def run_scan(domestic: dict, threshold_pct: float) -> pd.DataFrame:
    """Runs the full volatility-indicator scan against an already-
    discovered/domicile-filtered `domestic` dict (see swiss_universe.
    filter_domestic). Returns a DataFrame, empty if no company had a
    qualifying day - never raises for "no results," only for a genuine
    fetch failure.
    """
    return find_volatility_days(
        list(domestic.keys()), domestic, LOOKBACK_MONTHS, HISTORY_PERIOD, threshold_pct,
    )
