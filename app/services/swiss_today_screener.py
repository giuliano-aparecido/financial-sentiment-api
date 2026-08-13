"""
Swiss "big loss" volatility screener - TODAY only.

Finds SIX Swiss Exchange-listed, Switzerland-domiciled stocks that are
down >= LOSS_THRESHOLD_PCT today. Universe defaults to a small-cap band
but can widen to include mid/large caps too (SMI's 20 largest excluded
either way) - see swiss_crash_rebound.py's module docstring for the same
volatility-vs-small-cap framing, which applies here identically since
both scans share the same universe discovery. This module's OWN logic
does no volume filtering of its own - volume vs. each stock's own
3-month average is shown and used to SORT the results (thinnest first),
but never excludes a row here: a big loss on unusually thin volume vs.
one on heavy volume tell different stories, and both are worth seeing,
not just one of them. (A separate, absolute liquidity floor IS applied
upstream, in swiss_universe.filter_domestic - see MIN_INTRADAY_VOLUME -
that's excluding unreliably-thin prints entirely, a different concern
from this module's own thin-vs-heavy sort.)

Ported from the standalone research/ project (D:\\projects\\research) - see
swiss_crash_rebound.py's module docstring for why this now runs
as a FastAPI background job (app/services/research_job.py) instead of a
Colab/Kaggle notebook.

Free tools only: yfinance's public screener (no API key). Universe
discovery (market-cap band + domicile filter) lives in swiss_universe.py,
shared with swiss_crash_rebound.py.

Unlike that other script, this one needs no separate price-history
download: the screener's own live quote already carries today's change%,
today's volume, and the 3-month average volume to compare it against
(confirmed live - see swiss_universe.discover_candidates' docstring), so
discovery and "today's numbers" come from the exact same call - run_scan()
below just reads them off the already-discovered/filtered `domestic` dict.
"""

import pandas as pd

# --- Config ---

LOSS_THRESHOLD_PCT = -5.0  # today's regularMarketChangePercent <= this


def find_big_loss(domestic: dict, loss_threshold: float) -> pd.DataFrame:
    """Reads today's change%/volume/3-month-average-volume straight off
    each candidate's already-fetched screener quote (see
    swiss_universe.filter_domestic) - no extra fetch needed. Skips any
    candidate missing a live change% or volume figure, which happens for
    very illiquid names with no trade yet today rather than being a fetch
    failure.
    """
    rows = []
    for symbol, entry in domestic.items():
        quote = entry["quote"]
        change_pct = quote.get("regularMarketChangePercent")
        volume_today = quote.get("regularMarketVolume")
        avg_volume_3mo = quote.get("averageDailyVolume3Month")

        if change_pct is None or volume_today is None:
            continue
        if change_pct > loss_threshold:
            continue

        volume_ratio = round(volume_today / avg_volume_3mo, 2) if avg_volume_3mo else None

        rows.append({
            "ticker": symbol,
            "name": entry["name"],
            "sector": entry["sector"],
            "market_cap": entry["market_cap"],
            "price": quote.get("regularMarketPrice"),
            "change_pct": round(change_pct, 2),
            "volume_today": int(volume_today),
            "avg_volume_3mo": int(avg_volume_3mo) if avg_volume_3mo else None,
            "volume_vs_3mo_avg": volume_ratio,
        })

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    # Reading aid, not a further filter (see module docstring) - thinnest
    # volume first, biggest loss as the tiebreaker within similarly-thin
    # names.
    return results.sort_values(
        ["volume_vs_3mo_avg", "change_pct"], ascending=[True, True]
    ).reset_index(drop=True)


def run_scan(domestic: dict) -> pd.DataFrame:
    """Runs the full today big-loss scan against an already-discovered/
    domicile-filtered `domestic` dict (see swiss_universe.filter_domestic)
    - see this module's docstring for why no separate fetch is needed
    here. Returns a DataFrame, empty if nothing is down >= 5% today.
    """
    return find_big_loss(domestic, LOSS_THRESHOLD_PCT)
