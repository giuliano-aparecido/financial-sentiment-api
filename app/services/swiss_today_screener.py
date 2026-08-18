"""
Swiss "big loss" volatility screener - TODAY only.

Finds SIX Swiss Exchange-listed, Switzerland-domiciled stocks that are
down >= LOSS_THRESHOLD_PCT today. Universe: SIX-listed, Switzerland-
domiciled companies with market cap > CHF 500M (SMI's 20 largest still
excluded) - see swiss_crash_rebound.py's module docstring for the same
framing, which applies here identically since both scans share the same
universe discovery. This module's OWN logic does no volume filtering of
its own - volume vs. each stock's own 10-day average is shown and used to
SORT the results (thinnest first), but never excludes a row here: a big
loss on unusually thin volume vs. one on heavy volume tell different
stories, and both are worth seeing, not just one of them. (A separate,
absolute liquidity floor IS applied upstream, in
swiss_universe.filter_domestic - see MIN_AVG_DAILY_VOLUME_10D - that's
excluding unreliably-thin prints entirely, a different concern from this
module's own thin-vs-heavy sort.)

Ported from the standalone research/ project (D:\\projects\\research) - see
swiss_crash_rebound.py's module docstring for why this now runs
as a FastAPI background job (app/services/research_job.py) instead of a
Colab/Kaggle notebook.

Free tools only: yfinance's public screener (no API key). Universe
discovery (market-cap band + domicile filter) lives in swiss_universe.py,
shared with swiss_crash_rebound.py.

Unlike that other script, this one needs no separate price-history
download: the screener's own live quote already carries today's change%
and today's volume, and swiss_universe.filter_domestic's own .info call
(already made per ticker for the domicile/liquidity check - see that
module's docstring) already carries the 10-day average volume to compare
today's volume against - avg_volume_10d, read straight off the
already-discovered/filtered `domestic` dict entry, not the screener quote
(changed 2026-08-18: previously read the screener's own
averageDailyVolume3Month field directly - switched to reuse the SAME
10-day figure swiss_crash_rebound.py now also uses, one volume-averaging
methodology for the whole feature instead of two). No extra fetch either
way - run_scan() below just reads everything off the already-fetched
`domestic` dict.
"""

import pandas as pd

# --- Config ---

LOSS_THRESHOLD_PCT = -5.0  # today's regularMarketChangePercent <= this


def find_big_loss(domestic: dict, loss_threshold: float) -> pd.DataFrame:
    """Reads today's change%/volume off each candidate's already-fetched
    screener quote, and 10-day average volume off the SAME already-
    fetched domestic entry (see swiss_universe.filter_domestic's
    avg_volume_10d) - no extra fetch needed either way. Skips any
    candidate missing a live change% or volume figure, which happens for
    very illiquid names with no trade yet today rather than being a fetch
    failure.
    """
    rows = []
    for symbol, entry in domestic.items():
        quote = entry["quote"]
        change_pct = quote.get("regularMarketChangePercent")
        volume_today = quote.get("regularMarketVolume")
        avg_volume_10d = entry.get("avg_volume_10d")

        if change_pct is None or volume_today is None:
            continue
        if change_pct > loss_threshold:
            continue

        volume_ratio = round(volume_today / avg_volume_10d, 2) if avg_volume_10d else None

        rows.append({
            "ticker": symbol,
            "name": entry["name"],
            "sector": entry["sector"],
            "market_cap": entry["market_cap"],
            "price": quote.get("regularMarketPrice"),
            "change_pct": round(change_pct, 2),
            "volume_today": int(volume_today),
            "avg_volume_10d": int(avg_volume_10d) if avg_volume_10d else None,
            "volume_vs_10d_avg": volume_ratio,
        })

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    # Reading aid, not a further filter (see module docstring) - thinnest
    # volume first, biggest loss as the tiebreaker within similarly-thin
    # names.
    return results.sort_values(
        ["volume_vs_10d_avg", "change_pct"], ascending=[True, True]
    ).reset_index(drop=True)


def run_scan(domestic: dict) -> pd.DataFrame:
    """Runs the full today big-loss scan against an already-discovered/
    domicile-filtered `domestic` dict (see swiss_universe.filter_domestic)
    - see this module's docstring for why no separate fetch is needed
    here. Returns a DataFrame, empty if nothing is down >= 5% today.
    """
    return find_big_loss(domestic, LOSS_THRESHOLD_PCT)
