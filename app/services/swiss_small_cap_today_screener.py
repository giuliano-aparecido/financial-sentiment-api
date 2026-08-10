"""
Swiss small-cap "big loss, thin volume" screener - TODAY only.

Finds SIX Swiss Exchange-listed, Switzerland-domiciled small-cap stocks
that are down >= LOSS_THRESHOLD_PCT today AND trading on unusually LOW
volume relative to their own norm - i.e. a big price move without much
actual trading behind it, which can mean a stale/wide quote or a thin
order-book gap rather than real, broad selling pressure.

Ported from the standalone research/ project (D:\\projects\\research) - see
swiss_small_cap_crash_rebound.py's module docstring for why this now runs
as a FastAPI background job (app/services/research_job.py) instead of a
Colab/Kaggle notebook.

Free tools only: yfinance's public screener (no API key). Universe
discovery (market-cap band + domicile filter) lives in swiss_universe.py,
shared with swiss_small_cap_crash_rebound.py.

Unlike that other script, this one needs no separate price-history
download: the screener's own live quote already carries today's change%,
today's volume, and the 3-month average volume to compare it against
(confirmed live - see swiss_universe.discover_candidates' docstring), so
discovery and "today's numbers" come from the exact same call - run_scan()
below just reads them off the already-discovered/filtered `domestic` dict.

"Small volume" is intentionally NOT a hard filter with an invented cutoff
- it's a SORT key (volume_vs_3mo_avg ascending), so every qualifying loss
is shown and you can see for yourself where "small" starts. Set
MAX_VOLUME_RATIO below if you want a hard cutoff instead.
"""

import pandas as pd

# --- Config ---

LOSS_THRESHOLD_PCT = -5.0  # today's regularMarketChangePercent <= this

# None = no hard cutoff, just sort by volume ratio ascending (see module
# docstring). Set e.g. 1.0 to only keep rows trading below their own
# 3-month average volume today.
MAX_VOLUME_RATIO = None


def find_big_loss_thin_volume(domestic, loss_threshold, max_volume_ratio):
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
        if max_volume_ratio is not None and (volume_ratio is None or volume_ratio > max_volume_ratio):
            continue

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
    # "Ordered by volume and loss" - smallest volume ratio first (the
    # "thin volume" signal this script is specifically for), biggest loss
    # first as the tiebreaker within similarly-thin names.
    return results.sort_values(
        ["volume_vs_3mo_avg", "change_pct"], ascending=[True, True]
    ).reset_index(drop=True)


def run_scan(domestic: dict) -> pd.DataFrame:
    """Runs the full today-only big-loss/thin-volume scan against an
    already-discovered/domicile-filtered `domestic` dict (see
    swiss_universe.filter_domestic) - see this module's docstring for why
    no separate fetch is needed here. Returns a DataFrame, empty if no
    matches today.
    """
    return find_big_loss_thin_volume(domestic, LOSS_THRESHOLD_PCT, MAX_VOLUME_RATIO)
