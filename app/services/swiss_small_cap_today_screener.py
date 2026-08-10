"""
Swiss small-cap "today" snapshot.

Lists EVERY SIX Swiss Exchange-listed, Switzerland-domiciled small-cap
stock's today numbers (price, change%, volume, volume vs. its own
3-month average) - no loss or volume filtering, by design: this is meant
to show the full universe so you can read the whole picture yourself,
not a pre-filtered subset. Sorted by volume vs. 3-month average
(thinnest first, then biggest move) purely as a reading aid - a stock
near the top of the list is trading unusually thin today, which is
useful context whether it's up, down, or flat, not a criterion for
being included at all.

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
"""

import pandas as pd


def build_today_snapshot(domestic: dict) -> pd.DataFrame:
    """Reads today's change%/volume/3-month-average-volume straight off
    each candidate's already-fetched screener quote (see
    swiss_universe.filter_domestic) - no extra fetch needed. Skips any
    candidate missing a live change% or volume figure, which happens for
    very illiquid names with no trade yet today rather than being a fetch
    failure - there's no meaningful "today" row to show for those, not a
    "doesn't qualify" exclusion.
    """
    rows = []
    for symbol, entry in domestic.items():
        quote = entry["quote"]
        change_pct = quote.get("regularMarketChangePercent")
        volume_today = quote.get("regularMarketVolume")
        avg_volume_3mo = quote.get("averageDailyVolume3Month")

        if change_pct is None or volume_today is None:
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
    # Reading aid only, not a filter (see module docstring) - thinnest
    # volume first, biggest move as the tiebreaker within similarly-thin
    # names.
    return results.sort_values(
        ["volume_vs_3mo_avg", "change_pct"], ascending=[True, True]
    ).reset_index(drop=True)


def run_scan(domestic: dict) -> pd.DataFrame:
    """Runs the full today snapshot against an already-discovered/
    domicile-filtered `domestic` dict (see swiss_universe.filter_domestic)
    - see this module's docstring for why no separate fetch is needed
    here. Returns a DataFrame covering every ticker with a live quote
    today, empty only if none of them do.
    """
    return build_today_snapshot(domestic)
