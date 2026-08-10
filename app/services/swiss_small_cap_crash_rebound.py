"""
Swiss small-cap "crash then rebound" scanner.

Finds SIX Swiss Exchange-listed, Switzerland-domiciled small-cap stocks
that had a day with a >=5% loss followed IMMEDIATELY (next trading day) by
a >=5% gain, within the last N months.

Ported from the standalone research/ project (D:\\projects\\research) into
this app so it can run from a real, always-on server (Render) instead of a
Colab/Kaggle notebook - see app/routers/research.py for why: the original
research/ version was designed to run in Colab/Kaggle; this repo's version
is the same logic wired into a FastAPI background job instead (see
app/services/research_job.py), because the small-cap research page in
financial-sentiment-web needs a persistent Python backend to call, and
Vercel's serverless functions (where that Next.js app is deployed) can't
run a script like this at all - no Python runtime, and execution time caps
far below the ~1-2 minutes this scan takes.

Free tools only: yfinance (wraps Yahoo Finance's public screener + price
history endpoints, no API key).

Universe discovery (market-cap band + domicile filter) lives in
swiss_universe.py, shared with swiss_small_cap_today_screener.py - see
that module's docstring for why both a market-cap filter AND a per-company
`country` check are needed to get "small caps in Switzerland, excluding
foreign companies." run_scan() below takes an already-discovered/filtered
`domestic` dict rather than doing its own discovery, so a caller running
BOTH this scan and swiss_small_cap_today_screener.py's (see
research_job.py) only pays the ~30-60s domicile-filtering cost once, not
twice.
"""

import pandas as pd
import yfinance as yf

# --- Config ---

LOOKBACK_MONTHS = 3
# Fetched history is deliberately longer than the lookback window so the
# FIRST day inside the window still has a valid previous-close to compute
# a % change against - without this buffer, a lookback boundary that lands
# mid-week would silently drop that day's move.
HISTORY_PERIOD = "4mo"

DROP_THRESHOLD_PCT = -5.0   # day N close-to-close change <= this
GAIN_THRESHOLD_PCT = 5.0    # day N+1 close-to-close change >= this

# Manually researched via web search (not auto-fetched) - yfinance's
# Ticker.news only returns whatever is CURRENT news today, not an archive
# of what was published on a specific past date, so it can't answer "what
# news landed on the gain_date of this specific historical event." Keyed
# by (ticker, gain_date as "YYYY-MM-DD") since that's the day the news
# actually needs to line up with for a "smart money re-entry" read - add
# more entries here as you research more events; anything not in this
# dict just gets blank news columns rather than a guess.
NEWS_RESEARCH = {
    ("INRN.SW", "2026-08-04"): {
        "headline": "UBS raises Interroll price target to CHF 2,400 (from 2,295), reiterates Buy, "
                    "after H1 2026 results (sales +14% local currency, EBIT -2.2%, net income missed "
                    "consensus ~7%) - management cites 'solid basis for the remainder of the year' "
                    "and share gains in China.",
        "source": "https://www.streetinsider.com/Analyst+PT+Change/Interroll+Holding+AG+(INRN:SW)+PT+Raised+to+CHF2,400+at+UBS/26860350.html",
    },
    ("CNTL.SW", "2026-07-13"): {
        "headline": "Centiel secures its first U.S. market order, worth USD 8.7 million, under the Neo "
                    "Critical Power framework agreement for a data-center UPS project.",
        "source": "https://www.tradingview.com/news/eqs:57fd04b09094b:0-centiel-secures-first-order-in-the-u-s-market-worth-usd-8-7-million-under-the-neo-critical-power-framework-agreement/",
    },
    ("SWTQ.SW", "2026-07-28"): {
        "headline": "No specific headline confirmed for this date - an earnings report fell close to "
                    "this window (~July 24) and gain-day volume was 5.4x the 3-month average, "
                    "consistent with a post-earnings repricing, but not verified against a dated source.",
        "source": None,
    },
}


def find_crash_then_rebound(symbols, domestic, lookback_months, history_period, drop_threshold, gain_threshold):
    """Batch-downloads daily OHLCV for all `symbols` at once (one bulk
    request rather than one per ticker - yfinance/Yahoo handles this far
    better than a per-symbol loop for price history specifically, unlike
    the .info endpoint used in swiss_universe.filter_domestic). Returns a
    DataFrame of every (drop_date, gain_date) pair found within the last
    `lookback_months`, one row per match - a single ticker can appear more
    than once if it had multiple such events. Each match row carries the
    loss day's and rebound day's own OHLCV - VOLUME in particular, since a
    move on thin volume vs. heavy volume tells very different stories
    about how real/tradeable it was - plus an approximate P/E.

    P/E is deliberately labeled "_approx": yfinance's quarterly financials
    are EMPTY for most of this small/illiquid universe (confirmed live for
    several matched tickers), so a true point-in-time historical P/E isn't
    obtainable from free data here. This uses TODAY's trailing EPS applied
    to the historical close instead - fine for a rough read, but can be
    wrong if the company reported an earnings surprise between the event
    date and today (which, for a >=5% single-day move, is a real
    possibility, not an edge case). None when trailing_eps is missing or
    non-positive (a loss-making company has no meaningful P/E), same
    "don't guess" convention as the rest of this module.
    """
    if not symbols:
        return pd.DataFrame()

    data = yf.download(
        symbols, period=history_period, interval="1d",
        group_by="ticker", auto_adjust=True, threads=True, progress=False,
    )
    # yf.download returns a flat (non-multi-indexed) frame when given a
    # single symbol - normalize to the same multi-indexed shape as the
    # multi-symbol case so the loop below doesn't need two code paths.
    if len(symbols) == 1:
        data = pd.concat({symbols[0]: data}, axis=1)

    cutoff = pd.Timestamp.today(tz=data.index.tz) - pd.DateOffset(months=lookback_months)

    def approx_pe(close_price, trailing_eps):
        if not trailing_eps or trailing_eps <= 0:
            return None
        return round(close_price / trailing_eps, 2)

    matches = []
    for symbol in symbols:
        try:
            ohlcv = data[symbol][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            continue
        if ohlcv.empty:
            continue

        close = ohlcv["Close"]
        volume = ohlcv["Volume"]
        # 3-month AVERAGE volume as the yardstick for "was this day's
        # volume unusual" - computed over the full downloaded window
        # (HISTORY_PERIOD), not just the lookback window, for a more
        # stable baseline.
        avg_volume = volume.mean()
        pct_change = close.pct_change() * 100
        trailing_eps = domestic[symbol]["trailing_eps"]
        in_window = pct_change.index >= cutoff

        for i in range(len(pct_change) - 1):
            if not in_window[i + 1]:
                continue
            drop_pct = pct_change.iloc[i]
            gain_pct = pct_change.iloc[i + 1]
            if drop_pct <= drop_threshold and gain_pct >= gain_threshold:
                loss_volume = volume.iloc[i]
                gain_volume = volume.iloc[i + 1]
                matches.append({
                    "ticker": symbol,
                    "loss_date": pct_change.index[i].date().isoformat(),
                    "loss_open": round(ohlcv["Open"].iloc[i], 2),
                    "loss_high": round(ohlcv["High"].iloc[i], 2),
                    "loss_low": round(ohlcv["Low"].iloc[i], 2),
                    "loss_close": round(close.iloc[i], 2),
                    "loss_volume": int(loss_volume) if pd.notna(loss_volume) else None,
                    "loss_volume_vs_3mo_avg": round(loss_volume / avg_volume, 2) if avg_volume else None,
                    "loss_pe_approx": approx_pe(close.iloc[i], trailing_eps),
                    "drop_pct": round(drop_pct, 2),
                    "gain_date": pct_change.index[i + 1].date().isoformat(),
                    "gain_open": round(ohlcv["Open"].iloc[i + 1], 2),
                    "gain_high": round(ohlcv["High"].iloc[i + 1], 2),
                    "gain_low": round(ohlcv["Low"].iloc[i + 1], 2),
                    "gain_close": round(close.iloc[i + 1], 2),
                    "gain_volume": int(gain_volume) if pd.notna(gain_volume) else None,
                    "gain_volume_vs_3mo_avg": round(gain_volume / avg_volume, 2) if avg_volume else None,
                    "gain_pe_approx": approx_pe(close.iloc[i + 1], trailing_eps),
                    "gain_pct": round(gain_pct, 2),
                })

    return pd.DataFrame(matches)


def attach_news(results):
    """Adds news_headline/news_source columns from NEWS_RESEARCH, matched
    on (ticker, gain_date) - blank for any row not manually researched
    (most of them, by design - see NEWS_RESEARCH's comment)."""
    def lookup(row, field):
        entry = NEWS_RESEARCH.get((row["ticker"], row["gain_date"]))
        return entry[field] if entry else None

    results = results.copy()
    results["news_headline"] = results.apply(lambda r: lookup(r, "headline"), axis=1)
    results["news_source"] = results.apply(lambda r: lookup(r, "source"), axis=1)
    return results


def run_scan(domestic: dict) -> pd.DataFrame:
    """Runs the full crash-then-rebound scan against an already-discovered/
    domicile-filtered `domestic` dict (see swiss_universe.filter_domestic) -
    discovery isn't repeated here so a caller running this alongside
    swiss_small_cap_today_screener.py's scan (see research_job.py) only
    pays that ~30-60s cost once. Returns a DataFrame, empty if no matches -
    never raises for "no results," only for a genuine fetch failure.
    """
    results = find_crash_then_rebound(
        list(domestic.keys()), domestic, LOOKBACK_MONTHS, HISTORY_PERIOD, DROP_THRESHOLD_PCT, GAIN_THRESHOLD_PCT
    )
    if results.empty:
        return results

    results["name"] = results["ticker"].map(lambda t: domestic[t]["name"])
    results["sector"] = results["ticker"].map(lambda t: domestic[t]["sector"])
    results["market_cap"] = results["ticker"].map(lambda t: domestic[t]["market_cap"])
    # Current-snapshot company fields (not day-specific like loss_pe_approx/
    # gain_pe_approx above, which use the historical close - these are
    # today's values, pulled from the SAME .info call swiss_universe.
    # filter_domestic already makes, no extra yfinance request) - see that
    # function's docstring for the full field list.
    results["trailing_pe"] = results["ticker"].map(lambda t: domestic[t]["trailing_pe"])
    results["forward_pe"] = results["ticker"].map(lambda t: domestic[t]["forward_pe"])
    results["dividend_yield"] = results["ticker"].map(lambda t: domestic[t]["dividend_yield"])
    results["ex_dividend_date"] = results["ticker"].map(lambda t: domestic[t]["ex_dividend_date"])
    results["beta"] = results["ticker"].map(lambda t: domestic[t]["beta"])
    results["fifty_two_week_high"] = results["ticker"].map(lambda t: domestic[t]["fifty_two_week_high"])
    results["fifty_two_week_low"] = results["ticker"].map(lambda t: domestic[t]["fifty_two_week_low"])
    results = results.sort_values("loss_date", ascending=False).reset_index(drop=True)
    results = results[[
        "ticker", "name", "sector", "market_cap",
        "trailing_pe", "forward_pe", "dividend_yield", "ex_dividend_date",
        "beta", "fifty_two_week_high", "fifty_two_week_low",
        "loss_date", "loss_open", "loss_high", "loss_low", "loss_close",
        "loss_volume", "loss_volume_vs_3mo_avg", "loss_pe_approx", "drop_pct",
        "gain_date", "gain_open", "gain_high", "gain_low", "gain_close",
        "gain_volume", "gain_volume_vs_3mo_avg", "gain_pe_approx", "gain_pct",
    ]]
    return attach_news(results)
