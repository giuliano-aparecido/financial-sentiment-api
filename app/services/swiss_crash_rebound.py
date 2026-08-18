"""
Swiss "crash then rebound" volatility scanner.

Finds SIX Swiss Exchange-listed, Switzerland-domiciled stocks that had a
day with a >=5% loss followed, within the next REBOUND_WINDOW_TRADING_DAYS
trading days, by a close that's >=5% ABOVE THE CRASH DAY'S OWN CLOSE (not
the previous day's close - see find_crash_then_rebound's own docstring for
why that distinction matters), within the last N months (12 as of
2026-08-18, widened from 3 at the user's request). Universe: SIX-listed,
Switzerland-domiciled companies with market cap > CHF 500M (SMI's 20
largest still excluded regardless - see swiss_universe.py's MIN/
MAX_MARKET_CAP_CHF and SMI_TICKERS). This is looking for VOLATILE movers,
not necessarily small companies specifically.

Ported from the standalone research/ project (D:\\projects\\research) into
this app so it can run from a real, always-on server (Render) instead of a
Colab/Kaggle notebook - see app/routers/research.py for why: the original
research/ version was designed to run in Colab/Kaggle; this repo's version
is the same logic wired into a FastAPI background job instead (see
app/services/research_job.py), because the volatility research page in
financial-sentiment-web needs a persistent Python backend to call, and
Vercel's serverless functions (where that Next.js app is deployed) can't
run a script like this at all - no Python runtime, and execution time caps
far below the ~1-2 minutes this scan takes.

Free tools only: yfinance (wraps Yahoo Finance's public screener + price
history endpoints, no API key).

Universe discovery (market-cap band + domicile filter) lives in
swiss_universe.py, shared with swiss_today_screener.py - see
that module's docstring for why both a market-cap filter AND a per-company
`country` check are needed to get "Swiss-domiciled stocks, excluding
foreign companies." run_scan() below takes an already-discovered/filtered
`domestic` dict rather than doing its own discovery, so a caller running
BOTH this scan and swiss_today_screener.py's (see
research_job.py) only pays the ~30-60s domicile-filtering cost once, not
twice.
"""

import time

import pandas as pd
import yfinance as yf

# --- Config ---

LOOKBACK_MONTHS = 12  # widened from 3 at the user's request, 2026-08-18
# Fetched history is deliberately longer than the lookback window so the
# FIRST day inside the window still has a valid previous-close to compute
# a % change against - without this buffer, a lookback boundary that lands
# mid-week would silently drop that day's move. Same 1-month buffer ratio
# as the original 3mo/"4mo" pair, scaled up with LOOKBACK_MONTHS.
HISTORY_PERIOD = "13mo"

DROP_THRESHOLD_PCT = -5.0   # day N close-to-close change <= this
GAIN_THRESHOLD_PCT = 5.0    # rebound-day close vs CRASH DAY's close >= this

# How many trading days after the crash day to look for a qualifying
# rebound - was hardcoded to "the immediate next day" before this;
# widened to a window since a real rebound often takes a couple of days
# to show up, not necessarily tomorrow. The FIRST day within this window
# whose close clears GAIN_THRESHOLD_PCT above the crash day's own close
# is the one recorded (see find_crash_then_rebound below) - later days
# within the window aren't separately reported once an earlier one
# already qualifies.
REBOUND_WINDOW_TRADING_DAYS = 3

# find_crash_then_rebound used to download ALL domestic tickers (100+) in a
# single yf.download(..., threads=True) call - one burst of fully-concurrent
# requests against Yahoo's unofficial history endpoint, no pacing at all
# (unlike swiss_universe.filter_domestic's own per-ticker .info loop, which
# already paces itself - see INFO_REQUEST_DELAY_SECONDS there). Confirmed
# live: this is what was actually tripping "Too Many Requests. Rate
# limited." on every scan attempt, not a longer-lived Yahoo IP block -
# retrying (even after waiting) reproduced the identical failure every
# time, because it's the SAME concurrent-burst request pattern being
# replayed, not an elapsed-time cooldown. Worse, a failure here happens
# BEFORE _crash_rebound_result caches anything (see research_job.py), so a
# failed attempt is never cached and gets retried from scratch, burst and
# all, on the very next scan. Chunking + no internal threading applies the
# same politeness-delay philosophy already used for the .info loop.
DOWNLOAD_CHUNK_SIZE = 25
DOWNLOAD_CHUNK_DELAY_SECONDS = 2.0


def download_ohlcv_chunked(symbols, period):
    """Batch-downloads daily OHLCV for all `symbols`, chunked + sequential
    (threads=False) + a delay between chunks - see DOWNLOAD_CHUNK_SIZE's
    own comment for why. Each chunk is normalized the same way a single
    yf.download call needs to be (yf.download returns a flat, non-multi-
    indexed frame for a single symbol, multi-indexed for more than one -
    a one-symbol final chunk needs the same treatment a one-symbol
    overall call used to), then concatenated column-wise since each chunk
    covers a disjoint set of symbols. Returns an empty DataFrame for an
    empty `symbols` list rather than erroring - callers can index into
    the result unconditionally.

    Factored out of find_crash_then_rebound below (which still uses it)
    so swiss_volatility_indicator.py's own 12-month scan can reuse the
    EXACT same chunking/pacing logic rather than a second, independently
    drifting copy of code that's already been the source of one real,
    confirmed-live rate-limit bug (see DOWNLOAD_CHUNK_SIZE's own comment)
    - a second copy would risk that fix (or a future one) landing in only
    one of the two places.

    Reads DOWNLOAD_CHUNK_SIZE/DOWNLOAD_CHUNK_DELAY_SECONDS as module
    globals inside the function body, not as default parameter values -
    deliberate: a default value is bound once, at function-DEFINITION
    time, so a test's monkeypatch.setattr(module, "DOWNLOAD_CHUNK_SIZE",
    ...) would silently have no effect on an already-bound default (a
    real, confirmed-live bug hit once already writing this function -
    see test_find_crash_then_rebound_chunks_download_calls, which
    monkeypatches this exact constant and would have failed against a
    default-arg version).
    """
    if not symbols:
        return pd.DataFrame()

    chunks = []
    for i in range(0, len(symbols), DOWNLOAD_CHUNK_SIZE):
        chunk_symbols = symbols[i:i + DOWNLOAD_CHUNK_SIZE]
        chunk_data = yf.download(
            chunk_symbols, period=period, interval="1d",
            group_by="ticker", auto_adjust=True, threads=False, progress=False,
        )
        if len(chunk_symbols) == 1:
            chunk_data = pd.concat({chunk_symbols[0]: chunk_data}, axis=1)
        chunks.append(chunk_data)
        if i + DOWNLOAD_CHUNK_SIZE < len(symbols):
            time.sleep(DOWNLOAD_CHUNK_DELAY_SECONDS)
    return pd.concat(chunks, axis=1)


def find_crash_then_rebound(
    symbols, domestic, lookback_months, history_period, drop_threshold, gain_threshold,
    rebound_window_days=REBOUND_WINDOW_TRADING_DAYS,
):
    """Batch-downloads daily OHLCV for all `symbols` at once (one bulk
    request rather than one per ticker - yfinance/Yahoo handles this far
    better than a per-symbol loop for price history specifically, unlike
    the .info endpoint used in swiss_universe.filter_domestic). Returns a
    DataFrame of every (drop_date, gain_date) pair found within the last
    `lookback_months`, one row per match - a single ticker can appear more
    than once if it had multiple such events. Each match row carries the
    loss day's and rebound day's own OHLC plus an approximate P/E - no
    per-event volume (removed 2026-08-18 at the user's request: the
    company-level 10-day average volume, added by run_scan below, is
    shown instead of a per-loss-day/per-gain-day volume figure).

    The rebound check looks up to `rebound_window_days` trading days ahead
    of the crash day for the FIRST day whose close is >= gain_threshold%
    above the CRASH DAY'S close - deliberately the crash day's close, not
    a rolling previous-day close, on every day checked within the window.
    This is what makes "still going down" not quietly count as progress
    toward a rebound: if day N+1 is DOWN another 3% from the crash close,
    day N+2 doesn't just need +5% over day N+1 (which would only be
    partial recovery) - it needs +5% over day N's original close, i.e. it
    has to make up its own drop AND day N+1's before it counts at all.
    `days_to_rebound` in each match row records how many trading days that
    actually took (1 = the old "immediate next day" behavior, still
    matched here as the day-1 case of the same window).

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

    data = download_ohlcv_chunked(symbols, history_period)

    cutoff = pd.Timestamp.today(tz=data.index.tz) - pd.DateOffset(months=lookback_months)

    def approx_pe(close_price, trailing_eps):
        if not trailing_eps or trailing_eps <= 0:
            return None
        return round(close_price / trailing_eps, 2)

    matches = []
    for symbol in symbols:
        try:
            ohlcv = data[symbol][["Open", "High", "Low", "Close"]].dropna(subset=["Close"])
        except KeyError:
            continue
        if ohlcv.empty:
            continue

        close = ohlcv["Close"]
        pct_change = close.pct_change() * 100
        trailing_eps = domestic[symbol]["trailing_eps"]
        in_window = pct_change.index >= cutoff

        n = len(close)
        for i in range(n):
            if not in_window[i]:
                continue
            drop_pct = pct_change.iloc[i]
            # NaN happens on a ticker's own FIRST available trading day
            # (pct_change has no prior close to compare against) - real,
            # confirmed-live bug: `drop_pct > drop_threshold` is silently
            # False for NaN (neither > nor <= any threshold evaluates
            # True), so this check alone let a NaN drop_pct fall through
            # as if it were a real, non-qualifying day instead of being
            # excluded outright - producing a match row with drop_pct
            # (and everything computed from it) as null, which crashed
            # the frontend's row.drop_pct.toFixed(2) on receipt. Confirmed
            # live for a recently-listed company whose own trading
            # history starts inside the lookback window - HISTORY_PERIOD's
            # buffer only protects against the WINDOW boundary landing
            # mid-week, not against an individual ticker's history being
            # shorter than the buffer itself. Explicit pd.isna check
            # covers both cases the bare comparison silently missed.
            if pd.isna(drop_pct) or drop_pct > drop_threshold:
                continue
            crash_close = close.iloc[i]

            # First trading day within the window whose close clears
            # gain_threshold% above crash_close - see this function's own
            # docstring for why crash_close (not a rolling previous-day
            # close) is the baseline on every day checked, not just day 1.
            rebound_j = None
            rebound_pct = None
            for offset in range(1, rebound_window_days + 1):
                j = i + offset
                if j >= n:
                    break
                pct_vs_crash = (close.iloc[j] - crash_close) / crash_close * 100
                if pct_vs_crash >= gain_threshold:
                    rebound_j = j
                    rebound_pct = pct_vs_crash
                    break
            if rebound_j is None:
                continue

            matches.append({
                "ticker": symbol,
                "loss_date": pct_change.index[i].date().isoformat(),
                "loss_open": round(ohlcv["Open"].iloc[i], 2),
                "loss_high": round(ohlcv["High"].iloc[i], 2),
                "loss_low": round(ohlcv["Low"].iloc[i], 2),
                "loss_close": round(close.iloc[i], 2),
                "loss_pe_approx": approx_pe(close.iloc[i], trailing_eps),
                "drop_pct": round(drop_pct, 2),
                "days_to_rebound": rebound_j - i,
                "gain_date": pct_change.index[rebound_j].date().isoformat(),
                "gain_open": round(ohlcv["Open"].iloc[rebound_j], 2),
                "gain_high": round(ohlcv["High"].iloc[rebound_j], 2),
                "gain_low": round(ohlcv["Low"].iloc[rebound_j], 2),
                "gain_close": round(close.iloc[rebound_j], 2),
                "gain_pe_approx": approx_pe(close.iloc[rebound_j], trailing_eps),
                "gain_pct": round(rebound_pct, 2),
            })

    return pd.DataFrame(matches)


def run_scan(domestic: dict) -> pd.DataFrame:
    """Runs the full crash-then-rebound scan against an already-discovered/
    domicile-filtered `domestic` dict (see swiss_universe.filter_domestic) -
    discovery isn't repeated here so a caller running this alongside
    swiss_today_screener.py's scan (see research_job.py) only
    pays that ~30-60s cost once. Returns a DataFrame, empty if no matches -
    never raises for "no results," only for a genuine fetch failure.
    """
    results = find_crash_then_rebound(
        list(domestic.keys()), domestic, LOOKBACK_MONTHS, HISTORY_PERIOD, DROP_THRESHOLD_PCT, GAIN_THRESHOLD_PCT,
        REBOUND_WINDOW_TRADING_DAYS,
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
    # Company-level (not event-level) 10-day average trading volume - see
    # swiss_universe.filter_domestic's avg_volume_10d, same already-
    # fetched figure reused here, no extra request. Replaces the former
    # per-event loss_volume/gain_volume/loss_volume_vs_10d_avg/
    # gain_volume_vs_10d_avg fields entirely (removed 2026-08-18 at the
    # user's request: "Remove Loss volume and Gain volume. Show only
    # average daily trading volume (ADTV)").
    results["avg_volume_10d"] = results["ticker"].map(lambda t: domestic[t].get("avg_volume_10d"))
    results = results.sort_values("loss_date", ascending=False).reset_index(drop=True)
    results = results[[
        "ticker", "name", "sector", "market_cap",
        "trailing_pe", "forward_pe", "dividend_yield", "ex_dividend_date",
        "beta", "fifty_two_week_high", "fifty_two_week_low", "avg_volume_10d",
        "loss_date", "loss_open", "loss_high", "loss_low", "loss_close",
        "loss_pe_approx", "drop_pct",
        "days_to_rebound",
        "gain_date", "gain_open", "gain_high", "gain_low", "gain_close",
        "gain_pe_approx", "gain_pct",
    ]]
    return results
