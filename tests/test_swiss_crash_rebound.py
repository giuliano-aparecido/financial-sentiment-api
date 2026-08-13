import pandas as pd
import pytest

import app.services.swiss_crash_rebound as crash_rebound_module
from app.services.swiss_crash_rebound import attach_news, find_crash_then_rebound, run_scan


def _ohlcv_frame(dates, closes, volumes):
    # Open/High/Low set equal to Close - none of this module's logic reads
    # them for anything except passthrough display, so exact values don't
    # matter for these tests, only that the columns exist.
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _fake_download(symbols, **kwargs):
    # Matches real yf.download's actual shape: FLAT (no ticker level) for
    # a single symbol, multi-indexed ({ticker: {OHLCV}}) for more than
    # one - find_crash_then_rebound's own single-symbol normalization
    # step (pd.concat) exists specifically to paper over this asymmetry,
    # so a fake that returns the multi-indexed shape unconditionally would
    # get double-wrapped for the single-symbol case and break indexing.
    if len(symbols) == 1:
        return crash_rebound_module._TEST_FRAMES[symbols[0]]
    frames = {symbol: crash_rebound_module._TEST_FRAMES[symbol] for symbol in symbols}
    return pd.concat(frames, axis=1)


@pytest.fixture
def today():
    return pd.Timestamp.today().normalize()


def test_find_crash_then_rebound_detects_a_match(monkeypatch, today):
    dates = pd.date_range(end=today, periods=10, freq="B")
    # Day -2: drop 6% (100 -> 94), Day -1: gain 6% (94 -> 99.64)
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 94, 99.64]
    volumes = [1000] * 8 + [5000, 8000]
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _ohlcv_frame(dates, closes, volumes)}
    monkeypatch.setattr(crash_rebound_module.yf, "download", _fake_download)

    domestic = {"TEST.SW": {"trailing_eps": 5.0}}
    results = find_crash_then_rebound(["TEST.SW"], domestic, lookback_months=3,
                                       history_period="4mo", drop_threshold=-5.0, gain_threshold=5.0)

    assert len(results) == 1
    row = results.iloc[0]
    assert row["ticker"] == "TEST.SW"
    assert row["drop_pct"] == -6.0
    assert row["gain_pct"] == 6.0
    assert row["loss_close"] == 94.0
    assert row["gain_close"] == 99.64
    assert row["loss_volume"] == 5000
    assert row["gain_volume"] == 8000


def test_find_crash_then_rebound_no_match_when_gain_too_small(monkeypatch, today):
    dates = pd.date_range(end=today, periods=10, freq="B")
    # Drop 6%, but only a 2% rebound - shouldn't qualify.
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 94, 95.88]
    volumes = [1000] * 10
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _ohlcv_frame(dates, closes, volumes)}
    monkeypatch.setattr(crash_rebound_module.yf, "download", _fake_download)

    domestic = {"TEST.SW": {"trailing_eps": 5.0}}
    results = find_crash_then_rebound(["TEST.SW"], domestic, lookback_months=3,
                                       history_period="4mo", drop_threshold=-5.0, gain_threshold=5.0)
    assert results.empty


def test_find_crash_then_rebound_excludes_matches_outside_lookback_window(monkeypatch, today):
    # 4 months of history (buffer for the lookback window - see module
    # docstring), but the crash+rebound pair happened right at the START,
    # well outside the 1-month lookback used in this test.
    dates = pd.date_range(end=today, periods=80, freq="B")
    closes = [100.0] * 80
    closes[1] = 94.0    # -6% - old, outside a 1-month lookback
    closes[2] = 99.64   # +6% - old, outside a 1-month lookback
    volumes = [1000] * 80
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _ohlcv_frame(dates, closes, volumes)}
    monkeypatch.setattr(crash_rebound_module.yf, "download", _fake_download)

    domestic = {"TEST.SW": {"trailing_eps": 5.0}}
    results = find_crash_then_rebound(["TEST.SW"], domestic, lookback_months=1,
                                       history_period="4mo", drop_threshold=-5.0, gain_threshold=5.0)
    assert results.empty


def test_find_crash_then_rebound_pe_approx_none_for_lossmaking_company(monkeypatch, today):
    dates = pd.date_range(end=today, periods=10, freq="B")
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 94, 99.64]
    volumes = [1000] * 10
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _ohlcv_frame(dates, closes, volumes)}
    monkeypatch.setattr(crash_rebound_module.yf, "download", _fake_download)

    domestic = {"TEST.SW": {"trailing_eps": -1.5}}  # loss-making - no meaningful P/E
    results = find_crash_then_rebound(["TEST.SW"], domestic, lookback_months=3,
                                       history_period="4mo", drop_threshold=-5.0, gain_threshold=5.0)
    assert results.iloc[0]["loss_pe_approx"] is None
    assert results.iloc[0]["gain_pe_approx"] is None


def test_find_crash_then_rebound_chunks_download_calls(monkeypatch, today):
    # Regression test: find_crash_then_rebound used to download every
    # symbol in one single yf.download(..., threads=True) call - a fully
    # concurrent burst against Yahoo with no pacing, confirmed live as what
    # was actually tripping "Too Many Requests" on every scan attempt (see
    # DOWNLOAD_CHUNK_SIZE's own comment). Asserts BOTH that downloading
    # happens in separate, smaller, sequential (threads=False) calls, AND
    # that chunking doesn't lose or misattribute any results once combined
    # back together.
    dates = pd.date_range(end=today, periods=10, freq="B")
    closes_match = [100, 100, 100, 100, 100, 100, 100, 100, 94, 99.64]
    closes_flat = [100] * 10
    volumes_match = [1000] * 8 + [5000, 8000]
    volumes_flat = [1000] * 10

    symbols = [f"T{i}.SW" for i in range(5)]
    crash_rebound_module._TEST_FRAMES = {
        symbols[0]: _ohlcv_frame(dates, closes_match, volumes_match),  # match, 1st chunk
        symbols[1]: _ohlcv_frame(dates, closes_flat, volumes_flat),
        symbols[2]: _ohlcv_frame(dates, closes_flat, volumes_flat),
        symbols[3]: _ohlcv_frame(dates, closes_match, volumes_match),  # match, 2nd chunk
        symbols[4]: _ohlcv_frame(dates, closes_flat, volumes_flat),
    }
    monkeypatch.setattr(crash_rebound_module, "DOWNLOAD_CHUNK_SIZE", 2)
    monkeypatch.setattr(crash_rebound_module, "DOWNLOAD_CHUNK_DELAY_SECONDS", 0)

    call_symbol_lists = []

    def _tracking_fake_download(syms, **kwargs):
        call_symbol_lists.append(list(syms))
        assert kwargs.get("threads") is False
        return _fake_download(syms, **kwargs)

    monkeypatch.setattr(crash_rebound_module.yf, "download", _tracking_fake_download)

    domestic = {s: {"trailing_eps": 5.0} for s in symbols}
    results = find_crash_then_rebound(symbols, domestic, lookback_months=3,
                                       history_period="4mo", drop_threshold=-5.0, gain_threshold=5.0)

    # 5 symbols chunked at size 2 -> 3 separate, smaller calls, not one big
    # burst.
    assert call_symbol_lists == [symbols[0:2], symbols[2:4], symbols[4:5]]
    # Matches from both the first AND second chunk survive being combined
    # back into one result set.
    assert set(results["ticker"]) == {symbols[0], symbols[3]}


def test_attach_news_matches_on_ticker_and_gain_date():
    results = pd.DataFrame([
        {"ticker": "INRN.SW", "gain_date": "2026-08-04"},
        {"ticker": "UNKNOWN.SW", "gain_date": "2026-01-01"},
    ])
    with_news = attach_news(results)
    assert with_news.iloc[0]["news_headline"] is not None
    assert with_news.iloc[0]["news_source"] is not None
    # pandas represents "no match" as NaN here (not None) internally - the
    # JSON-safety conversion to a real None happens one layer up, at
    # research_job._json_safe_records, not inside attach_news itself.
    assert pd.isna(with_news.iloc[1]["news_headline"])
    assert pd.isna(with_news.iloc[1]["news_source"])


def test_run_scan_returns_empty_dataframe_when_no_symbols():
    assert run_scan({}).empty


def test_run_scan_attaches_current_snapshot_company_fields(monkeypatch, today):
    dates = pd.date_range(end=today, periods=10, freq="B")
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 94, 99.64]
    volumes = [1000] * 8 + [5000, 8000]
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _ohlcv_frame(dates, closes, volumes)}
    monkeypatch.setattr(crash_rebound_module.yf, "download", _fake_download)

    domestic = {
        "TEST.SW": {
            "name": "Test AG", "sector": "Industrials", "market_cap": 1e9, "trailing_eps": 5.0,
            "trailing_pe": 20.0, "forward_pe": 17.5, "dividend_yield": 2.24,
            "ex_dividend_date": "2026-06-13", "beta": 1.27,
            "fifty_two_week_high": 2590.0, "fifty_two_week_low": 1258.0,
        },
    }
    results = run_scan(domestic)

    row = results.iloc[0]
    assert row["trailing_pe"] == 20.0
    assert row["forward_pe"] == 17.5
    assert row["dividend_yield"] == 2.24
    assert row["ex_dividend_date"] == "2026-06-13"
    assert row["beta"] == 1.27
    assert row["fifty_two_week_high"] == 2590.0
    assert row["fifty_two_week_low"] == 1258.0
    # Raw volume (not just the vs-3mo-average ratio) is present for both
    # the loss day and the gain day.
    assert row["loss_volume"] == 5000
    assert row["gain_volume"] == 8000
