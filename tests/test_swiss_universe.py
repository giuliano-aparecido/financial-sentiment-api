import json
import threading
import time

from yfinance.exceptions import YFRateLimitError

import app.services.swiss_universe as swiss_universe_module
from app.services.swiss_universe import _ex_dividend_date, discover_candidates, filter_domestic, filter_domestic_batched


class _FakeTicker:
    def __init__(self, info):
        self.info = info


def _base_info(**overrides):
    info = {
        "country": "Switzerland",
        "sector": "Industrials",
        "trailingEps": 5.0,
        "trailingPE": 20.0,
        "forwardPE": 17.5,
        "dividendYield": 2.24,
        "exDividendDate": 1781568000,  # 2026-06-16T00:00:00Z
        "beta": 1.27,
        "fiftyTwoWeekHigh": 2590.0,
        "fiftyTwoWeekLow": 1258.0,
        "averageDailyVolume10Day": 100_000,  # comfortably above MIN_AVG_DAILY_VOLUME_10D (50k)
    }
    info.update(overrides)
    return info


def test_ex_dividend_date_converts_unix_timestamp_to_iso_date():
    assert _ex_dividend_date({"exDividendDate": 1781568000}) == "2026-06-16"


def test_ex_dividend_date_none_when_absent():
    assert _ex_dividend_date({}) is None


def test_ex_dividend_date_none_when_falsy_zero():
    # Confirmed live: some tickers have exDividendDate=0 rather than the
    # key being absent - fromtimestamp(0) would produce a bogus 1970 date
    # if not guarded.
    assert _ex_dividend_date({"exDividendDate": 0}) is None


def test_filter_domestic_captures_current_snapshot_fields(monkeypatch):
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(_base_info()))
    candidates = {"TEST.SW": {"longName": "Test AG", "marketCap": 1e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)

    assert failed == []
    entry = domestic["TEST.SW"]
    assert entry["trailing_pe"] == 20.0
    assert entry["forward_pe"] == 17.5
    assert entry["dividend_yield"] == 2.24
    assert entry["ex_dividend_date"] == "2026-06-16"
    assert entry["beta"] == 1.27
    assert entry["fifty_two_week_high"] == 2590.0
    assert entry["fifty_two_week_low"] == 1258.0


def test_filter_domestic_new_fields_are_none_when_missing(monkeypatch):
    info = _base_info()
    for key in ["trailingPE", "forwardPE", "dividendYield", "exDividendDate", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]:
        info.pop(key, None)
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    candidates = {"TEST.SW": {"longName": "Test AG", "marketCap": 1e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)

    assert failed == []
    entry = domestic["TEST.SW"]
    assert entry["trailing_pe"] is None
    assert entry["forward_pe"] is None
    assert entry["dividend_yield"] is None
    assert entry["ex_dividend_date"] is None
    assert entry["beta"] is None
    assert entry["fifty_two_week_high"] is None
    assert entry["fifty_two_week_low"] is None


def test_filter_domestic_excludes_foreign_domiciled(monkeypatch):
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(_base_info(country="Germany")))
    candidates = {"FOREIGN.SW": {"longName": "Foreign SE", "marketCap": 1e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == []  # a legitimate exclusion (wrong domicile), not a fetch failure - must not be retried


def test_filter_domestic_excludes_below_min_avg_daily_volume_10d(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(averageDailyVolume10Day=swiss_universe_module.MIN_AVG_DAILY_VOLUME_10D - 1)),
    )
    candidates = {"THIN.SW": {"longName": "Thin AG", "marketCap": 1e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == []  # legitimate exclusion (thin volume), not a fetch failure


def test_filter_domestic_includes_at_exactly_min_avg_daily_volume_10d(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(averageDailyVolume10Day=swiss_universe_module.MIN_AVG_DAILY_VOLUME_10D)),
    )
    candidates = {"EDGE.SW": {"longName": "Edge AG", "marketCap": 1e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert "EDGE.SW" in domestic
    assert failed == []


def test_filter_domestic_excludes_missing_avg_daily_volume_10d(monkeypatch):
    info = _base_info()
    del info["averageDailyVolume10Day"]
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    candidates = {"NOVOL.SW": {"longName": "No Volume AG", "marketCap": 1e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == []  # legitimate exclusion (no volume data), not a fetch failure


def test_filter_domestic_captures_avg_volume_10d(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(averageDailyVolume10Day=250_000)),
    )
    candidates = {"TEST.SW": {"longName": "Test AG", "marketCap": 1e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic["TEST.SW"]["avg_volume_10d"] == 250_000


def test_filter_domestic_market_cap_falls_back_to_info_when_quote_lacks_it(monkeypatch):
    # Regression test: STATIC_DOMESTIC_TICKER_SNAPSHOT's synthesized quote
    # (used when the live screener fails - see discover_candidates' own
    # docstring) never carries marketCap at all, which was silently
    # showing "N/A" market cap for every ticker discovered that way in
    # every research table - not a display bug, a missing fallback here.
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(marketCap=4.2e9)),
    )
    # No "marketCap" key in the candidate quote at all - matches the
    # static-fallback discovery path's synthesized quote shape exactly.
    candidates = {"TEST.SW": {"longName": "Test AG"}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic["TEST.SW"]["market_cap"] == 4.2e9


def test_filter_domestic_market_cap_prefers_quote_over_info_when_both_present(monkeypatch):
    # The live screener path's quote already carries marketCap at no
    # extra request cost - it should win over info's own value rather
    # than the two being merged unpredictably.
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(marketCap=999e9)),
    )
    candidates = {"TEST.SW": {"longName": "Test AG", "marketCap": 4.2e9}}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic["TEST.SW"]["market_cap"] == 4.2e9


def test_filter_domestic_handles_many_tickers_correctly_under_concurrency(monkeypatch):
    # Not a timing/speed assertion - just confirms results are complete
    # and correct (nothing dropped/duplicated/mixed up between symbols)
    # when run through the thread pool instead of a serial loop.
    def fake_ticker(symbol):
        return _FakeTicker(_base_info(sector=symbol))  # sector encodes which symbol answered

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {f"T{i}.SW": {"longName": f"Company {i}", "marketCap": 1e9} for i in range(12)}

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0, max_workers=5)

    assert failed == []
    assert set(domestic.keys()) == set(candidates.keys())
    for symbol, entry in domestic.items():
        assert entry["sector"] == symbol  # each entry answered for itself, not a neighbor's data


def test_filter_domestic_runs_fetches_concurrently(monkeypatch):
    # Proves the thread pool is actually overlapping requests, not just
    # preserving correct results while secretly still serialized - each
    # fake fetch blocks briefly and records how many were in-flight at
    # once; with max_workers=3 over 3 tickers that each take longer than
    # the gaps between submissions, at least 2 should overlap.
    lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0}

    def fake_ticker(symbol):
        with lock:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        time.sleep(0.05)
        with lock:
            state["in_flight"] -= 1
        return _FakeTicker(_base_info())

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {f"T{i}.SW": {"longName": f"Company {i}", "marketCap": 1e9} for i in range(3)}

    filter_domestic(candidates, delay_seconds=0, max_workers=3)

    assert state["max_in_flight"] >= 2


def test_filter_domestic_marks_genuine_fetch_exception_as_failed(monkeypatch):
    def fake_ticker(symbol):
        raise Exception("simulated transient fetch failure, NOT a rate limit")

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {"BROKEN.SW": {"longName": "Broken AG", "marketCap": 1e9}}

    domestic, failed, hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == ["BROKEN.SW"]
    assert hit_rate_limit is False  # an ordinary exception, not YFRateLimitError - no cooldown signal


def test_filter_domestic_only_marks_the_symbol_that_actually_raised(monkeypatch):
    def fake_ticker(symbol):
        if symbol == "BROKEN.SW":
            raise Exception("simulated transient fetch failure")
        return _FakeTicker(_base_info())

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {
        "OK.SW": {"longName": "Fine AG", "marketCap": 1e9},
        "BROKEN.SW": {"longName": "Broken AG", "marketCap": 1e9},
    }

    domestic, failed, _hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert "OK.SW" in domestic
    assert "BROKEN.SW" not in domestic
    assert failed == ["BROKEN.SW"]


def test_filter_domestic_flags_hit_rate_limit_on_real_yf_rate_limit_error(monkeypatch):
    def fake_ticker(symbol):
        raise YFRateLimitError()

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {"RATELIMITED.SW": {"longName": "Rate Limited AG", "marketCap": 1e9}}

    domestic, failed, hit_rate_limit = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == ["RATELIMITED.SW"]
    assert hit_rate_limit is True


def test_filter_domestic_rate_limit_on_one_ticker_does_not_hide_others_already_in_flight(monkeypatch):
    # max_workers=1 (serial) so results are deterministic: OK.SW completes
    # first and lands in `domestic` before RATELIMITED.SW's failure is
    # even seen - a rate limit detected partway through must not discard
    # real successes already collected.
    def fake_ticker(symbol):
        if symbol == "RATELIMITED.SW":
            raise YFRateLimitError()
        return _FakeTicker(_base_info())

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {
        "OK.SW": {"longName": "Fine AG", "marketCap": 1e9},
        "RATELIMITED.SW": {"longName": "Rate Limited AG", "marketCap": 1e9},
    }

    domestic, failed, hit_rate_limit = filter_domestic(candidates, delay_seconds=0, max_workers=1)
    assert "OK.SW" in domestic
    assert failed == ["RATELIMITED.SW"]
    assert hit_rate_limit is True


def test_filter_domestic_cancels_queued_work_after_rate_limit(monkeypatch):
    # With max_workers=1 (serial), only the first ticker ever actually
    # runs - everything queued behind it should be cancelled once that
    # first call raises YFRateLimitError, not attempted one-by-one.
    call_count = {"n": 0}

    def fake_ticker(symbol):
        call_count["n"] += 1
        raise YFRateLimitError()

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {f"T{i}.SW": {"longName": f"Company {i}", "marketCap": 1e9} for i in range(20)}

    domestic, failed, hit_rate_limit = filter_domestic(candidates, delay_seconds=0, max_workers=1)
    assert hit_rate_limit is True
    # Not a strict ==1 (cancellation is best-effort against a queue that
    # could theoretically hand off one more before the cancel lands), but
    # this must be far short of all 20 - proves the short-circuit worked.
    assert call_count["n"] < 5


def test_discover_candidates_excludes_configured_tickers(monkeypatch):
    def fake_screen(query, offset, size, sortField, sortAsc):
        if offset > 0:
            return {"quotes": [], "total": 1}
        return {
            "quotes": [
                {"symbol": "SNBN.SW", "quoteType": "EQUITY"},
                {"symbol": "REAL.SW", "quoteType": "EQUITY"},
            ],
            "total": 2,
        }

    monkeypatch.setattr(swiss_universe_module.yf, "screen", fake_screen)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates()
    assert "SNBN.SW" not in candidates
    assert "REAL.SW" in candidates


def test_discover_candidates_falls_back_to_static_snapshot_on_screener_failure(monkeypatch):
    # Regression test: confirmed live, yf.screen (the live screener) is the
    # ONLY endpoint that has ever failed across every "Too Many Requests"
    # incident this scan has hit - this fallback exists specifically so a
    # screener-only outage degrades the scan instead of failing it
    # outright.
    def _boom(*a, **kw):
        raise Exception("simulated screener outage")

    monkeypatch.setattr(swiss_universe_module.yf, "screen", _boom)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates()
    assert candidates == {
        symbol: {"symbol": symbol, "quoteType": "EQUITY", "longName": name, "sector": sector}
        for symbol, (name, sector) in swiss_universe_module.STATIC_DOMESTIC_TICKER_SNAPSHOT.items()
        if symbol not in swiss_universe_module.EXCLUDED_TICKERS
    }


def test_discover_candidates_static_fallback_excludes_configured_tickers(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module, "STATIC_DOMESTIC_TICKER_SNAPSHOT",
        {"SNBN.SW": ("Swiss National Bank", "Financial Services"), "REAL.SW": ("Real Co", "Industrials")},
    )

    def _boom(*a, **kw):
        raise Exception("simulated screener outage")

    monkeypatch.setattr(swiss_universe_module.yf, "screen", _boom)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates()
    assert "SNBN.SW" not in candidates
    assert "REAL.SW" in candidates


def test_discover_candidates_prefers_live_screener_when_it_works(monkeypatch):
    # The fallback must not be used when the live screener succeeds -
    # self-healing depends on always trying live first.
    def fake_screen(query, offset, size, sortField, sortAsc):
        if offset > 0:
            return {"quotes": [], "total": 1}
        return {"quotes": [{"symbol": "REAL.SW", "quoteType": "EQUITY"}], "total": 1}

    monkeypatch.setattr(swiss_universe_module.yf, "screen", fake_screen)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates()
    assert set(candidates) == {"REAL.SW"}
    # Live quote shape (no "longName"/"sector" keys forced in) confirms this
    # came from the live path, not the fallback's synthesized quote dict.
    assert candidates["REAL.SW"] == {"symbol": "REAL.SW", "quoteType": "EQUITY"}


def test_discover_candidates_fallback_quotes_are_usable_by_filter_domestic(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module, "STATIC_DOMESTIC_TICKER_SNAPSHOT",
        {"REAL.SW": ("Real Co", "Industrials")},
    )

    def _boom(*a, **kw):
        raise Exception("simulated screener outage")

    monkeypatch.setattr(swiss_universe_module.yf, "screen", _boom)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(_base_info()))
    monkeypatch.setattr(swiss_universe_module.time, "sleep", lambda *a: None)

    candidates = discover_candidates()
    domestic, failed, _hit_rate_limit = filter_domestic(candidates)
    assert domestic["REAL.SW"]["name"] == "Real Co"
    assert failed == []


class _FakeCookies(dict):
    def set(self, name, value):
        self[name] = value


class _FakeSession:
    def __init__(self):
        self.cookies = _FakeCookies()


class _FakeYfData:
    def __init__(self, crumb=None):
        self._crumb = crumb
        self._session = _FakeSession()


def test_seed_yf_session_from_env_noop_when_unset(monkeypatch):
    monkeypatch.delenv("YF_SEED_CRUMB", raising=False)
    monkeypatch.delenv("YF_SEED_COOKIES", raising=False)
    # Asserts YfData() is never even constructed - not just "did nothing to
    # it" - so the normal (no env vars set) path has zero overhead/side
    # effects on the real yfinance singleton.
    monkeypatch.setattr(swiss_universe_module, "YfData", lambda: (_ for _ in ()).throw(
        AssertionError("YfData() should not be called when env vars are unset")))
    swiss_universe_module._seed_yf_session_from_env()


def test_seed_yf_session_from_env_seeds_crumb_and_cookies(monkeypatch):
    monkeypatch.setenv("YF_SEED_CRUMB", "test-crumb-123")
    monkeypatch.setenv("YF_SEED_COOKIES", json.dumps({"A1": "abc", "A3": "def"}))
    fake = _FakeYfData()
    monkeypatch.setattr(swiss_universe_module, "YfData", lambda: fake)

    swiss_universe_module._seed_yf_session_from_env()

    assert fake._crumb == "test-crumb-123"
    assert dict(fake._session.cookies) == {"A1": "abc", "A3": "def"}


def test_discover_candidates_static_fallback_includes_smi_names(monkeypatch):
    # Regression guard: SMI names (Nestle/Novartis/UBS etc.) used to be
    # excluded unconditionally - removed 2026-08-19 at the user's explicit
    # direction (confirmed live it was hiding real matches, e.g. Nestle
    # showing genuine 2%+ single-day losses and gains). The only
    # requirement now is the market-cap band + Switzerland domicile, so
    # these names must come through like any other qualifying company.
    def _boom(*a, **kw):
        raise Exception("simulated screener outage")

    monkeypatch.setattr(swiss_universe_module.yf, "screen", _boom)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates()
    assert "NESN.SW" in candidates  # Nestle
    assert "NOVN.SW" in candidates  # Novartis
    assert "UBSG.SW" in candidates  # UBS
    assert "INRN.SW" in candidates


# --- filter_domestic_batched (scheduled-scan batching, see scheduler.py) ---


def _four_candidates():
    return {f"T{i}.SW": {"longName": f"Ticker {i}"} for i in range(4)}


def test_filter_domestic_batched_single_batch_matches_filter_domestic(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(swiss_universe_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(_base_info()))

    candidates = _four_candidates()
    domestic, failed, _hit_rate_limit = filter_domestic_batched(candidates, num_batches=1, batch_delay_seconds=999, delay_seconds=0)

    assert set(domestic) == set(candidates)
    assert failed == []
    # num_batches=1 must not sleep at all (no "between batches" gap when
    # there's only one) - delay_seconds=0 above also rules out the
    # per-ticker politeness sleep as a false positive here.
    assert 999 not in sleep_calls


def test_filter_domestic_batched_sleeps_between_but_not_after_last_batch(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(swiss_universe_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(_base_info()))

    candidates = _four_candidates()
    domestic, failed, _hit_rate_limit = filter_domestic_batched(candidates, num_batches=4, batch_delay_seconds=42, delay_seconds=0)

    assert set(domestic) == set(candidates)
    assert failed == []
    # 4 candidates / 4 batches = 1 ticker each -> 3 gaps between 4 batches,
    # never a 4th trailing sleep after the last one.
    assert sleep_calls.count(42) == 3


def test_filter_domestic_batched_merges_results_and_failures_across_batches(monkeypatch):
    def fake_ticker(symbol):
        if symbol == "T2.SW":
            raise Exception("simulated transient fetch failure")
        return _FakeTicker(_base_info())

    monkeypatch.setattr(swiss_universe_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)

    candidates = _four_candidates()
    domestic, failed, _hit_rate_limit = filter_domestic_batched(candidates, num_batches=2, batch_delay_seconds=0, delay_seconds=0)

    assert set(domestic) == {"T0.SW", "T1.SW", "T3.SW"}
    assert failed == ["T2.SW"]


def test_filter_domestic_batched_stops_issuing_further_batches_after_rate_limit(monkeypatch):
    # T0/T1 are batch 1, T2/T3 are batch 2 - batch 1's rate limit must
    # stop batch 2 from ever being attempted (no sleep, no live calls),
    # with T2/T3 still landing in failed_symbols for the next scheduled
    # run's retry rather than silently vanishing.
    sleep_calls = []
    fetched = []

    def fake_ticker(symbol):
        fetched.append(symbol)
        if symbol in ("T0.SW", "T1.SW"):
            raise YFRateLimitError()
        return _FakeTicker(_base_info())

    monkeypatch.setattr(swiss_universe_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)

    candidates = _four_candidates()
    domestic, failed, hit_rate_limit = filter_domestic_batched(candidates, num_batches=2, batch_delay_seconds=42, delay_seconds=0)

    assert hit_rate_limit is True
    assert domestic == {}
    assert set(failed) == {"T0.SW", "T1.SW", "T2.SW", "T3.SW"}  # batch 2's tickers included even though never fetched
    assert not any(s.startswith("T2") or s.startswith("T3") for s in fetched)  # batch 2 never actually attempted
    assert 42 not in sleep_calls  # no inter-batch delay paid for a batch that never ran


def test_discover_candidates_live_includes_smi_names(monkeypatch):
    def fake_screen(query, offset, size, sortField, sortAsc):
        if offset > 0:
            return {"quotes": [], "total": 1}
        return {
            "quotes": [
                {"symbol": "NESN.SW", "quoteType": "EQUITY"},
                {"symbol": "REAL.SW", "quoteType": "EQUITY"},
            ],
            "total": 2,
        }

    monkeypatch.setattr(swiss_universe_module.yf, "screen", fake_screen)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates()
    assert "NESN.SW" in candidates
    assert "REAL.SW" in candidates


def test_market_cap_band_has_no_meaningful_upper_bound():
    # MAX_MARKET_CAP_CHF (1 trillion) should comfortably clear even the
    # largest SIX-listed company, so the band is effectively "500M+, no
    # real ceiling" - not a second small-vs-large distinction to maintain.
    assert swiss_universe_module.MAX_MARKET_CAP_CHF > 500_000_000_000


def test_static_snapshot_includes_smi_large_caps_not_just_small_caps():
    # Regression guard: the static fallback used to be captured at a
    # small-cap band only, so it was missing SMI giants entirely. It's
    # captured at the full 500M+ band now, so this checks a couple of
    # well-known large caps are actually present, not just that the dict
    # got bigger.
    snapshot = swiss_universe_module.STATIC_DOMESTIC_TICKER_SNAPSHOT
    assert "NESN.SW" in snapshot  # Nestle
    assert "NOVN.SW" in snapshot  # Novartis
    assert "UBSG.SW" in snapshot  # UBS


def test_seed_yf_session_from_env_does_not_clobber_existing_crumb(monkeypatch):
    # A process that already fetched (or already seeded) its own crumb this
    # run shouldn't have it overwritten - guards against a real fetch that
    # succeeded moments ago being clobbered by a now-stale seed value.
    monkeypatch.setenv("YF_SEED_CRUMB", "seed-crumb")
    monkeypatch.setenv("YF_SEED_COOKIES", json.dumps({"A1": "abc"}))
    fake = _FakeYfData(crumb="already-have-one")
    monkeypatch.setattr(swiss_universe_module, "YfData", lambda: fake)

    swiss_universe_module._seed_yf_session_from_env()

    assert fake._crumb == "already-have-one"
    assert dict(fake._session.cookies) == {}


def test_reseed_yf_session_overwrites_existing_crumb(monkeypatch):
    # Unlike _seed_yf_session_from_env, reseed_yf_session's whole point is
    # replacing an already-seeded, now-stale crumb - it must NOT skip just
    # because one is already present.
    fake = _FakeYfData(crumb="old-stale-crumb")
    fake._session.cookies.set("OLD", "cookie")
    monkeypatch.setattr(swiss_universe_module, "YfData", lambda: fake)

    swiss_universe_module.reseed_yf_session("fresh-crumb", {"A1": "abc"})

    assert fake._crumb == "fresh-crumb"
    assert dict(fake._session.cookies) == {"A1": "abc"}


def test_reseed_yf_session_clears_stale_cookies_not_just_adds(monkeypatch):
    # A hot reseed replaces the WHOLE cookie jar - a stale cookie from the
    # old session lingering alongside the new ones could confuse Yahoo's
    # own session validation.
    fake = _FakeYfData(crumb="old-crumb")
    fake._session.cookies.set("STALE", "leftover")
    monkeypatch.setattr(swiss_universe_module, "YfData", lambda: fake)

    swiss_universe_module.reseed_yf_session("fresh-crumb", {"NEW": "cookie"})

    assert "STALE" not in fake._session.cookies
    assert dict(fake._session.cookies) == {"NEW": "cookie"}
