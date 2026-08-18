import json
import threading
import time

import app.services.swiss_universe as swiss_universe_module
from app.services.swiss_universe import _ex_dividend_date, discover_candidates, filter_domestic


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

    domestic, failed = filter_domestic(candidates, delay_seconds=0)

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

    domestic, failed = filter_domestic(candidates, delay_seconds=0)

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

    domestic, failed = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == []  # a legitimate exclusion (wrong domicile), not a fetch failure - must not be retried


def test_filter_domestic_excludes_below_min_avg_daily_volume_10d(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(averageDailyVolume10Day=swiss_universe_module.MIN_AVG_DAILY_VOLUME_10D - 1)),
    )
    candidates = {"THIN.SW": {"longName": "Thin AG", "marketCap": 1e9}}

    domestic, failed = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == []  # legitimate exclusion (thin volume), not a fetch failure


def test_filter_domestic_includes_at_exactly_min_avg_daily_volume_10d(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(averageDailyVolume10Day=swiss_universe_module.MIN_AVG_DAILY_VOLUME_10D)),
    )
    candidates = {"EDGE.SW": {"longName": "Edge AG", "marketCap": 1e9}}

    domestic, failed = filter_domestic(candidates, delay_seconds=0)
    assert "EDGE.SW" in domestic
    assert failed == []


def test_filter_domestic_excludes_missing_avg_daily_volume_10d(monkeypatch):
    info = _base_info()
    del info["averageDailyVolume10Day"]
    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", lambda symbol: _FakeTicker(info))
    candidates = {"NOVOL.SW": {"longName": "No Volume AG", "marketCap": 1e9}}

    domestic, failed = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == []  # legitimate exclusion (no volume data), not a fetch failure


def test_filter_domestic_captures_avg_volume_10d(monkeypatch):
    monkeypatch.setattr(
        swiss_universe_module.yf, "Ticker",
        lambda symbol: _FakeTicker(_base_info(averageDailyVolume10Day=250_000)),
    )
    candidates = {"TEST.SW": {"longName": "Test AG", "marketCap": 1e9}}

    domestic, failed = filter_domestic(candidates, delay_seconds=0)
    assert domestic["TEST.SW"]["avg_volume_10d"] == 250_000


def test_filter_domestic_handles_many_tickers_correctly_under_concurrency(monkeypatch):
    # Not a timing/speed assertion - just confirms results are complete
    # and correct (nothing dropped/duplicated/mixed up between symbols)
    # when run through the thread pool instead of a serial loop.
    def fake_ticker(symbol):
        return _FakeTicker(_base_info(sector=symbol))  # sector encodes which symbol answered

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {f"T{i}.SW": {"longName": f"Company {i}", "marketCap": 1e9} for i in range(12)}

    domestic, failed = filter_domestic(candidates, delay_seconds=0, max_workers=5)

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
        raise Exception("YFRateLimitError: Too Many Requests")

    monkeypatch.setattr(swiss_universe_module.yf, "Ticker", fake_ticker)
    candidates = {"RATELIMITED.SW": {"longName": "Rate Limited AG", "marketCap": 1e9}}

    domestic, failed = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}
    assert failed == ["RATELIMITED.SW"]


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

    domestic, failed = filter_domestic(candidates, delay_seconds=0)
    assert "OK.SW" in domestic
    assert "BROKEN.SW" not in domestic
    assert failed == ["BROKEN.SW"]


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
    domestic, failed = filter_domestic(candidates)
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


def test_smi_tickers_are_excluded_from_excluded_tickers():
    # SMI_TICKERS must actually take effect via EXCLUDED_TICKERS, not just
    # exist as an unused set - both discover_candidates code paths filter
    # on EXCLUDED_TICKERS (see _discover_candidates_live and the static
    # fallback branch), so this is the one thing that has to be true for
    # either path to actually exclude them.
    assert swiss_universe_module.SMI_TICKERS <= swiss_universe_module.EXCLUDED_TICKERS


def test_smi_tickers_all_present_in_static_snapshot():
    # Sanity check against typos in SMI_TICKERS (hand-maintained, see its
    # own comment) - every symbol in it should be a real ticker that
    # actually showed up in a live capture, not a guessed/misremembered
    # one.
    missing = swiss_universe_module.SMI_TICKERS - set(swiss_universe_module.STATIC_DOMESTIC_TICKER_SNAPSHOT)
    assert missing == set()


def test_discover_candidates_static_fallback_excludes_smi_names(monkeypatch):
    def _boom(*a, **kw):
        raise Exception("simulated screener outage")

    monkeypatch.setattr(swiss_universe_module.yf, "screen", _boom)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates()
    assert "NESN.SW" not in candidates  # Nestle
    assert "NOVN.SW" not in candidates  # Novartis
    assert "UBSG.SW" not in candidates  # UBS
    # A genuinely non-SMI name should still be present - this isn't
    # asserting the fallback returns an empty dict.
    assert "INRN.SW" in candidates


def test_discover_candidates_live_excludes_smi_names(monkeypatch):
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
    assert "NESN.SW" not in candidates
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
