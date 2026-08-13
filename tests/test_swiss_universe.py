import json

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

    domestic = filter_domestic(candidates, delay_seconds=0)

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

    domestic = filter_domestic(candidates, delay_seconds=0)

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

    domestic = filter_domestic(candidates, delay_seconds=0)
    assert domestic == {}


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
    domestic = filter_domestic(candidates)
    assert domestic["REAL.SW"]["name"] == "Real Co"


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

    candidates = discover_candidates(
        min_market_cap=swiss_universe_module.ALL_CAPS_MIN_MARKET_CAP_CHF,
        max_market_cap=swiss_universe_module.ALL_CAPS_MAX_MARKET_CAP_CHF,
    )
    assert "NESN.SW" not in candidates  # Nestle
    assert "NOVN.SW" not in candidates  # Novartis
    assert "UBSG.SW" not in candidates  # UBS
    # A genuinely small/mid-cap name should still be present - this isn't
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


def test_all_caps_band_is_wider_than_small_cap_band():
    assert swiss_universe_module.ALL_CAPS_MIN_MARKET_CAP_CHF == swiss_universe_module.MIN_MARKET_CAP_CHF
    assert swiss_universe_module.ALL_CAPS_MAX_MARKET_CAP_CHF > swiss_universe_module.MAX_MARKET_CAP_CHF


def test_static_snapshot_includes_smi_large_caps_not_just_small_caps():
    # Regression guard: the static fallback used to be captured at the
    # small-cap band only (MIN/MAX_MARKET_CAP_CHF), so it was missing SMI
    # giants entirely. It's now captured at the wide all-caps band so it
    # works as a fallback for an all-caps request too - this checks a
    # couple of well-known large caps are actually present, not just that
    # the dict got bigger.
    snapshot = swiss_universe_module.STATIC_DOMESTIC_TICKER_SNAPSHOT
    assert "NESN.SW" in snapshot  # Nestle
    assert "NOVN.SW" in snapshot  # Novartis
    assert "UBSG.SW" in snapshot  # UBS


def test_discover_candidates_static_fallback_still_used_for_all_caps_bounds(monkeypatch):
    # discover_candidates' fallback doesn't filter STATIC_DOMESTIC_TICKER_
    # SNAPSHOT by the requested min/max (no market-cap data on static
    # entries to filter with - see that dict's own comment) - this just
    # confirms calling with the all-caps bounds still falls back cleanly
    # rather than erroring.
    def _boom(*a, **kw):
        raise Exception("simulated screener outage")

    monkeypatch.setattr(swiss_universe_module.yf, "screen", _boom)
    monkeypatch.setattr(swiss_universe_module.yf, "EquityQuery", lambda *a, **k: None)

    candidates = discover_candidates(
        min_market_cap=swiss_universe_module.ALL_CAPS_MIN_MARKET_CAP_CHF,
        max_market_cap=swiss_universe_module.ALL_CAPS_MAX_MARKET_CAP_CHF,
    )
    assert "INRN.SW" in candidates  # non-SMI name, present regardless of band


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
