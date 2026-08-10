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
