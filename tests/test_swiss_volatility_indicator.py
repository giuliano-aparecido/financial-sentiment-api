import pandas as pd
import pytest

import app.services.swiss_crash_rebound as crash_rebound_module
import app.services.swiss_volatility_indicator as indicator_module
from app.services.swiss_volatility_indicator import find_volatility_days, run_scan


def _close_frame(dates, closes):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1000] * len(closes)},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def _fake_download(symbols, **kwargs):
    # Same shape contract as test_swiss_crash_rebound.py's fake - FLAT for
    # a single symbol, multi-indexed for more than one.
    if len(symbols) == 1:
        return crash_rebound_module._TEST_FRAMES[symbols[0]]
    frames = {symbol: crash_rebound_module._TEST_FRAMES[symbol] for symbol in symbols}
    return pd.concat(frames, axis=1)


@pytest.fixture
def today():
    return pd.Timestamp.today().normalize()


@pytest.fixture(autouse=True)
def patch_download(monkeypatch):
    # download_ohlcv_chunked lives in swiss_crash_rebound.py and is
    # imported by name into swiss_volatility_indicator.py - patch the
    # ORIGINAL module's yf.download (what download_ohlcv_chunked actually
    # calls), not a nonexistent copy on this module.
    monkeypatch.setattr(crash_rebound_module.yf, "download", _fake_download)


def _domestic_entry(name="Test AG", sector="Industrials", market_cap=1e9):
    return {"name": name, "sector": sector, "market_cap": market_cap}


def test_find_volatility_days_counts_loss_and_gain_days(today):
    dates = pd.date_range(end=today, periods=10, freq="B")
    # Day changes vs previous close: flat, flat, -3% (loss), flat, +3%
    # (gain), flat, -4% (loss), flat, flat, +5% (gain)
    closes = [100.0]
    for pct in [0, 0, -3, 0, 3, 0, -4, 0, 0, 5]:
        closes.append(round(closes[-1] * (1 + pct / 100), 4))
    closes = closes[1:]
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _close_frame(dates, closes)}

    domestic = {"TEST.SW": _domestic_entry()}
    results = find_volatility_days(["TEST.SW"], domestic, lookback_months=12,
                                    history_period="13mo", threshold_pct=2.0)

    assert len(results) == 1
    row = results.iloc[0]
    assert row["loss_days"] == 2
    assert row["gain_days"] == 2
    assert row["total_days"] == 4


def test_find_volatility_days_excludes_days_below_threshold(today):
    dates = pd.date_range(end=today, periods=5, freq="B")
    closes = [100.0]
    for pct in [0, -1.5, 0, 1.5, 0]:  # all below a 2% threshold
        closes.append(round(closes[-1] * (1 + pct / 100), 4))
    closes = closes[1:]
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _close_frame(dates, closes)}

    domestic = {"TEST.SW": _domestic_entry()}
    results = find_volatility_days(["TEST.SW"], domestic, lookback_months=12,
                                    history_period="13mo", threshold_pct=2.0)
    assert results.empty


def test_find_volatility_days_includes_at_exactly_threshold(today):
    dates = pd.date_range(end=today, periods=3, freq="B")
    closes = [100.0, 98.0, 99.96]  # day1 exactly -2.0%, day2 exactly +2.0%
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _close_frame(dates, closes)}

    domestic = {"TEST.SW": _domestic_entry()}
    results = find_volatility_days(["TEST.SW"], domestic, lookback_months=12,
                                    history_period="13mo", threshold_pct=2.0)
    assert len(results) == 1
    assert results.iloc[0]["loss_days"] == 1
    assert results.iloc[0]["gain_days"] == 1


def test_find_volatility_days_excludes_company_with_only_loss_days(today):
    # Regression test: previously a company with ANY qualifying day of
    # EITHER kind was included (loss_days=0 AND gain_days=0 was the only
    # exclusion) - changed 2026-08-19 at the user's explicit request to
    # require BOTH kinds present, since "brings only losses, never a
    # qualifying gain" isn't what a volatility-in-both-directions
    # indicator is supposed to surface.
    dates = pd.date_range(end=today, periods=10, freq="B")
    closes = [100.0]
    for pct in [0, -3, 0, -4, 0, 0, 0, 0, 0, 0]:  # only losses, never a qualifying gain
        closes.append(round(closes[-1] * (1 + pct / 100), 4))
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _close_frame(dates, closes[1:])}

    domestic = {"TEST.SW": _domestic_entry()}
    results = find_volatility_days(["TEST.SW"], domestic, lookback_months=12,
                                    history_period="13mo", threshold_pct=2.0)
    assert results.empty


def test_find_volatility_days_excludes_company_with_only_gain_days(today):
    dates = pd.date_range(end=today, periods=10, freq="B")
    closes = [100.0]
    for pct in [0, 3, 0, 4, 0, 0, 0, 0, 0, 0]:  # only gains, never a qualifying loss
        closes.append(round(closes[-1] * (1 + pct / 100), 4))
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _close_frame(dates, closes[1:])}

    domestic = {"TEST.SW": _domestic_entry()}
    results = find_volatility_days(["TEST.SW"], domestic, lookback_months=12,
                                    history_period="13mo", threshold_pct=2.0)
    assert results.empty


def test_find_volatility_days_excludes_matches_outside_lookback_window(today):
    dates = pd.date_range(end=today, periods=80, freq="B")
    closes = [100.0] * 80
    closes[1] = 90.0  # -10% - old, well outside a 1-month lookback
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _close_frame(dates, closes)}

    domestic = {"TEST.SW": _domestic_entry()}
    results = find_volatility_days(["TEST.SW"], domestic, lookback_months=1,
                                    history_period="13mo", threshold_pct=2.0)
    assert results.empty


def test_find_volatility_days_omits_companies_with_zero_qualifying_days(today):
    dates = pd.date_range(end=today, periods=10, freq="B")
    volatile_closes = [100.0]
    for pct in [0, -5, 0, 5, 0, 0, 0, 0, 0, 0]:  # both a qualifying loss AND gain day
        volatile_closes.append(round(volatile_closes[-1] * (1 + pct / 100), 4))
    flat_closes = [100.0] * 10
    crash_rebound_module._TEST_FRAMES = {
        "VOLATILE.SW": _close_frame(dates, volatile_closes[1:]),
        "FLAT.SW": _close_frame(dates, flat_closes),
    }

    domestic = {"VOLATILE.SW": _domestic_entry("Volatile AG"), "FLAT.SW": _domestic_entry("Flat AG")}
    results = find_volatility_days(["VOLATILE.SW", "FLAT.SW"], domestic, lookback_months=12,
                                    history_period="13mo", threshold_pct=2.0)

    assert list(results["ticker"]) == ["VOLATILE.SW"]


def test_find_volatility_days_sorted_by_total_days_descending(today):
    dates = pd.date_range(end=today, periods=10, freq="B")

    def _closes(loss_pcts):
        c = [100.0]
        for pct in loss_pcts:
            c.append(round(c[-1] * (1 + pct / 100), 4))
        return c[1:]

    crash_rebound_module._TEST_FRAMES = {
        "MOST.SW": _close_frame(dates, _closes([0, -3, 3, -3, 3, -3, 3, 0, 0, 0])),   # 6 qualifying days (3 loss, 3 gain)
        "MID.SW": _close_frame(dates, _closes([0, -3, 3, -3, 0, 0, 0, 0, 0, 0])),      # 3 qualifying days (2 loss, 1 gain)
        "LEAST.SW": _close_frame(dates, _closes([0, -3, 3, 0, 0, 0, 0, 0, 0, 0])),     # 2 qualifying days (1 loss, 1 gain)
    }
    domestic = {t: _domestic_entry() for t in ["MOST.SW", "MID.SW", "LEAST.SW"]}
    results = find_volatility_days(["MOST.SW", "MID.SW", "LEAST.SW"], domestic, lookback_months=12,
                                    history_period="13mo", threshold_pct=2.0)

    assert list(results["ticker"]) == ["MOST.SW", "MID.SW", "LEAST.SW"]


def test_run_scan_returns_empty_dataframe_when_no_symbols():
    assert run_scan({}, threshold_pct=2.0).empty


def test_run_scan_attaches_company_fields(today):
    dates = pd.date_range(end=today, periods=4, freq="B")
    crash_rebound_module._TEST_FRAMES = {"TEST.SW": _close_frame(dates, [100.0, 94.0, 94.0, 98.7])}

    domestic = {"TEST.SW": {"name": "Test AG", "sector": "Healthcare", "market_cap": 5e9}}
    results = run_scan(domestic, threshold_pct=5.0)

    row = results.iloc[0]
    assert row["name"] == "Test AG"
    assert row["sector"] == "Healthcare"
    assert row["market_cap"] == 5e9
    assert row["loss_days"] == 1
    assert row["gain_days"] == 1
    assert row["total_days"] == 2


def test_allowed_threshold_pcts_are_2_3_5():
    # Regression guard for the frontend's fixed selector options - a
    # change here silently desyncs the UI's dropdown from what the
    # backend actually validates.
    assert indicator_module.ALLOWED_THRESHOLD_PCTS == (2.0, 3.0, 5.0)
