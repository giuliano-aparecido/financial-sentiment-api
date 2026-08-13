from app.services.swiss_today_screener import find_big_loss, run_scan


def _domestic_entry(name, sector, market_cap, change_pct, volume_today, avg_volume_3mo, price=100.0):
    return {
        "name": name,
        "sector": sector,
        "market_cap": market_cap,
        "quote": {
            "regularMarketChangePercent": change_pct,
            "regularMarketVolume": volume_today,
            "averageDailyVolume3Month": avg_volume_3mo,
            "regularMarketPrice": price,
        },
    }


def test_find_big_loss_filters_to_threshold():
    domestic = {
        "DROP.SW": _domestic_entry("Drop AG", "Industrials", 1e9, -6.0, 1000, 5000),
        "FLAT.SW": _domestic_entry("Flat AG", "Industrials", 1e9, -1.0, 1000, 5000),
        "GAIN.SW": _domestic_entry("Gain AG", "Industrials", 1e9, 3.0, 1000, 5000),
    }
    results = find_big_loss(domestic, loss_threshold=-5.0)
    assert list(results["ticker"]) == ["DROP.SW"]


def test_find_big_loss_includes_exactly_at_threshold():
    domestic = {"DROP.SW": _domestic_entry("Drop AG", "Industrials", 1e9, -5.0, 1000, 5000)}
    results = find_big_loss(domestic, loss_threshold=-5.0)
    assert len(results) == 1


def test_find_big_loss_skips_missing_live_data():
    domestic = {
        "NOQUOTE.SW": {
            "name": "No Quote AG", "sector": "Industrials", "market_cap": 1e9,
            "quote": {"regularMarketChangePercent": None, "regularMarketVolume": None},
        },
    }
    results = find_big_loss(domestic, loss_threshold=-5.0)
    assert results.empty


def test_find_big_loss_does_not_filter_by_volume():
    # A big loss on very thin volume and a big loss on heavy volume must
    # BOTH show up - volume is a sort key, never an exclusion (see module
    # docstring).
    domestic = {
        "THIN.SW": _domestic_entry("Thin AG", "Industrials", 1e9, -6.0, 10, 5000),      # ratio 0.002
        "THICK.SW": _domestic_entry("Thick AG", "Industrials", 1e9, -6.0, 50000, 5000),  # ratio 10.0
    }
    results = find_big_loss(domestic, loss_threshold=-5.0)
    assert set(results["ticker"]) == {"THIN.SW", "THICK.SW"}


def test_find_big_loss_sorted_by_volume_ratio_then_loss():
    domestic = {
        "THICK.SW": _domestic_entry("Thick AG", "Industrials", 1e9, -10.0, 5000, 5000),   # ratio 1.0
        "THIN.SW": _domestic_entry("Thin AG", "Industrials", 1e9, -5.5, 100, 5000),        # ratio 0.02
        "MID.SW": _domestic_entry("Mid AG", "Industrials", 1e9, -20.0, 2500, 5000),        # ratio 0.5
    }
    results = find_big_loss(domestic, loss_threshold=-5.0)
    assert list(results["ticker"]) == ["THIN.SW", "MID.SW", "THICK.SW"]


def test_find_big_loss_none_when_no_matches():
    domestic = {"FLAT.SW": _domestic_entry("Flat AG", "Industrials", 1e9, -1.0, 1000, 5000)}
    results = find_big_loss(domestic, loss_threshold=-5.0)
    assert results.empty


def test_run_scan_uses_module_default_threshold():
    domestic = {"DROP.SW": _domestic_entry("Drop AG", "Industrials", 1e9, -6.0, 1000, 5000)}
    results = run_scan(domestic)
    assert list(results["ticker"]) == ["DROP.SW"]
