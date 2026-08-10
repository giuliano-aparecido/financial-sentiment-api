from app.services.swiss_small_cap_today_screener import build_today_snapshot, run_scan


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


def test_build_today_snapshot_includes_losers_gainers_and_flat():
    domestic = {
        "DROP.SW": _domestic_entry("Drop AG", "Industrials", 1e9, -6.0, 1000, 5000),
        "FLAT.SW": _domestic_entry("Flat AG", "Industrials", 1e9, -1.0, 1000, 5000),
        "GAIN.SW": _domestic_entry("Gain AG", "Industrials", 1e9, 3.0, 1000, 5000),
    }
    results = build_today_snapshot(domestic)
    assert set(results["ticker"]) == {"DROP.SW", "FLAT.SW", "GAIN.SW"}


def test_build_today_snapshot_skips_missing_live_data():
    domestic = {
        "NOQUOTE.SW": {
            "name": "No Quote AG", "sector": "Industrials", "market_cap": 1e9,
            "quote": {"regularMarketChangePercent": None, "regularMarketVolume": None},
        },
    }
    results = build_today_snapshot(domestic)
    assert results.empty


def test_build_today_snapshot_sorted_by_volume_ratio_then_change():
    domestic = {
        "THICK.SW": _domestic_entry("Thick AG", "Industrials", 1e9, -10.0, 5000, 5000),   # ratio 1.0
        "THIN.SW": _domestic_entry("Thin AG", "Industrials", 1e9, -5.5, 100, 5000),        # ratio 0.02
        "MID.SW": _domestic_entry("Mid AG", "Industrials", 1e9, -20.0, 2500, 5000),        # ratio 0.5
        "GAINER.SW": _domestic_entry("Gainer AG", "Industrials", 1e9, 8.0, 100, 5000),      # ratio 0.02, tie with THIN
    }
    results = build_today_snapshot(domestic)
    # Thinnest volume first; within the THIN.SW/GAINER.SW tie (both ratio
    # 0.02), the bigger (more negative) change comes first.
    assert list(results["ticker"]) == ["THIN.SW", "GAINER.SW", "MID.SW", "THICK.SW"]


def test_build_today_snapshot_empty_when_no_live_data_for_any_ticker():
    domestic = {
        "NOQUOTE.SW": {
            "name": "No Quote AG", "sector": "Industrials", "market_cap": 1e9,
            "quote": {"regularMarketChangePercent": None, "regularMarketVolume": None},
        },
    }
    assert build_today_snapshot(domestic).empty


def test_build_today_snapshot_volume_ratio_none_when_no_3mo_average():
    domestic = {
        "NEWLIST.SW": _domestic_entry("Newly Listed AG", "Industrials", 1e9, 2.0, 500, None),
    }
    results = build_today_snapshot(domestic)
    assert results.iloc[0]["volume_vs_3mo_avg"] is None


def test_run_scan_returns_full_universe_not_a_filtered_subset():
    domestic = {
        "DROP.SW": _domestic_entry("Drop AG", "Industrials", 1e9, -6.0, 1000, 5000),
        "GAIN.SW": _domestic_entry("Gain AG", "Industrials", 1e9, 3.0, 1000, 5000),
    }
    results = run_scan(domestic)
    assert set(results["ticker"]) == {"DROP.SW", "GAIN.SW"}
