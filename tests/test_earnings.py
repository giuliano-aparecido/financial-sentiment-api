from app.services.earnings import earnings_block

FULL_EARNINGS = {
    "last_quarter_date": "2026-06-30",
    "revenue": 85.8e9,
    "yoy_growth_pct": 4.9,
    "eps_actual": 1.40,
    "eps_estimate": 1.35,
    "next_earnings_date": "2026-10-29",
}


def test_earnings_block_renders_full_shape_with_beat():
    block = earnings_block(FULL_EARNINGS)
    assert block == (
        "Last Quarter (2026-06-30): Revenue $85.8B (+4.9% YoY), EPS $1.40 (beat est. $1.35)\n"
        "Next Earnings Date: 2026-10-29"
    )


def test_earnings_block_reports_miss():
    earnings = {**FULL_EARNINGS, "eps_actual": 1.20}
    assert "missed est. $1.35" in earnings_block(earnings)


def test_earnings_block_reports_in_line():
    earnings = {**FULL_EARNINGS, "eps_actual": 1.35}
    assert "in line with est. $1.35" in earnings_block(earnings)


def test_earnings_block_omits_yoy_when_unavailable():
    earnings = {**FULL_EARNINGS, "yoy_growth_pct": None}
    block = earnings_block(earnings)
    assert "YoY" not in block
    assert "Revenue $85.8B, EPS" in block


def test_earnings_block_omits_eps_note_when_estimate_missing():
    earnings = {**FULL_EARNINGS, "eps_actual": None, "eps_estimate": None}
    block = earnings_block(earnings)
    assert block.startswith("Last Quarter (2026-06-30): Revenue $85.8B (+4.9% YoY)\n")
    assert "EPS" not in block


def test_earnings_block_omits_next_date_when_unavailable():
    earnings = {**FULL_EARNINGS, "next_earnings_date": None}
    assert "Next Earnings Date" not in earnings_block(earnings)


def test_earnings_block_data_unavailable_when_none():
    assert earnings_block(None) == "Data unavailable."


def test_earnings_block_data_unavailable_when_revenue_missing():
    assert earnings_block({"revenue": None}) == "Data unavailable."


def test_earnings_block_negative_yoy_has_no_plus_sign():
    earnings = {**FULL_EARNINGS, "yoy_growth_pct": -5.6}
    assert "(-5.6% YoY)" in earnings_block(earnings)
