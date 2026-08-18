import pytest

from app.services.fusion import FUSION_TABLE, NEWS_REACTIONS, fuse, valuation_bucket

# Every (news_reaction, valuation_bucket) cell's expected (recommendation,
# base_confidence) - asserted literally so a change to FUSION_TABLE (the
# only place this service ever decides BUY/SELL/HOLD) is a deliberate,
# visible diff here, not a silent behavior change.
EXPECTED_TABLE = {
    ("good", "undervalued"): ("BUY", 0.80),
    ("good", "near_fair"): ("HOLD", 0.60),
    ("good", "overvalued"): ("HOLD", 0.65),
    ("good", "no_data"): ("HOLD", 0.55),
    ("bad", "undervalued"): ("HOLD", 0.60),
    ("bad", "near_fair"): ("SELL", 0.70),
    ("bad", "overvalued"): ("SELL", 0.80),
    ("bad", "no_data"): ("SELL", 0.55),
    ("neutral", "undervalued"): ("BUY", 0.65),
    ("neutral", "near_fair"): ("HOLD", 0.70),
    ("neutral", "overvalued"): ("SELL", 0.65),
    ("neutral", "no_data"): ("HOLD", 0.60),
    ("overreaction_down", "undervalued"): ("BUY", 0.85),
    ("overreaction_down", "near_fair"): ("BUY", 0.70),
    ("overreaction_down", "overvalued"): ("HOLD", 0.60),
    ("overreaction_down", "no_data"): ("HOLD", 0.55),
    ("overreaction_up", "undervalued"): ("HOLD", 0.60),
    ("overreaction_up", "near_fair"): ("SELL", 0.70),
    ("overreaction_up", "overvalued"): ("SELL", 0.85),
    ("overreaction_up", "no_data"): ("HOLD", 0.55),
}


def test_fusion_table_matches_expected_cells_exactly():
    assert FUSION_TABLE == EXPECTED_TABLE


def test_fusion_table_has_exactly_20_cells():
    assert len(FUSION_TABLE) == 5 * 4


# --- valuation_bucket ---


def test_valuation_bucket_none_is_no_data():
    assert valuation_bucket(None) == "no_data"


@pytest.mark.parametrize("gap_pct", [-100.0, -50.0, -15.0])
def test_valuation_bucket_undervalued_boundary_inclusive(gap_pct):
    assert valuation_bucket(gap_pct) == "undervalued"


def test_valuation_bucket_just_inside_undervalued_boundary_is_near_fair():
    assert valuation_bucket(-14.9) == "near_fair"


@pytest.mark.parametrize("gap_pct", [15.0, 50.0, 200.0])
def test_valuation_bucket_overvalued_boundary_inclusive(gap_pct):
    assert valuation_bucket(gap_pct) == "overvalued"


def test_valuation_bucket_just_inside_overvalued_boundary_is_near_fair():
    assert valuation_bucket(14.9) == "near_fair"


def test_valuation_bucket_zero_is_near_fair():
    assert valuation_bucket(0.0) == "near_fair"


# --- fuse() ---


@pytest.mark.parametrize("reaction", NEWS_REACTIONS)
@pytest.mark.parametrize("bucket,gap_pct", [
    ("undervalued", -40.0), ("near_fair", 0.0), ("overvalued", 40.0), ("no_data", None),
])
def test_fuse_matches_table_for_every_cell(reaction, bucket, gap_pct):
    expected_recommendation, expected_base_confidence = EXPECTED_TABLE[(reaction, bucket)]
    result = fuse(reaction, gap_pct)
    assert result.recommendation == expected_recommendation
    assert result.valuation_bucket == bucket
    if bucket == "no_data":
        assert result.confidence == expected_base_confidence


def test_fuse_unknown_reaction_raises_value_error():
    with pytest.raises(ValueError):
        fuse("bullish", 10.0)  # old vocabulary, no longer valid


def test_fuse_confidence_increases_with_gap_magnitude():
    small_gap = fuse("good", -20.0).confidence
    large_gap = fuse("good", -80.0).confidence
    assert large_gap > small_gap


def test_fuse_confidence_never_exceeds_0_95():
    # -1000% is absurd but shouldn't be able to push confidence past the cap.
    result = fuse("overreaction_down", -1000.0)
    assert result.confidence <= 0.95


def test_fuse_confidence_caps_gap_magnitude_contribution_at_50_points():
    # magnitude = min(abs(gap_pct) / 50, 1.0) - so anything past a 50-point
    # gap should give the identical confidence as exactly 50.
    at_fifty = fuse("bad", -50.0).confidence
    past_fifty = fuse("bad", -90.0).confidence
    assert at_fifty == past_fifty


def test_fuse_returns_a_fusion_result_namedtuple_with_expected_fields():
    result = fuse("neutral", None)
    assert result._fields == ("recommendation", "confidence", "valuation_bucket")
