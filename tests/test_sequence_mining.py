"""Checks for ordered-window mining."""
import pytest
from app.sequence_mining import mine_sequences

def patterns(rows, lengths=(2,)):
    return {
        tuple(row["events"]): row
        for row in mine_sequences(
            rows, min_support=0.01, lengths=lengths
        )
    }

def test_reversing_events_changes_the_pattern():
    forward = patterns([{"events": ["hire", "orientation"]}])
    reverse = patterns([{"events": ["orientation", "hire"]}])

    assert ("hire", "orientation") in forward
    assert ("hire", "orientation") not in reverse
    assert ("orientation", "hire") in reverse

def test_repeated_events_keep_their_positions():
    found = patterns(
        [{"events": ["training", "review", "training"]}],
        lengths=(3,),
    )
    row = found[("training", "review", "training")]
    assert row["count"] == 1
    assert row["support"] == 1.0

def test_support_counts_windows_not_histories():
    found = patterns([
        {"events": ["a", "b", "a", "b"]},
        {"events": ["a", "c"]},
    ])
    row = found[("a", "b")]
    assert row["count"] == 2
    assert row["total_windows"] == 4
    assert row["support"] == pytest.approx(0.5)
    assert ("a", "a") not in found
