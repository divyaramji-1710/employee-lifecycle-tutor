"""Checks for synthetic ordered employee histories."""
import json
from app.event_data import milestone_1

def test_histories_are_unique_across_splits(tmp_path):
    splits = milestone_1(tmp_path)
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 48, "validation": 12, "test": 12
    }
    rows = [row for group in splits.values() for row in group]
    assert len({row["history_id"] for row in rows}) == 72
    assert len({tuple(row["events"]) for row in rows}) == 72

def test_event_order_and_saved_data(tmp_path):
    splits = milestone_1(tmp_path)
    for name, rows in splits.items():
        saved = [
            json.loads(line)
            for line in (tmp_path / f"{name}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert saved == rows
        for row in rows:
            events = row["events"]
            assert row["source"] == "synthetic"
            assert events[0] == "hire"
            assert events.index("probation_review") < events.index(
                "probation_confirmation"
            )
            if "employment_end" in events:
                assert events[-1] == "employment_end"

def test_generation_is_reproducible(tmp_path):
    assert milestone_1(tmp_path / "a") == milestone_1(tmp_path / "b")
