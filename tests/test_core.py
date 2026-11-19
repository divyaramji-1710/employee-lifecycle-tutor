"""Check the implemented milestone entry points."""
from app import core

def test_all_milestone_entry_points_exist():
    for number in range(1, 6):
        assert callable(getattr(core, f"milestone_{number}"))
