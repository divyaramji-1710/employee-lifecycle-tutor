"""Evaluation checks without loading a model."""
import pytest
from app.evaluation import END, make_cases, rank_of, summarize

def test_cases_preserve_prefixes_and_include_end():
    cases = make_cases([{
        "history_id": "example",
        "events": ["hire", "orientation", "training"],
    }])
    assert [case["prefix"] for case in cases] == [
        ["hire"],
        ["hire", "orientation"],
        ["hire", "orientation", "training"],
    ]
    assert [case["target"] for case in cases] == [
        "orientation", "training", END
    ]

def test_ties_count_against_target():
    assert rank_of({"a": 5, "b": 5, "c": 1}, "a") == 2

def test_summary_metrics():
    result = summarize(
        ["a", "wrong", "c"],
        [1, 4, 2],
        ["a", "b", "c"],
    )
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["MRR"] == pytest.approx((1 + 1/4 + 1/2) / 3)
    assert result["hits_at_3"] == pytest.approx(2 / 3)
