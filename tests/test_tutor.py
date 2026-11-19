"""Tutor interface checks without model downloads."""
import pytest

from app.cli import main, parse_events
from app.event_encoding import EVENTS
from app.explanations import NOTES, explain
from app.tutor import LifecycleTutor

def test_parse_preserves_order_and_repeated_events():
    assert parse_events("hire, skills_training, skills_training") == [
        "hire", "skills_training", "skills_training"
    ]

@pytest.mark.parametrize("text", ["", "hire,", ",hire", "hire,,orientation"])
def test_parse_rejects_empty_events(text):
    with pytest.raises(ValueError):
        parse_events(text)

def test_every_event_has_a_written_note():
    assert set(EVENTS) <= set(NOTES)
    assert all(explain(event).strip() for event in EVENTS)

def test_unknown_explanation_is_rejected():
    with pytest.raises(ValueError, match="Unknown event"):
        explain("unknown_event")

def test_explain_command_labels_written_content(capsys):
    main(["explain", "probation_review"])
    assert "Written learning note:" in capsys.readouterr().out

def test_empty_history_is_rejected_before_model_access():
    tutor = LifecycleTutor.__new__(LifecycleTutor)
    with pytest.raises(ValueError, match="at least one event"):
        tutor.predict([])

def test_unknown_event_is_rejected_before_model_access():
    tutor = LifecycleTutor.__new__(LifecycleTutor)
    tutor.event_ids = {"hire": 1}
    with pytest.raises(ValueError, match="Unknown events"):
        tutor.predict(["unknown_event"])
