"""Smoke checks for the implemented command-line tutor."""
from app.cli import main
from app.event_encoding import EVENTS

def test_hello_command(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out

def test_events_command(capsys):
    main(["events"])
    assert set(capsys.readouterr().out.splitlines()) == set(EVENTS)
