"""Project milestone entry points."""
from app.event_data import milestone_1

def milestone_2(**kwargs):
    from app.sequence_mining import milestone_2 as mine
    return mine(**kwargs)

def milestone_3():
    raise NotImplementedError("Milestone 3 is next.")

def milestone_4():
    raise NotImplementedError("Milestone 4 is next.")

def milestone_5():
    raise NotImplementedError("Milestone 5 is next.")
