"""Project milestone entry points."""
from app.event_data import milestone_1

def milestone_2(**kwargs):
    from app.sequence_mining import milestone_2 as mine
    return mine(**kwargs)

def milestone_3(**kwargs):
    from app.training import train_model
    return train_model(**kwargs)

def milestone_4(events, model_dir="artifacts/hr-model"):
    from app.tutor import LifecycleTutor
    return LifecycleTutor(model_dir).predict(events)

def milestone_5():
    raise NotImplementedError("Milestone 5 is next.")
