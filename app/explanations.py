"""Written educational notes, separate from model predictions."""
NOTES = {
    "hire": "The synthetic employment history begins with hiring.",
    "orientation": "Introduce workplace processes, contacts, and resources.",
    "mandatory_training": "Complete the organization's required introductory training.",
    "mentor_assignment": "Assign a colleague to support learning and questions.",
    "role_training": "Learn the responsibilities and tools of the role.",
    "skills_training": "Develop skills relevant to current or future work.",
    "project_assignment": "Apply learning through an assigned project.",
    "probation_review": "Review initial progress against the role's expectations.",
    "probation_confirmation": "Record completion of the simulated probation stage.",
    "probation_extension": "Continue the simulated probation period with follow-up.",
    "follow_up_review": "Review progress after an earlier assessment.",
    "performance_review": "Discuss performance, feedback, and development goals.",
    "development_plan": "Agree learning goals and actions for development.",
    "promotion_review": "Assess readiness for increased responsibilities.",
    "promotion": "Move into a role with increased responsibilities.",
    "leadership_training": "Develop skills for leadership responsibilities.",
    "transfer_request": "Request a move to another role or team.",
    "transfer_review": "Review the proposed move and its requirements.",
    "internal_transfer": "Move to another role within the organization.",
    "resignation": "Begin the simulated employee-initiated departure path.",
    "notice_period": "Prepare for departure during the applicable notice period.",
    "handover": "Transfer responsibilities and relevant work information.",
    "exit_interview": "Collect feedback about the employee's experience.",
    "employment_end": "The simulated employment relationship ends.",
    "end_of_history": "The model predicts the end of this recorded history.",
}

def explain(event):
    if event not in NOTES:
        raise ValueError(f"Unknown event: {event}")
    return NOTES[event]
