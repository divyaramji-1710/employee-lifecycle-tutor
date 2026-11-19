"""Synthetic employee histories for an educational tutor."""
import itertools
import json
import random
from pathlib import Path

def build_histories():
    onboarding = [
        ["orientation", "mandatory_training"],
        ["orientation", "mentor_assignment", "mandatory_training"],
        ["orientation", "mandatory_training", "mentor_assignment"],
    ]
    development = [
        ["role_training"],
        ["role_training", "skills_training"],
        ["role_training", "project_assignment", "skills_training"],
    ]
    reviews = [
        ["probation_review", "probation_confirmation", "performance_review"],
        ["probation_review", "probation_extension", "follow_up_review",
         "probation_confirmation", "performance_review"],
    ]
    outcomes = [
        ["development_plan", "skills_training", "performance_review"],
        ["promotion_review", "promotion", "leadership_training"],
        ["transfer_request", "transfer_review", "internal_transfer",
         "role_training"],
        ["resignation", "notice_period", "handover", "exit_interview",
         "employment_end"],
    ]

    histories = []
    for number, (start, learning, review, outcome) in enumerate(
        itertools.product(onboarding, development, reviews, outcomes)
    ):
        histories.append({
            "history_id": f"synthetic-{number:03d}",
            "events": ["hire"] + start + learning + review + outcome,
            "source": "synthetic",
        })
    return histories

def milestone_1(output_dir="data", seed=42):
    histories = build_histories()
    random.Random(seed).shuffle(histories)

    splits = {
        "train": histories[:48],
        "validation": histories[48:60],
        "test": histories[60:],
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for name, rows in splits.items():
        content = "".join(
            json.dumps(row) + "\n" for row in rows
        )
        (destination / f"{name}.jsonl").write_text(
            content, encoding="utf-8"
        )
    return splits
