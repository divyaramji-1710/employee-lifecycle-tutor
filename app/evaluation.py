"""Next-event evaluation against training-only counting baselines."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.event_encoding import EVENTS
from app.tutor import LifecycleTutor

END = "end_of_history"

def load_histories(split, data_dir="data"):
    return [
        json.loads(line)
        for line in (Path(data_dir) / f"{split}.jsonl")
            .read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

def make_cases(histories):
    cases = []
    for row in histories:
        events = row["events"]
        for length in range(1, len(events) + 1):
            cases.append({
                "history_id": row["history_id"],
                "prefix": events[:length],
                "target": events[length] if length < len(events) else END,
            })
    return cases

def rank_of(scores, target):
    # Pessimistic ties, as in the drill.
    return 1 + sum(
        score >= scores[target]
        for event, score in scores.items()
        if event != target
    )

def summarize(predictions, ranks, targets):
    count = len(targets)
    return {
        "accuracy": sum(a == b for a, b in zip(predictions, targets)) / count,
        "MRR": sum(1 / rank for rank in ranks) / count,
        "hits_at_3": sum(rank <= 3 for rank in ranks) / count,
    }

def evaluate(split="validation", model_dir="artifacts/hr-model"):
    if split not in {"validation", "test"}:
        raise ValueError("Choose validation or test.")

    training_cases = make_cases(load_histories("train"))
    cases = make_cases(load_histories(split))
    vocabulary = EVENTS + [END]

    popularity = Counter()
    transitions = defaultdict(Counter)
    for case in training_cases:
        popularity[case["target"]] += 1
        transitions[case["prefix"][-1]][case["target"]] += 1

    train_prefixes = {tuple(case["prefix"]) for case in training_cases}
    tutor = LifecycleTutor(model_dir)

    predictions = {"LM": [], "Popularity": [], "Last-event counts": []}
    ranks = {name: [] for name in predictions}
    details = []

    for number, case in enumerate(cases, 1):
        result = tutor.predict(case["prefix"], top_k=len(vocabulary))
        lm_scores = {
            item["event"]: item["conditional_score"]
            for item in result["suggestions"]
        }

        counts = transitions.get(case["prefix"][-1], popularity)
        baseline_scores = {
            "Popularity": {event: popularity[event] for event in vocabulary},
            "Last-event counts": {event: counts[event] for event in vocabulary},
        }

        predictions["LM"].append(result["unrestricted_prediction"])
        ranks["LM"].append(rank_of(lm_scores, case["target"]))

        for name, scores in baseline_scores.items():
            predictions[name].append(max(vocabulary, key=scores.get))
            ranks[name].append(rank_of(scores, case["target"]))

        details.append({
            **case,
            "prediction": result["unrestricted_prediction"],
            "top_3": [item["event"] for item in result["suggestions"][:3]],
        })

        if number % 25 == 0:
            print(f"Evaluated {number}/{len(cases)} prefixes", flush=True)

    targets = [case["target"] for case in cases]
    metrics = {
        name: summarize(predictions[name], ranks[name], targets)
        for name in predictions
    }

    report = {
        "split": split,
        "histories": len(load_histories(split)),
        "prefix_cases": len(cases),
        "unique_prefixes": len({tuple(c["prefix"]) for c in cases}),
        "prefixes_also_in_training": sum(
            tuple(c["prefix"]) in train_prefixes for c in cases
        ),
        "ranking_candidates": len(vocabulary),
        "note": (
            "LM accuracy uses unrestricted output. MRR and hits@3 rank "
            "the 24 events plus end-of-history, with pessimistic ties. "
            "Prefixes from the same history are correlated."
        ),
        "metrics": metrics,
        "predictions": details,
    }

    destination = Path("reports")
    destination.mkdir(exist_ok=True)
    path = destination / f"{split}_evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{split}: {report['histories']} histories, {len(cases)} prefixes")
    print("Unique prefixes:", report["unique_prefixes"])
    print("Prefix cases also seen in training:", report["prefixes_also_in_training"])
    print(json.dumps(metrics, indent=2))
    print("Saved:", path)
    return report
