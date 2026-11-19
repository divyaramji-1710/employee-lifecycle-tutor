"""Command-line interface for the Employee Lifecycle Tutor."""
import argparse

def parse_events(text):
    events = [part.strip() for part in text.split(",")]
    if not all(events):
        raise ValueError("Use non-empty event names separated by commas.")
    return events

def show_prediction(tutor, events):
    from app.explanations import explain

    result = tutor.predict(events)
    print("\nHistory:", " -> ".join(events))
    print("Unrestricted model prediction:", result["unrestricted_prediction"])
    print("Ranked event suggestions (conditional model scores):")

    for suggestion in result["suggestions"]:
        event = suggestion["event"]
        print(f"  {event}: {suggestion['conditional_score']:.3f}")
        print("    Written learning note:", explain(event))

    if result["unrestricted_prediction"] == "unknown":
        print("The unrestricted output was not a recognized event; review manually.")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="employee-lifecycle-tutor")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello")
    sub.add_parser("events")

    explanation = sub.add_parser("explain")
    explanation.add_argument("event")

    prediction = sub.add_parser("predict")
    prediction.add_argument("history", help="Comma-separated ordered events")
    prediction.add_argument("--model-dir", default="artifacts/hr-model")

    interactive = sub.add_parser("interactive")
    interactive.add_argument("--model-dir", default="artifacts/hr-model")

    args = parser.parse_args(argv)

    try:
        if args.command == "hello":
            print("Employee Lifecycle Tutor is ready.")
        elif args.command == "events":
            from app.event_encoding import EVENTS
            print("\n".join(EVENTS))
        elif args.command == "explain":
            from app.explanations import explain
            print("Written learning note:", explain(args.event))
        else:
            from app.tutor import LifecycleTutor
            tutor = LifecycleTutor(args.model_dir)
            print("Educational prototype using synthetic histories.")
            print("Suggestions are not company policy or employment advice.")
            print("Model scores are not calibrated confidence.")

            if args.command == "predict":
                show_prediction(tutor, parse_events(args.history))
            else:
                print("Enter comma-separated events, 'explain EVENT', or 'quit'.")
                while True:
                    try:
                        text = input("\nHistory> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nTutor closed.")
                        break
                    if text.lower() in {"quit", "exit"}:
                        break
                    try:
                        if text.startswith("explain "):
                            from app.explanations import explain
                            print("Written learning note:", explain(text[8:].strip()))
                        else:
                            show_prediction(tutor, parse_events(text))
                    except ValueError as error:
                        print("Input error:", error)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")

if __name__ == "__main__":
    main()
