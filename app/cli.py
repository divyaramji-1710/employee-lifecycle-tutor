"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="employee-lifecycle-tutor", description="New HR personnel use an interactive tutor that explains common employee lifecycle event sequences (e.g., hire -> onboarding -> performance review) and likely next steps.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Employee Lifecycle Tutor is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
