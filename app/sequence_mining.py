"""Mine ordered contiguous event windows using thilimarketbasket."""
import json
from pathlib import Path
from thilimarketbasket.mine import frequent_itemsets

def mine_sequences(histories, min_support=0.03, lengths=(2, 3)):
    results = []

    for length in lengths:
        windows = [
            row["events"][start:start + length]
            for row in histories
            for start in range(len(row["events"]) - length + 1)
        ]
        if not windows:
            continue

        transactions = [
            [f"{position}:{event}" for position, event in enumerate(window)]
            for window in windows
        ]

        found = frequent_itemsets(
            transactions,
            min_support=min_support,
            max_size=length,
        )

        for itemset, support in found.items():
            if len(itemset) != length:
                continue

            ordered = sorted(
                (int(token.split(":", 1)[0]), token.split(":", 1)[1])
                for token in itemset
            )
            if [position for position, _ in ordered] != list(range(length)):
                continue

            results.append({
                "events": [event for _, event in ordered],
                "length": length,
                "count": round(support * len(windows)),
                "total_windows": len(windows),
                "support": support,
            })

    return sorted(
        results,
        key=lambda row: (-row["support"], row["length"], row["events"]),
    )

def milestone_2(data_dir="data", output_dir="reports", min_support=0.03):
    path = Path(data_dir) / "train.jsonl"
    histories = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    results = mine_sequences(histories, min_support=min_support)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "frequent_sequences.json").write_text(
        json.dumps({
            "source_split": "train",
            "support_definition": "fraction of contiguous windows of the same length",
            "min_support": min_support,
            "patterns": results,
        }, indent=2),
        encoding="utf-8",
    )

    print("Training histories:", len(histories))
    print("Frequent ordered patterns:", len(results))
    for row in results[:12]:
        print(
            f"{row['support']:.2%} "
            f"({row['count']}/{row['total_windows']} windows) | "
            + " -> ".join(row["events"])
        )
    return results
