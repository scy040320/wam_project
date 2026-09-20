"""Summarize frozen-split CF-WAM method metrics with paired bootstrap CIs.

Input is a CSV with one row per matched initial state and columns:
task_id, episode_id, method, <one-or-more numeric metric columns>.
The script never chooses thresholds or modifies raw records.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean


REQUIRED = {"task_id", "episode_id", "method"}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"CSV must include {sorted(REQUIRED)}")
        metric_names = [name for name in reader.fieldnames if name not in REQUIRED]
        if not metric_names:
            raise ValueError("CSV must include at least one numeric metric column.")
        rows = list(reader)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["method"])].append(row)

    rng = random.Random(args.seed)
    result: dict[str, object] = {"input": str(args.input_csv), "bootstrap_samples": args.bootstrap, "seed": args.seed, "groups": {}}
    for (task_id, method), group in sorted(grouped.items()):
        summary = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            bootstrap_means = [fmean(rng.choice(values) for _ in values) for _ in range(args.bootstrap)]
            summary[metric] = {
                "n_matched_states": len(values),
                "mean": fmean(values),
                "ci95": [percentile(bootstrap_means, 0.025), percentile(bootstrap_means, 0.975)],
            }
        result["groups"][f"{task_id}/{method}"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
