"""Descriptive paired outcomes, not an oracle deployment policy or safety proof."""
import argparse
import json
import math
from pathlib import Path
from .protocol import ARMS, CONDITIONS, STAGES


def summarize(records):
    groups = {}
    for row in records:
        key = tuple(row[k] for k in ("task", "seed", "stage", "condition"))
        if row["arm"] not in ARMS or row["condition"] not in CONDITIONS or row["stage"] not in STAGES:
            raise ValueError("Unknown treatment, stage or condition")
        if type(row["success"]) is not bool:
            raise ValueError("success must be a boolean, not a string or score")
        for field in ("environment_steps", "actual_new_calls", "wam_seconds"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid resource measure: " + field)
        arms = groups.setdefault(key, {})
        if row["arm"] in arms:
            raise ValueError("Duplicate treatment in paired context")
        arms[row["arm"]] = row
    if not groups or any(set(arms) != set(ARMS) for arms in groups.values()):
        raise ValueError("Require nonempty complete triplets; do not silently drop failures")
    totals = {a: {"complete": len(groups), "successes": 0} for a in ARMS}
    pairs = []
    for key, arms in sorted(groups.items()):
        baseline = arms["continue"]
        for arm, row in arms.items():
            totals[arm]["successes"] += int(row["success"])
        for arm in ARMS[1:]:
            row = arms[arm]
            both = baseline["success"] and row["success"]
            pairs.append(dict(zip(("task", "seed", "stage", "condition"), key),
                arm=arm, rescued=not baseline["success"] and row["success"],
                harmed=baseline["success"] and not row["success"],
                both_successful=both,
                steps_delta=row["environment_steps"] - baseline["environment_steps"] if both else None,
                branch_calls_delta=row["actual_new_calls"] - baseline["actual_new_calls"] if both else None,
                inference_seconds_delta=row["wam_seconds"] - baseline["wam_seconds"] if both else None))
    return {"contexts": len(groups), "arms": totals, "paired": pairs,
            "scope": "Descriptive development evidence. Prefix generation cost is separate; no safety or deployed-policy claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.results.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
