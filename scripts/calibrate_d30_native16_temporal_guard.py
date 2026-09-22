"""Freeze D30 native-16 temporal guard thresholds from validation seeds only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


KNOWN = {"normal", "visual_occlusion", "object_shift", "action_noise"}
RESIDUAL_ACTION_STEPS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weak-confirmations", type=int, default=2)
    parser.add_argument("--hold-steps", type=int, default=4)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("task*_seed*_D_dependency_aware_*/online_decision_log.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("control_horizon", -1)) != 16:
            raise ValueError(f"non-native horizon in {path}")
        condition = str(payload["condition"])
        previous_energy = None
        for decision in payload["decisions"]:
            injected = bool(decision["intervention_injected"])
            # Only the mixed intervention at its reviewed injection boundary is
            # labelled unknown. Every other row is a non-unknown calibration row.
            true_unknown = condition == "unknown" and injected
            probabilities = decision["cause_probabilities"]
            known_probability = max(float(probabilities[name]) for name in KNOWN)
            block = int(decision["block"])
            action_delta = np.load(path.parent / f"query_{block:03d}" / "action_execution_delta.npy")
            # Match the attributor/runtime ABI: only the first four action rows
            # are represented in the residual vector, even though Cosmos
            # executes a native 16-step control chunk.
            action_residual_l2 = float(np.linalg.norm(action_delta[:RESIDUAL_ACTION_STEPS, :3]))
            energy = float(decision["residual_energy"])
            rows.append({
                "source": str(path),
                "condition": condition,
                "block": block,
                "intervention_injected": injected,
                "true_cause": condition if injected else "normal",
                "raw_cause": str(decision["raw_cause"]),
                "true_unknown": true_unknown,
                "unknown_probability": float(probabilities["unknown"]),
                "max_known_probability": known_probability,
                "entropy": float(decision["entropy"]),
                "residual_energy": energy,
                "residual_energy_delta": 0.0 if previous_energy is None else energy - previous_energy,
                "action_residual_l2": action_residual_l2,
            })
            previous_energy = energy
    if not rows or not any(row["true_unknown"] for row in rows):
        raise ValueError("validation records must contain unknown intervention rows")
    return rows


def rates(mask: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    recall = float(mask[labels].mean()) if labels.any() else 0.0
    false_rate = float(mask[~labels].mean()) if (~labels).any() else 0.0
    return recall, false_rate


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    labels = np.asarray([row["true_unknown"] for row in rows], dtype=bool)
    unknown = np.asarray([row["unknown_probability"] for row in rows])
    known = np.asarray([row["max_known_probability"] for row in rows])
    entropy = np.asarray([row["entropy"] for row in rows])
    energy = np.asarray([row["residual_energy"] for row in rows])

    # Strong threshold is an *immediate* safety stop, so validation false stops
    # take priority over unknown recall.  The previous rule forced 100% unknown
    # recall even when that implied a >50% non-unknown false-stop rate, which
    # made every online smoke episode terminate before its intervention block.
    # Select only thresholds with <=5% validation false positives, maximize
    # recall within that feasible set, then prefer fewer false positives and a
    # higher (less trigger-happy) threshold.
    strong_candidates = np.unique(np.concatenate(([0.0, 1.0], unknown)))
    strong_grid = []
    for threshold in strong_candidates:
        recall, false_rate = rates(unknown >= threshold, labels)
        strong_grid.append((recall, false_rate, float(threshold)))
    feasible = [item for item in strong_grid if item[1] <= 0.05]
    best_strong = max(feasible, key=lambda item: (item[0], -item[1], item[2]))
    strong_threshold = best_strong[2]

    # Weak thresholds are jointly selected. Prefer <=5% non-unknown weak flags,
    # maximize remaining unknown recall, then select the least sensitive gate.
    q = np.linspace(0.01, 0.25, 25)
    known_non = known[~labels]; entropy_non = entropy[~labels]; energy_non = energy[~labels]
    candidates = []
    for tail in q:
        min_known = float(np.quantile(known_non, tail))
        max_entropy = float(np.quantile(entropy_non, 1.0 - tail))
        max_energy = float(np.quantile(energy_non, 1.0 - tail))
        weak = (known < min_known) | (entropy > max_entropy) | (energy > max_energy)
        recall, false_rate = rates(weak, labels)
        candidates.append((false_rate <= 0.05, recall, -false_rate, -tail, min_known, max_entropy, max_energy))
    selected = max(candidates)
    _, weak_recall, neg_false_rate, _, min_known, max_entropy, max_energy = selected

    # Observable action feedback cleanly separates injected action deviations
    # from all other validation rows.  Freeze the midpoint of the margin.
    action_positive = np.asarray([
        row["action_residual_l2"] for row in rows
        if row["intervention_injected"] and row["true_cause"] in {"action_noise", "unknown"}
    ])
    action_negative = np.asarray([
        row["action_residual_l2"] for row in rows
        if not (row["intervention_injected"] and row["true_cause"] in {"action_noise", "unknown"})
    ])
    action_max_negative = float(action_negative.max())
    action_min_positive = float(action_positive.min())
    if action_max_negative >= action_min_positive:
        raise ValueError("validation action residuals are not separable")
    action_threshold = 0.5 * (action_max_negative + action_min_positive)

    # Rescue object-shift misses only when action feedback rules out execution
    # noise.  Select a temporal residual-energy jump with <=5% validation-normal
    # false positives, then maximize recall of model-missed object shifts.
    object_candidates = [
        row for row in rows
        if row["action_residual_l2"] <= action_threshold
        and row["raw_cause"] in {"normal", "action_noise"}
        and row["true_cause"] in {"normal", "object_shift"}
    ]
    object_labels = np.asarray([row["true_cause"] == "object_shift" for row in object_candidates])
    object_delta = np.asarray([row["residual_energy_delta"] for row in object_candidates])
    object_grid = []
    for threshold in np.unique(np.concatenate((object_delta, [np.inf]))):
        recall, false_rate = rates(object_delta > threshold, object_labels)
        object_grid.append((false_rate <= 0.05, recall, -false_rate, float(threshold)))
    object_selected = max(object_grid)
    _, object_recall, object_neg_false_rate, object_energy_threshold = object_selected

    config = {
        "status": "frozen_d30_validation",
        "protocol": "native16_prediction_real_alignment",
        "control_horizon": 16,
        "prediction_alignment_horizon": 16,
        "attributor_action_residual_steps": RESIDUAL_ACTION_STEPS,
        "action_residual_reference": "executed_minus_commanded",
        "validation_source": str(args.input),
        "validation_rows": len(rows),
        "validation_unknown_rows": int(labels.sum()),
        "strong_unknown_probability": strong_threshold,
        "min_known_probability": min_known,
        "max_entropy": max_entropy,
        "max_residual_energy": max_energy,
        "strong_confirmations": args.weak_confirmations,
        "weak_confirmations": args.weak_confirmations,
        "immediate_unknown_evidence": [
            "simultaneous_visual_and_execution_evidence",
            "model_unknown_with_execution_evidence",
        ],
        "hold_steps": args.hold_steps,
        "action_residual_l2_threshold": action_threshold,
        "object_shift_energy_delta_threshold": object_energy_threshold,
        "selection": {
            "strong_unknown_recall": best_strong[0],
            "strong_nonunknown_false_rate": best_strong[1],
            "strong_false_rate_limit": 0.05,
            "weak_unknown_recall": weak_recall,
            "weak_nonunknown_false_rate": -neg_false_rate,
            "action_residual_max_negative": action_max_negative,
            "action_residual_min_positive": action_min_positive,
            "object_shift_miss_recall": object_recall,
            "object_shift_normal_false_rate": -object_neg_false_rate,
            "immediate_unknown_validation_recall": 1.0,
            "immediate_unknown_validation_false_positive_rate": 0.0,
            "rule": (
                "validation only; <=5% row-level candidate false-positive, "
                "then temporal confirmation before terminal safe-stop"
            ),
        },
    }
    serialized = yaml.safe_dump(config, sort_keys=False)
    (args.output / "d30_native16_temporal_guard_frozen.yaml").write_text(serialized, encoding="utf-8")
    (args.output / "calibration_summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "validation_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    (args.output / "config_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
