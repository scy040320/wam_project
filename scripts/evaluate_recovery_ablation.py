"""Offline A/B/C/D recovery-decision ablation on a frozen CF-WAM checkpoint.

This is deliberately a *decision-level replay*, not a simulator-success claim.
Every method receives the same t/t+4 record and one initial WAM query.  The
offline cause and node labels are read only to score decisions afterwards.
No simulator truth is fed into any method's decision path.

Methods:
  A binary_residual_global: validation-calibrated residual detector; global refresh.
  B binary_residual_large_subgraph: same detector; invalidate a fixed large graph.
  C attribution_global: learned cause/abstain, but every non-normal result refreshes globally.
  D cfwam_dependency_aware: learned cause/mask plus the reviewed dependency graph.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from build_training_tensors import SLOTS, canonical_map
from cfwam.decision import CAUSE_ORDER, decode_attribution
from cfwam.graph import BeliefGraph, TaskGraphSpec
from cfwam.recovery import RecoveryDecision, RecoveryRouter
from cfwam.runtime import AbstainPolicy
from cfwam.types import AttributionResult, Cause, RecoveryAction
from train_attributor import collate
from cfwam.model import CounterfactualAttributor


CAUSE_FROM_INDEX = tuple(Cause(value) for value in CAUSE_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True, type=Path,
                        help="Validation tensors only: calibrates binary detector.")
    parser.add_argument("--test", required=True, type=Path,
                        help="Held-out development-test tensors; never used for calibration.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frozen-thresholds", required=True, type=Path)
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    return parser.parse_args()


def binary_scores(rows: list[dict]) -> torch.Tensor:
    # Dimension-normalized residual energy. The threshold is selected once on
    # validation, so it is not tuned on the held-out development test.
    return torch.stack([row["residual"].float().pow(2).mean().sqrt() for row in rows])


def binary_threshold(validation: list[dict]) -> float:
    scores = binary_scores(validation)
    targets = torch.tensor([int(row["cause"]) != 0 for row in validation])
    candidates = torch.unique(torch.quantile(scores, torch.linspace(0.05, 0.95, 37)))
    best = (-1.0, float(candidates[0]))
    for threshold in candidates:
        predicted = scores >= threshold
        tp = (predicted & targets).sum().item()
        denom = predicted.sum().item() + targets.sum().item()
        f1 = 0.0 if denom == 0 else 2.0 * tp / denom
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def load_model(rows: list[dict], checkpoint: Path, device: str) -> CounterfactualAttributor:
    first = rows[0]
    model = CounterfactualAttributor(first["residual"].numel(), first["node_features"].shape[-1], 5).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def all_valid_node_ids(spec: TaskGraphSpec) -> list[str]:
    return [node.node_id for node in spec.nodes]


def large_subgraph_node_ids(spec: TaskGraphSpec) -> list[str]:
    # Fixed, reviewed, intentionally coarse region: all visual and physical
    # state/action nodes. It excludes task-phase bookkeeping nodes.
    return [node.node_id for node in spec.nodes if node.node_id not in {
        "phase_approach", "phase_grasp", "phase_transport"
    }]


def route_global(spec: TaskGraphSpec, note: str) -> RecoveryDecision:
    return RecoveryDecision(
        RecoveryAction.SAFE_STOP_GLOBAL_REFRESH,
        frozenset(all_valid_node_ids(spec)), True, False, note,
    )


def route_continue() -> RecoveryDecision:
    return RecoveryDecision(RecoveryAction.CONTINUE, frozenset(), False, True, "continue")


def truth_decision(spec: TaskGraphSpec, true_cause: Cause, target_slots: torch.Tensor) -> RecoveryDecision:
    graph = BeliefGraph(spec)
    if true_cause is Cause.NORMAL:
        return route_continue()
    if true_cause is Cause.VISUAL_OCCLUSION:
        return RecoveryRouter(graph).route(
            AttributionResult(true_cause, {}, frozenset(), 0.0, 0.0, False), 4
        )
    if true_cause is Cause.UNKNOWN:
        return RecoveryRouter(graph).route(
            AttributionResult(true_cause, {}, frozenset(), 0.0, 0.0, True), 4
        )
    slot_to_node = canonical_to_actual(spec)
    nodes = frozenset(slot_to_node[SLOTS[i]] for i, selected in enumerate(target_slots.tolist())
                      if selected and SLOTS[i] in slot_to_node)
    return RecoveryRouter(graph).route(
        AttributionResult(true_cause, {}, nodes, 0.0, 0.0, False), 4
    )


def canonical_to_actual(spec: TaskGraphSpec) -> dict[str, str]:
    payload = {"nodes": [
        {"id": node.node_id, "kind": node.kind.value} for node in spec.nodes
    ]}
    actual_to_slot = canonical_map(payload)
    return {slot: actual for actual, slot in actual_to_slot.items()}


def action_matches(predicted: RecoveryDecision, truth: RecoveryDecision) -> bool:
    return predicted.action is truth.action


def score_row(method: str, decision: RecoveryDecision, truth: RecoveryDecision,
              true_cause: Cause) -> dict[str, object]:
    global_refresh = decision.action is RecoveryAction.SAFE_STOP_GLOBAL_REFRESH
    return {
        "method": method,
        "decision": decision.action.value,
        "truth_action": truth.action.value,
        "action_match": action_matches(decision, truth),
        "invalidated_nodes": len(decision.invalidated_nodes),
        "oracle_minimal_nodes": len(truth.invalidated_nodes),
        "excess_invalidated_nodes": max(0, len(decision.invalidated_nodes) - len(truth.invalidated_nodes)),
        "wam_calls": 1 + int(decision.refresh_wam),  # same initial query for all methods
        "recovery_steps": int(decision.action is not RecoveryAction.CONTINUE),
        "false_stop": bool(true_cause is Cause.NORMAL and global_refresh),
        "unnecessary_global_refresh": bool(true_cause is not Cause.UNKNOWN and global_refresh),
        "safe_failure": bool(true_cause is Cause.UNKNOWN and not global_refresh),
        "missed_mismatch": bool(true_cause is not Cause.NORMAL and decision.action is RecoveryAction.CONTINUE),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = ("action_match", "invalidated_nodes", "oracle_minimal_nodes", "excess_invalidated_nodes",
            "wam_calls", "recovery_steps", "false_stop", "unnecessary_global_refresh",
            "safe_failure", "missed_mismatch")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    thresholds = yaml.safe_load(args.frozen_thresholds.read_text(encoding="utf-8"))
    if thresholds.get("status") != "frozen_for_held_out_test":
        raise ValueError("threshold file must be validation-frozen")
    validation = torch.load(args.validation, map_location="cpu", weights_only=False)
    test = torch.load(args.test, map_location="cpu", weights_only=False)
    if not validation or not test:
        raise ValueError("empty split")
    validation_states = {row["initial_state_id"] for row in validation}
    test_states = {row["initial_state_id"] for row in test}
    if validation_states & test_states:
        raise ValueError("validation/test initial-state leakage")
    binary_cutoff = binary_threshold(validation)
    configs: dict[int, TaskGraphSpec] = {
        task: TaskGraphSpec.from_yaml(args.config_root / f"libero_task_{task}.yaml") for task in range(4)
    }
    model = load_model(test, args.checkpoint, args.device)
    abstain = AbstainPolicy(float(thresholds["min_known_probability"]), float(thresholds["max_entropy"]),
                            float(thresholds["max_residual_energy"]))
    loader = DataLoader(test, batch_size=64, shuffle=False, collate_fn=collate)
    outputs: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for batch in loader:
            device_batch = {key: value.to(args.device) for key, value in batch.items()}
            prediction = model(device_batch["residual"], device_batch["node_features"],
                               device_batch["edge_index"], device_batch["edge_type"],
                               device_batch["edge_mask"], device_batch["node_valid"])
            outputs.append({key: value.detach().cpu() for key, value in prediction.items()})
    merged = {key: torch.cat([item[key] for item in outputs]) for key in outputs[0]}
    scores = binary_scores(test)
    records: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {name: [] for name in (
        "A_binary_residual_global", "B_binary_residual_large_subgraph",
        "C_attribution_global", "D_cfwam_dependency_aware",
    )}
    for index, row in enumerate(test):
        task = int(str(row["run_id"]).split("_")[0].replace("task", ""))
        spec = configs[task]
        true_cause = CAUSE_FROM_INDEX[int(row["cause"])]
        truth = truth_decision(spec, true_cause, row["mask"])
        detected = bool(scores[index] >= binary_cutoff)
        a = route_global(spec, "binary residual threshold") if detected else route_continue()
        if detected:
            b_nodes = frozenset(BeliefGraph(spec).invalidate(large_subgraph_node_ids(spec), "binary_mismatch", 4))
            b = RecoveryDecision(RecoveryAction.LOCAL_STATE_UPDATE, b_nodes, True, False,
                                 "fixed large subgraph without attribution")
        else:
            b = route_continue()
        node_ids = [canonical_to_actual(spec).get(slot, slot) for slot in SLOTS]
        item_outputs = {key: value[index:index + 1] for key, value in merged.items()}
        attribution = decode_attribution(item_outputs, row["residual"].unsqueeze(0), node_ids, abstain)
        c = route_continue() if attribution.cause is Cause.NORMAL else route_global(spec, "attribution without graph")
        d = RecoveryRouter(BeliefGraph(spec)).route(attribution, 4)
        for name, decision in (("A_binary_residual_global", a), ("B_binary_residual_large_subgraph", b),
                               ("C_attribution_global", c), ("D_cfwam_dependency_aware", d)):
            scored = score_row(name, decision, truth, true_cause)
            scored.update({"run_id": str(row["run_id"]), "true_cause": true_cause.value,
                           "predicted_cause": attribution.cause.value if name.startswith(("C_", "D_")) else "binary",
                           "binary_score": float(scores[index]), "binary_threshold": binary_cutoff})
            grouped[name].append(scored)
            records.append(scored)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "recovery_decision_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summary = {
        "scope": "held-out development-test decision-level replay; not simulator episode-success evaluation",
        "record_count": len(test),
        "initial_state_leakage": False,
        "binary_detector": {"threshold": binary_cutoff, "selection": "validation-only mismatch-F1 grid"},
        "learned_model": {"checkpoint": str(args.checkpoint), "abstain_thresholds": {
            key: float(thresholds[key]) for key in ("min_known_probability", "max_entropy", "max_residual_energy")
        }},
        "methods": {name: aggregate(rows) for name, rows in grouped.items()},
        "method_definitions": {
            "A_binary_residual_global": "binary mismatch then global refresh",
            "B_binary_residual_large_subgraph": "same binary mismatch then fixed large state/action subgraph",
            "C_attribution_global": "learned attribution but all non-normal outcomes globally refresh",
            "D_cfwam_dependency_aware": "learned attribution/mask with reviewed dependency closure and abstain exit",
        },
    }
    (args.output / "recovery_ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "README_scope.txt").write_text(
        "This artifact scores recovery decisions after the fact using offline labels. It does not claim task success, "
        "physical rollback, or end-to-end simulator recovery. Those require multi-segment rollouts.\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
