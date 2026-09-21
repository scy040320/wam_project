"""Build fixed-shape, grouped CF-WAM training tensors from audited records.

The builder joins offline labels only at this supervised-training boundary.
Deployment features are constructed exclusively from online records, frozen
DINO embeddings, proprioception, action-execution feedback and predicted
value context.  The current v1 collector does not contain a re-queried value
at t+4; this field is explicitly recorded as a limitation rather than being
silently fabricated as a value-change residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml


CAUSES = ("normal", "visual_occlusion", "object_shift", "action_noise", "unknown")
SLOTS = (
    "phase_approach", "phase_grasp", "phase_transport", "camera_primary", "camera_wrist",
    "effector_gripper", "object_primary", "object_secondary", "target", "action_segment_4",
)
KIND_INDEX = {"task_phase": 0, "observation": 1, "effector_gripper": 2, "object": 3, "target_region": 4, "action_segment": 5}
EDGE_INDEX = torch.tensor([[3, 3, 4, 9, 5, 5, 6, 7, 0, 1], [6, 7, 5, 5, 6, 7, 8, 8, 1, 2]], dtype=torch.long)
EDGE_TYPE = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3, 4, 4], dtype=torch.long)


def canonical_map(config: dict[str, object]) -> dict[str, str]:
    nodes = config["nodes"]
    objects = [item["id"] for item in nodes if item["kind"] == "object"]
    mapping = {item["id"]: item["id"] for item in nodes if item["kind"] != "object"}
    if objects:
        mapping[objects[0]] = "object_primary"
    if len(objects) > 1:
        mapping[objects[1]] = "object_secondary"
    target = next(item["id"] for item in nodes if item["kind"] == "target_region")
    mapping[target] = "target"
    return mapping


def graph_tensors(config: dict[str, object], phase: str) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = canonical_map(config)
    present = set(mapping.values())
    valid = torch.tensor([slot in present for slot in SLOTS], dtype=torch.float32)
    edge_mask = torch.tensor([
        all(slot in present for slot in (SLOTS[source], SLOTS[target]))
        for source, target in EDGE_INDEX.t().tolist()
    ], dtype=torch.float32)
    feature = torch.zeros((len(SLOTS), 8), dtype=torch.float32)
    for item in config["nodes"]:
        slot = mapping[item["id"]]
        index = SLOTS.index(slot)
        feature[index, KIND_INDEX[item["kind"]]] = 1.0
        feature[index, 6] = 1.0
        feature[index, 7] = float(slot == f"phase_{phase}")
    return feature, valid, edge_mask, mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configs = {task: yaml.safe_load((args.config_root / f"libero_task_{task}.yaml").read_text(encoding="utf-8")) for task in range(4)}
    splits: dict[str, list[dict[str, torch.Tensor | str]]] = {"train": [], "validation": [], "test": []}
    missing_features = []
    for record in sorted(path for path in args.record_root.iterdir() if path.is_dir() and path.name.startswith("task")):
        feature_path = args.feature_root / f"{record.name}.pt"
        if not feature_path.is_file():
            missing_features.append(record.name)
            continue
        online = json.loads((record / "online_record.json").read_text(encoding="utf-8"))
        offline = json.loads((record / "offline_label.json").read_text(encoding="utf-8"))
        task = int(str(online["task_id"]).rsplit("_", 1)[-1])
        feature, node_valid, edge_mask, mapping = graph_tensors(configs[task], str(online["phase"]))
        dino = torch.load(feature_path, map_location="cpu", weights_only=False)
        proprio_delta = torch.from_numpy(np.load(record / str(online["proprio_actual"]))).float().flatten() - torch.from_numpy(np.load(record / str(online["proprio_current"]))).float().flatten()
        action_delta = torch.from_numpy(np.load(record / str(online["action_execution_delta"]))).float().flatten()
        residual = torch.cat((dino["primary_prediction_residual"].flatten(), dino["wrist_prediction_residual"].flatten(), proprio_delta, action_delta, torch.tensor([float(online["value_prediction"])], dtype=torch.float32)))
        dynamic = {"camera_primary": float(torch.linalg.vector_norm(dino["primary_prediction_residual"])), "camera_wrist": float(torch.linalg.vector_norm(dino["wrist_prediction_residual"])), "effector_gripper": float(torch.linalg.vector_norm(proprio_delta)), "action_segment_4": float(torch.linalg.vector_norm(action_delta))}
        for slot, value in dynamic.items():
            feature[SLOTS.index(slot), 7] = value
        mask = torch.zeros(len(SLOTS), dtype=torch.float32)
        for node_id in offline.get("affected_nodes", []):
            slot = mapping.get(node_id)
            if slot is not None:
                mask[SLOTS.index(slot)] = 1.0
        splits[str(online["split"])].append({
            "run_id": record.name, "initial_state_id": dino["initial_state_id"], "residual": residual,
            "node_features": feature, "edge_index": EDGE_INDEX, "edge_type": EDGE_TYPE,
            "edge_mask": edge_mask, "node_valid": node_valid, "cause": torch.tensor(CAUSES.index(offline["cause"])), "mask": mask,
        })
    if missing_features:
        raise RuntimeError(f"Missing frozen DINO files for {len(missing_features)} records; first={missing_features[:5]}")
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        torch.save(rows, args.output / f"{split}.pt")
    (args.output / "tensor_manifest.json").write_text(json.dumps({
        "records": {split: len(rows) for split, rows in splits.items()}, "cause_order": CAUSES,
        "canonical_slots": SLOTS, "residual": "primary_dino_residual + wrist_dino_residual + proprio_delta + action_delta + predicted_value_context",
        "value_change_available": False, "value_change_note": "t+4 WAM value was not collected; predicted value is contextual only",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": {split: len(rows) for split, rows in splits.items()}, "residual_dim": len(splits["train"][0]["residual"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
