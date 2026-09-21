"""Held-out CF-WAM test evaluation with validation-frozen abstain thresholds.

There is deliberately no calibration routine in this script.  The threshold
file must have been written by validation-only evaluation before this command
is run, so the test split cannot influence model or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from cfwam.metrics import macro_f1, mask_iou
from cfwam.model import CounterfactualAttributor
from cfwam.types import Cause
from evaluate_validation import CAUSES, candidate_energy, ece, mask_f1, per_class_recall
from train_attributor import collate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frozen-thresholds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    thresholds = yaml.safe_load(args.frozen_thresholds.read_text(encoding="utf-8"))
    if thresholds.get("status") != "frozen_for_held_out_test":
        raise ValueError("threshold file is not explicitly frozen for held-out testing")
    rows = torch.load(args.test, map_location="cpu", weights_only=False)
    first = rows[0]
    model = CounterfactualAttributor(first["residual"].numel(), first["node_features"].shape[-1], 5).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    loader = DataLoader(rows, batch_size=64, shuffle=False, collate_fn=collate)
    parts = []
    with torch.inference_mode():
        for batch in loader:
            device_batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(device_batch["residual"], device_batch["node_features"], device_batch["edge_index"], device_batch["edge_type"], device_batch["edge_mask"], device_batch["node_valid"])
            parts.append({key: value.detach().cpu() for key, value in {
                "cause_logits": output["cause_logits"], "mask_logits": output["mask_logits"], "hypotheses": output["hypothesis_residuals"],
                "residual": device_batch["residual"], "target": device_batch["cause"], "mask": device_batch["mask"], "node_valid": device_batch["node_valid"],
            }.items()})
    merged = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    probabilities = torch.softmax(merged["cause_logits"], dim=1)
    raw = probabilities.argmax(dim=1)
    known_confidence = probabilities[:, :4].max(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    energy = candidate_energy(merged["hypotheses"], merged["residual"], raw)
    predicted = raw.clone()
    abstain = (raw == 4) | (known_confidence < float(thresholds["min_known_probability"])) | (entropy > float(thresholds["max_entropy"])) | (energy > float(thresholds["max_residual_energy"]))
    predicted[abstain] = 4
    mask_prediction = torch.sigmoid(merged["mask_logits"]) >= 0.5
    valid = merged["node_valid"].bool()
    irrelevant = (mask_prediction & ~merged["mask"].bool() & valid).sum(dim=1).float()
    summary = {
        "split": "held_out_development_test", "record_count": len(rows), "threshold_selection": "validation-frozen; no test calibration",
        "abstain_thresholds": {key: thresholds[key] for key in ("min_known_probability", "max_entropy", "max_residual_energy")},
        "attribution": {"macro_f1": macro_f1(predicted, merged["target"]), "per_class_recall": per_class_recall(predicted, merged["target"]), "ece": ece(probabilities, merged["target"])},
        "localization": {"mask_iou": mask_iou(mask_prediction.float() * valid, merged["mask"] * valid), "mask_f1": mask_f1(mask_prediction, merged["mask"], valid), "mean_irrelevant_nodes_predicted": float(irrelevant.mean())},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "heldout_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    errors = []
    confidence = probabilities.max(dim=1).values
    for index, row in enumerate(rows):
        if int(predicted[index]) != int(merged["target"][index]):
            errors.append({"run_id": row["run_id"], "true_cause": CAUSES[int(merged["target"][index])], "predicted_cause": CAUSES[int(predicted[index])], "confidence": float(confidence[index]), "irrelevant_nodes_predicted": float(irrelevant[index]), "image_record_dir": row["run_id"]})
    errors.sort(key=lambda item: item["confidence"], reverse=True)
    with (args.output / "heldout_test_error_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0]) if errors else ["run_id", "true_cause", "predicted_cause", "confidence", "irrelevant_nodes_predicted", "image_record_dir"])
        writer.writeheader(); writer.writerows(errors)
    (args.output / "heldout_test_error_cases_top20.json").write_text(json.dumps(errors[:20], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
