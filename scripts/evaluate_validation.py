"""Validation-only evaluation, abstention calibration and error-case audit.

This script intentionally accepts one split file.  It must be run on the
development validation tensors to freeze abstain thresholds before any held-out
test evaluation.  No raw simulator state is accessed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cfwam.metrics import macro_f1, mask_iou
from cfwam.model import CounterfactualAttributor
from cfwam.types import Cause
from train_attributor import collate


CAUSES = tuple(item.value for item in (Cause.NORMAL, Cause.VISUAL_OCCLUSION, Cause.OBJECT_SHIFT, Cause.ACTION_NOISE, Cause.UNKNOWN))


def ece(probabilities: torch.Tensor, target: torch.Tensor, bins: int = 15) -> float:
    confidence, predicted = probabilities.max(dim=1)
    accuracy = predicted.eq(target).float()
    score = torch.zeros((), dtype=torch.float32)
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if selected.any():
            score += selected.float().mean() * (confidence[selected].mean() - accuracy[selected].mean()).abs()
    return float(score)


def per_class_recall(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        label: float(((predicted == index) & (target == index)).sum() / (target == index).sum().clamp_min(1))
        for index, label in enumerate(CAUSES)
    }


def mask_f1(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    predicted, target = predicted.bool() & valid.bool(), target.bool() & valid.bool()
    tp = (predicted & target).sum().float()
    denominator = predicted.sum().float() + target.sum().float()
    return float(2 * tp / denominator.clamp_min(1.0))


def candidate_energy(hypotheses: torch.Tensor, residual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    selected = hypotheses[torch.arange(len(predicted)), predicted]
    return torch.nn.functional.huber_loss(selected, residual, reduction="none").mean(dim=1)


def calibrate(probabilities: torch.Tensor, hypotheses: torch.Tensor, residual: torch.Tensor, target: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    raw = probabilities.argmax(dim=1)
    known_confidence = probabilities[:, :4].max(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    energy = candidate_energy(hypotheses, residual, raw)
    candidates = []
    for minimum in (0.30, 0.40, 0.50, 0.60, 0.70):
        for max_entropy in (1.00, 1.20, 1.40, 1.60):
            for max_energy in torch.quantile(energy, torch.tensor([0.50, 0.65, 0.80, 0.90, 0.95])).tolist():
                predicted = raw.clone()
                abstain = (raw == 4) | (known_confidence < minimum) | (entropy > max_entropy) | (energy > max_energy)
                predicted[abstain] = 4
                candidates.append((float(macro_f1(predicted, target)), {
                    "min_known_probability": minimum, "max_entropy": max_entropy,
                    "max_residual_energy": float(max_energy),
                }, predicted))
    best = max(candidates, key=lambda item: item[0])
    return best[1], best[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = torch.load(args.validation, map_location="cpu", weights_only=False)
    if not rows:
        raise ValueError("empty validation tensor file")
    first = rows[0]
    model = CounterfactualAttributor(first["residual"].numel(), first["node_features"].shape[-1], 5).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    loader = DataLoader(rows, batch_size=64, shuffle=False, collate_fn=collate)
    payloads = []
    with torch.inference_mode():
        for batch in loader:
            device_batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(device_batch["residual"], device_batch["node_features"], device_batch["edge_index"], device_batch["edge_type"], device_batch["edge_mask"], device_batch["node_valid"])
            payloads.append({key: value.detach().cpu() for key, value in {
                "cause_logits": output["cause_logits"], "mask_logits": output["mask_logits"], "hypotheses": output["hypothesis_residuals"],
                "residual": device_batch["residual"], "target": device_batch["cause"], "mask": device_batch["mask"], "node_valid": device_batch["node_valid"],
            }.items()})
    merged = {key: torch.cat([part[key] for part in payloads]) for key in payloads[0]}
    probabilities = torch.softmax(merged["cause_logits"], dim=1)
    thresholds, predicted = calibrate(probabilities, merged["hypotheses"], merged["residual"], merged["target"])
    mask_prediction = torch.sigmoid(merged["mask_logits"]) >= 0.5
    valid = merged["node_valid"].bool()
    irrelevant = (mask_prediction & ~merged["mask"].bool() & valid).sum(dim=1).float()
    summary = {
        "split": "validation_only", "record_count": len(rows), "threshold_selection": "one validation-only grid search; freeze before test",
        "abstain_thresholds": thresholds,
        "attribution": {"macro_f1": macro_f1(predicted, merged["target"]), "per_class_recall": per_class_recall(predicted, merged["target"]), "ece": ece(probabilities, merged["target"])},
        "localization": {"mask_iou": mask_iou(mask_prediction.float() * valid, merged["mask"] * valid), "mask_f1": mask_f1(mask_prediction, merged["mask"], valid), "mean_irrelevant_nodes_predicted": float(irrelevant.mean())},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "abstain_thresholds_frozen.yaml").write_text("\n".join(f"{key}: {value}" for key, value in thresholds.items()) + "\nselection_source: validation_only\nstatus: frozen_for_held_out_test\n", encoding="utf-8")
    errors = []
    confidences = probabilities.max(dim=1).values
    for index, row in enumerate(rows):
        if int(predicted[index]) != int(merged["target"][index]):
            errors.append({"run_id": row["run_id"], "true_cause": CAUSES[int(merged["target"][index])], "predicted_cause": CAUSES[int(predicted[index])], "confidence": float(confidences[index]), "irrelevant_nodes_predicted": float(irrelevant[index]), "image_record_dir": row["run_id"]})
    errors.sort(key=lambda item: item["confidence"], reverse=True)
    with (args.output / "validation_error_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0]) if errors else ["run_id", "true_cause", "predicted_cause", "confidence", "irrelevant_nodes_predicted", "image_record_dir"])
        writer.writeheader(); writer.writerows(errors)
    (args.output / "validation_error_cases_top20.json").write_text(json.dumps(errors[:20], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
