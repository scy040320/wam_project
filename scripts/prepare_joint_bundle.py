#!/usr/bin/env python3
"""Build deployable D18-attribution + belief + candidate-effect training records."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from wam_reranking import (
    AttributionOutput,
    CoarseCause,
    ConsistencyFactor,
    EvidenceQuality,
    SelectorMode,
    TaskBinding,
    TriValue,
    initial_belief,
    localize_candidate_visual_evidence,
    select_hard_gate_value_tiebreak,
)
from wam_reranking.relation_evidence import RELATION_BINDINGS, RelationBinding
from wam_reranking.target_localization import CLIPSegTargetLocalizer


ANOMALY_FACTORS = (
    "visual_evidence_corrupted",
    "object_or_environment_state_changed",
    "execution_or_contact_deviated",
    "cross_view_conflict",
)

TASK_BINDINGS = {
    0: TaskBinding(0, "close the top drawer of the cabinet", "top_drawer", "cabinet_frame"),
    9: TaskBinding(9, "put the black bowl on the plate", "akita_black_bowl_1", "plate_1"),
    20: TaskBinding(20, "turn on the stove", "flat_stove_1_button", "flat_stove_1"),
    44: TaskBinding(44, "turn on the stove", "flat_stove_1_button", "flat_stove_1"),
    46: TaskBinding(46, "put alphabet soup in the basket", "alphabet_soup_1", "basket_1"),
    57: TaskBinding(57, "put cream cheese in the tray", "cream_cheese_1", "wooden_tray_1"),
}

FROZEN_RELATIONS = {
    (0, 0): RelationBinding("top drawer", "wooden cabinet frame", "articulated"),
    (9, 0): RelationBinding("black bowl", "plate", "on"),
    (20, 0): RelationBinding("stove button", "stove", "articulated"),
    (44, 0): RelationBinding("stove button", "stove", "articulated"),
    (46, 0): RelationBinding("alphabet soup", "basket", "inside"),
    (57, 0): RelationBinding("cream cheese", "tray", "inside"),
}


def entropy(values: Mapping[str, float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in values.values())


def attribution_output(record: Mapping[str, Any], source: str) -> AttributionOutput:
    factors = {name: float(record["factor_prob"][name]) for name in ANOMALY_FACTORS}
    unknown = float(record["unknown_prob"])
    threshold = float(record["unknown_threshold"])
    observation = 1.0 - max(factors["visual_evidence_corrupted"], factors["cross_view_conflict"])
    probabilities = {
        ConsistencyFactor.OBSERVATION_RELIABLE.value: observation,
        ConsistencyFactor.WORLD_STATE_CONSISTENT.value:
            1.0 - factors["object_or_environment_state_changed"],
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value:
            1.0 - factors["execution_or_contact_deviated"],
        ConsistencyFactor.TASK_STAGE_CONSISTENT.value: 1.0,
        ConsistencyFactor.CAUSE_RESOLVED.value: float(unknown < threshold),
    }
    states, confidences = {}, {}
    for name, value in probabilities.items():
        states[name] = TriValue.TRUE if value >= 0.5 else TriValue.FALSE
        confidences[name] = max(value, 1.0 - value)
    cause = CoarseCause(str(record["predicted"]))
    conflict = factors["cross_view_conflict"] >= 0.5
    if conflict or unknown >= threshold:
        cause = CoarseCause.UNKNOWN
    class_probs = {str(k): float(v) for k, v in record["class_probs"].items()}
    return AttributionOutput(
        probabilities,
        class_probs,
        cause,
        float(record["confidence"]),
        entropy(class_probs),
        EvidenceQuality(
            bool(record["primary_available"]) and observation >= 0.5,
            bool(record["wrist_available"]) and observation >= 0.5,
            bool(record["action_record_available"]),
            conflict,
        ),
        source,
        factor_states=states,
        factor_confidences=confidences,
    )


def image(path: Path) -> Image.Image:
    array = np.load(path)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"invalid RGB array: {path}")
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clipseg-cache", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    RELATION_BINDINGS.update(FROZEN_RELATIONS)

    payload = json.loads(args.predictions.read_text())
    predictions = {}
    for item in payload["records"]:
        record = dict(item)
        record["unknown_threshold"] = float(payload["unknown_threshold"])
        key = (
            int(record["task_id"]), int(record["state"]),
            int(record["moment_id"]), str(record["condition"]),
        )
        if key in predictions:
            raise RuntimeError(f"duplicate attribution prediction: {key}")
        predictions[key] = record

    localizer = CLIPSegTargetLocalizer(
        device="cuda", model_path=str(args.clipseg_cache), local_files_only=True,
    )
    paired = []
    for result_path in sorted((args.source_root / "outcomes").glob("*/results.json")):
        for row in json.loads(result_path.read_text()):
            key = (
                int(row["task_id"]), int(row["state"]), int(row["moment_id"]),
                str(row["condition"]),
            )
            if key not in predictions:
                raise RuntimeError(f"missing attribution prediction: {key}")
            root = result_path.parent
            binding = FROZEN_RELATIONS[(key[0], key[2])]
            current_primary = image(root / row["current_primary_path"])
            current_wrist = image(root / row["current_wrist_path"])
            actions, values, visuals = [], [], []
            for candidate in row["candidates"]:
                actions.append(np.load(root / candidate["actions_path"]))
                values.append(float(candidate["value"]))
                visuals.append(localize_candidate_visual_evidence(
                    localizer=localizer,
                    current_primary=current_primary,
                    current_wrist=current_wrist,
                    predicted_primary=image(root / candidate["predicted_primary_path"]),
                    predicted_wrist=image(root / candidate["predicted_wrist_path"]),
                    target_prompt=binding.subject,
                    anchor_prompt=binding.anchor,
                    relation=binding.relation,
                ))
            record = predictions[key]
            attribution = attribution_output(record, f"d21_v7:{key}")
            selection = select_hard_gate_value_tiebreak(
                mode=SelectorMode.LEARNED_HARD_GATE,
                belief=initial_belief(key[0], bindings=TASK_BINDINGS),
                attribution=attribution,
                block_index=3,
                candidate_actions=actions,
                official_values=values,
                visual_evidence=visuals,
                utility_model=None,
            )
            accepted = [decision.candidate_id for decision in selection.decisions if decision.accepted]
            selected_outcome = None
            if selection.selected_candidate_id is not None:
                selected_outcome = row["candidates"][selection.selected_candidate_id]["outcome"]
            elif selection.fallback is not None:
                selected_outcome = row.get("fallback_outcomes", {}).get(selection.fallback)
            paired.append({
                "task_id": key[0], "state": key[1], "moment_id": key[2],
                "condition": key[3], "split": "train" if key[1] <= 5 else "val",
                "candidate_visual_evidence": [asdict(item) for item in visuals],
                "learned_attribution": record,
                "belief_snapshot": selection.belief_snapshot,
                "learned": {
                    "accepted": accepted,
                    "fallback": selection.fallback,
                    "selected_candidate_id": selection.selected_candidate_id,
                    "outcome": selected_outcome,
                },
            })

    expected = int(json.loads((args.source_root / "protocol.json").read_text())["pool_expected"])
    if len(paired) != expected:
        raise RuntimeError(f"paired scenario count {len(paired)} != {expected}")
    keys = [(x["task_id"], x["state"], x["moment_id"], x["condition"]) for x in paired]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate paired scenario key")
    (args.output_root / "paired_scenarios.json").write_text(json.dumps(paired, indent=2) + "\n")
    (args.output_root / "learned_predictions.json").write_text(json.dumps(payload, indent=2) + "\n")
    os.symlink(args.source_root.resolve() / "outcomes", args.output_root / "candidate_outcomes")
    (args.output_root / "bundle_audit.json").write_text(json.dumps({
        "passed": True,
        "scenarios": len(paired),
        "train": sum(x["split"] == "train" for x in paired),
        "val": sum(x["split"] == "val" for x in paired),
        "accepted_candidate_histogram": {
            str(count): sum(len(x["learned"]["accepted"]) == count for x in paired)
            for count in range(5)
        },
        "attribution_and_selection_are_joint": True,
        "outcomes_not_used_to_construct_deployment_features": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
