"""Create auditable online belief fields from recorded deployment inputs only.

This exporter deliberately never opens offline_label.json.  Object poses are
not fabricated: without an explicit object detector they remain unasserted,
with camera-quality evidence and zero object-state confidence recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, pstdev

from PIL import Image


def image_quality(path: Path) -> dict[str, float | str]:
    image = Image.open(path).convert("L")
    values = list(image.getdata())
    return {
        "source": path.name,
        "mean_luminance": round(fmean(values), 4),
        "luminance_std": round(pstdev(values), 4),
        "dark_pixel_fraction": round(sum(value < 24 for value in values) / len(values), 6),
    }


def derive(record_dir: Path) -> dict[str, object]:
    online = json.loads((record_dir / "online_record.json").read_text(encoding="utf-8"))
    query_time = online["t_query"]
    primary = image_quality(record_dir / online["primary_current"])
    wrist = image_quality(record_dir / online["wrist_current"])
    confidence = max(0.0, min(1.0, 1.0 - primary["dark_pixel_fraction"]))
    return {
        "run_id": online["run_id"],
        "policy_input_only": True,
        "forbidden_sources": ["offline_label.json", "simulator_state"],
        "t_query": query_time,
        "nodes": {
            "camera_primary": {"valid": True, "observed_at": query_time, "provenance": "online_image", "quality": primary},
            "camera_wrist": {"valid": True, "observed_at": query_time, "provenance": "online_image", "quality": wrist},
            "action_segment_4": {"valid": True, "observed_at": query_time, "provenance": "online_action_feedback", "planned": online["action_plan"], "executed": online["action_executed"], "execution_delta": online["action_execution_delta"]},
            "effector_gripper": {"valid": True, "observed_at": query_time, "provenance": "online_proprioception", "current": online["proprio_current"], "actual_t_plus_4": online["proprio_actual"]},
        },
        "object_and_target_state": {
            "state": "unasserted_without_explicit_detector",
            "confidence": 0.0,
            "supporting_observation": "camera_primary",
            "camera_quality_confidence": confidence,
            "required_next_component": "object-centric detector or segmenter with validation-only calibration",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = sorted(path.parent for path in args.record_root.rglob("online_record.json"))
    if not records:
        raise FileNotFoundError("No online_record.json found below record_root")
    args.output.mkdir(parents=True, exist_ok=True)
    exported = []
    for record_dir in records:
        item = derive(record_dir)
        (args.output / f"{item['run_id']}.json").write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        exported.append(item["run_id"])
    (args.output / "belief_field_schema_v1.json").write_text(json.dumps({
        "record_count": len(exported),
        "online_only": True,
        "object_state_policy": "unasserted without an explicit detector",
        "records": exported,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(exported), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
