"""Resumable, online-only DINOv2-S feature extraction for CF-WAM records.

Each output file contains only deployment-time fields plus frozen image
embeddings. Offline labels are intentionally not opened in this script.  The
subsequent tensor builder may join labels by run id for supervised training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


VIEWS = ("primary", "wrist")


def record_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("online_record.json"))


def image_paths(record: Path, online: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    for view in VIEWS:
        paths.extend([
            record / str(online[f"{view}_current"]),
            record / str(online[f"{view}_prediction"]),
            record / str(online[f"{view}_actual"]),
        ])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--records-per-batch", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.records_per_batch < 1:
        raise ValueError("records-per-batch must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    records = [record for record in record_dirs(args.record_root) if not (args.output / f"{record.name}.pt").is_file()]
    if args.limit is not None:
        records = records[:args.limit]
    print(json.dumps({"pending_records": len(records), "device": args.device}, ensure_ascii=False), flush=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(args.device)

    completed = 0
    for start in range(0, len(records), args.records_per_batch):
        batch = records[start:start + args.records_per_batch]
        online_rows = [json.loads((record / "online_record.json").read_text(encoding="utf-8")) for record in batch]
        images = [Image.open(path).convert("RGB") for record, online in zip(batch, online_rows) for path in image_paths(record, online)]
        inputs = processor(images=images, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            cls_tokens = model(**inputs).last_hidden_state[:, 0].float().cpu()
        for index, (record, online) in enumerate(zip(batch, online_rows)):
            tokens = cls_tokens[index * 6:(index + 1) * 6]
            payload = {
                "run_id": online["run_id"],
                "task_id": online["task_id"],
                "episode_id": online["episode_id"],
                "split": online["split"],
                "phase": online["phase"],
                "initial_state_id": f"{online['task_id']}:seed{int(online['episode_id']):02d}",
                "value_prediction": float(online["value_prediction"]),
                "primary_current": tokens[0],
                "primary_prediction": tokens[1],
                "primary_actual": tokens[2],
                "wrist_current": tokens[3],
                "wrist_prediction": tokens[4],
                "wrist_actual": tokens[5],
                "primary_prediction_residual": tokens[2] - tokens[1],
                "wrist_prediction_residual": tokens[5] - tokens[4],
            }
            torch.save(payload, args.output / f"{record.name}.pt")
            completed += 1
        print(json.dumps({"completed_this_run": completed, "total_pending": len(records), "last": batch[-1].name}, ensure_ascii=False), flush=True)

    (args.output / "feature_schema.json").write_text(json.dumps({
        "encoder": "facebook/dinov2-small", "encoder_frozen": True,
        "views": list(VIEWS), "features_per_record": 6,
        "label_access": "none", "source": "online_record.json only",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
