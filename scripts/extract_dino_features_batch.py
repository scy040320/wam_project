"""Extract frozen DINOv2-S features and 10-record visual audit on local CPU/GPU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel


def encode(model, processor, image_path: Path, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        return model(**inputs).last_hidden_state[:, 0].cpu()


def thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records = sorted(path.parent for path in args.record_root.rglob("online_record.json"))[:args.limit]
    if len(records) < args.limit:
        raise ValueError(f"Expected {args.limit} records, found {len(records)}")
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(args.device)
    rows = []
    panel_size, label_h = (128, 96), 28
    montage = Image.new("RGB", (6 * panel_size[0], len(records) * (panel_size[1] + label_h + 22)), "white")
    draw = ImageDraw.Draw(montage)
    labels = ["P current", "P predict", "P real", "W current", "W predict", "W real"]
    for row_index, record_dir in enumerate(records):
        online = json.loads((record_dir / "online_record.json").read_text(encoding="utf-8"))
        tensors = {}
        for view in ("primary", "wrist"):
            tensors[f"{view}_current"] = encode(model, processor, record_dir / online[f"{view}_current"], args.device)
            tensors[f"{view}_prediction"] = encode(model, processor, record_dir / online[f"{view}_prediction"], args.device)
            tensors[f"{view}_actual"] = encode(model, processor, record_dir / online[f"{view}_actual"], args.device)
            tensors[f"{view}_prediction_residual"] = tensors[f"{view}_actual"] - tensors[f"{view}_prediction"]
        torch.save(tensors, args.output / f"{online['run_id']}_dinov2_s_features.pt")
        primary_norm = float(torch.linalg.vector_norm(tensors["primary_prediction_residual"]).item())
        wrist_norm = float(torch.linalg.vector_norm(tensors["wrist_prediction_residual"]).item())
        rows.append({"run_id": online["run_id"], "phase": online["phase"], "episode_id": online["episode_id"], "primary_residual_l2": primary_norm, "wrist_residual_l2": wrist_norm, "device": args.device})
        y = row_index * (panel_size[1] + label_h + 22)
        paths = [online["primary_current"], online["primary_prediction"], online["primary_actual"], online["wrist_current"], online["wrist_prediction"], online["wrist_actual"]]
        for col, (label, relative_path) in enumerate(zip(labels, paths)):
            x = col * panel_size[0]
            montage.paste(thumbnail(record_dir / relative_path, panel_size), (x, y))
            draw.text((x + 3, y + panel_size[1]), label, fill="black")
        draw.text((3, y + panel_size[1] + 13), f"{online['run_id']}  primary L2={primary_norm:.3f}; wrist L2={wrist_norm:.3f}", fill="black")
    with (args.output / "dino_residual_summary_10.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    montage.save(args.output / "dino_visual_audit_10.png")
    (args.output / "run_metadata.json").write_text(json.dumps({"model": "facebook/dinov2-small", "frozen": True, "device": args.device, "record_count": len(rows), "label_access": "none"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
