"""Render an auditable six-panel montage for the highest-confidence errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def tile(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", required=True, type=Path)
    parser.add_argument("--record-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    errors = json.loads(args.errors.read_text(encoding="utf-8"))[:args.limit]
    size, label = (160, 120), 20
    montage = Image.new("RGB", (6 * size[0], max(1, len(errors)) * (size[1] + label + 18)), "white")
    draw = ImageDraw.Draw(montage)
    titles = ("P current", "P predicted", "P real", "W current", "W predicted", "W real")
    keys = ("primary_current", "primary_prediction", "primary_actual", "wrist_current", "wrist_prediction", "wrist_actual")
    for row, error in enumerate(errors):
        directory = args.record_root / error["run_id"]
        online = json.loads((directory / "online_record.json").read_text(encoding="utf-8"))
        top = row * (size[1] + label + 18)
        for col, (title, key) in enumerate(zip(titles, keys)):
            left = col * size[0]
            montage.paste(tile(directory / online[key], size), (left, top))
            draw.text((left + 3, top + size[1]), title, fill="black")
        draw.text((3, top + size[1] + label), f"{error['run_id']} | true={error['true_cause']} pred={error['predicted_cause']} conf={error['confidence']:.3f}", fill="red")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)
    print(json.dumps({"rendered": len(errors), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
