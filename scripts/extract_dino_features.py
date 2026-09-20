"""Offline frozen-DINOv2 feature extraction for collected counterfactual records."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def encode(model, processor, image_path: Path, device: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        return model(**inputs).last_hidden_state[:, 0].cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dir")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    root = Path(args.record_dir)
    device = args.device
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(device)
    features = {}
    for view in ("primary", "wrist"):
        current = encode(model, processor, root / f"{view}_current.png", device)
        predicted = encode(model, processor, root / f"{view}_predicted.png", device)
        real = encode(model, processor, root / f"{view}_real_t_plus_4.png", device)
        features[f"{view}_prediction_residual"] = real - predicted
        features[f"{view}_current"] = current
    torch.save(features, root / "dinov2_s_features.pt")
    print(root / "dinov2_s_features.pt")


if __name__ == "__main__":
    main()
