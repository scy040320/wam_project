"""Deployable target-conditioned pooling for prediction/reality residuals.

The relevance map is produced from the task-language target phrase and RGB
images by a frozen open-vocabulary localizer. Simulator segmentation, object
poses, intervention labels, and success labels are deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


def canonical_target_prompt(entity: str) -> str:
    text = re.sub(r"_\d+(?=_|$)", "", str(entity).strip().lower())
    text = text.replace("_", " ")
    replacements = {
        "flat stove button": "stove button",
        "bottom level": "bottom drawer",
        "top level": "top drawer",
        "ketchup": "ketchup bottle",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def _resize_map(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("relevance maps must be finite 2-D arrays")
    if array.shape == shape:
        return array
    from PIL import Image

    return np.asarray(
        Image.fromarray(array, mode="F").resize((shape[1], shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class TargetResidualFeatures:
    values: np.ndarray
    names: tuple[str, ...]
    prompt: str


def structural_residual_grid(predicted_rgb: object, actual_rgb: object, *, size: int = 64) -> np.ndarray:
    """Measure internal edge changes using only predicted and observed RGB.

    Pixel residuals are diluted for articulated objects whose outer silhouette
    stays fixed (for example, a drawer opening inside a cabinet). Gradient
    magnitude exposes the displaced internal boundary without simulator state.
    """
    from PIL import Image

    def gray(image: object) -> np.ndarray:
        array = np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]

    def magnitude(array: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(array)
        return np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)

    return np.clip(
        np.abs(magnitude(gray(predicted_rgb)) - magnitude(gray(actual_rgb))),
        0.0,
        1.0,
    ).astype(np.float32)


def target_semantic_features(entity: str) -> TargetResidualFeatures:
    """Task-independent target semantics derived from deployment language."""
    prompt = canonical_target_prompt(entity)
    tokens = set(prompt.split())
    articulated = float(bool(tokens & {"drawer", "button", "door", "joint", "microwave"}))
    receptacle = float(bool(tokens & {"drawer", "cabinet", "bowl", "basket", "caddy", "tray"}))
    rigid_object = float(not articulated)
    return TargetResidualFeatures(
        np.asarray([articulated, receptacle, rigid_object], dtype=np.float32),
        (
            "target.semantic.articulated",
            "target.semantic.receptacle",
            "target.semantic.rigid_object",
        ),
        prompt,
    )


def pool_target_residual(
    residual_grid: np.ndarray,
    predicted_relevance: np.ndarray,
    actual_relevance: np.ndarray,
    *,
    prefix: str,
    prompt: str,
) -> TargetResidualFeatures:
    residual = np.asarray(residual_grid, dtype=np.float32)
    if residual.ndim != 2 or not np.isfinite(residual).all():
        raise ValueError("residual_grid must be a finite 2-D array")
    pred = np.clip(_resize_map(predicted_relevance, residual.shape), 0.0, 1.0)
    actual = np.clip(_resize_map(actual_relevance, residual.shape), 0.0, 1.0)
    union = np.maximum(pred, actual)
    intersection = np.minimum(pred, actual)
    mass = float(union.sum())
    weights = union / max(mass, 1e-8)
    flat_residual = residual.reshape(-1)
    flat_weights = weights.reshape(-1)
    target_mean = float(np.sum(flat_residual * flat_weights))
    order = np.argsort(flat_weights)[::-1]
    top_count = max(1, int(round(0.20 * len(order))))
    target_top20 = float(flat_residual[order[:top_count]].mean())
    background = 1.0 - union
    background_mean = float(np.sum(residual * background) / max(float(background.sum()), 1e-8))
    yy, xx = np.mgrid[0:residual.shape[0], 0:residual.shape[1]]
    centroid_x = float(np.sum(weights * xx) / max(1, residual.shape[1] - 1))
    centroid_y = float(np.sum(weights * yy) / max(1, residual.shape[0] - 1))
    pred_mass = float(pred.sum())
    actual_mass = float(actual.sum())
    pred_weights = pred / max(pred_mass, 1e-8)
    actual_weights = actual / max(actual_mass, 1e-8)
    pred_centroid_x = float(np.sum(pred_weights * xx) / max(1, residual.shape[1] - 1))
    pred_centroid_y = float(np.sum(pred_weights * yy) / max(1, residual.shape[0] - 1))
    actual_centroid_x = float(np.sum(actual_weights * xx) / max(1, residual.shape[1] - 1))
    actual_centroid_y = float(np.sum(actual_weights * yy) / max(1, residual.shape[0] - 1))
    centroid_delta_x = actual_centroid_x - pred_centroid_x
    centroid_delta_y = actual_centroid_y - pred_centroid_y
    centroid_delta_l2 = float(np.hypot(centroid_delta_x, centroid_delta_y))
    pred_residual_mean = float(np.sum(residual * pred_weights))
    actual_residual_mean = float(np.sum(residual * actual_weights))
    soft_iou = float(intersection.sum() / max(float(union.sum()), 1e-8))
    values = np.asarray([
        target_mean, target_top20, background_mean, target_mean - background_mean,
        float(pred.mean()), float(actual.mean()), soft_iou, centroid_x, centroid_y,
        float(np.max(union)), pred_residual_mean, actual_residual_mean,
        pred_centroid_x, pred_centroid_y, actual_centroid_x, actual_centroid_y,
        centroid_delta_x, centroid_delta_y, centroid_delta_l2,
        abs(float(actual.mean()) - float(pred.mean())),
    ], dtype=np.float32)
    names = tuple(f"{prefix}.{name}" for name in (
        "weighted_residual_mean", "top20_relevance_residual_mean",
        "background_residual_mean", "target_minus_background",
        "predicted_relevance_mean", "actual_relevance_mean",
        "predicted_actual_soft_iou", "relevance_centroid_x",
        "relevance_centroid_y", "relevance_max",
        "predicted_weighted_residual_mean", "actual_weighted_residual_mean",
        "predicted_centroid_x", "predicted_centroid_y",
        "actual_centroid_x", "actual_centroid_y",
        "centroid_delta_x", "centroid_delta_y", "centroid_delta_l2",
        "relevance_mass_abs_delta",
    ))
    return TargetResidualFeatures(values, names, canonical_target_prompt(prompt))


class CLIPSegTargetLocalizer:
    """Frozen text-conditioned RGB localizer used only through deployable inputs."""

    model_id = "CIDAS/clipseg-rd64-refined"

    def __init__(
        self,
        *,
        device: str,
        model_path: str | None = None,
        local_files_only: bool = True,
    ):
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self._torch = torch
        self.device = torch.device(device)
        source = model_path or self.model_id
        self.processor = CLIPSegProcessor.from_pretrained(
            source, local_files_only=local_files_only, use_fast=False
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            source, local_files_only=local_files_only
        ).eval().to(self.device)

    def relevance(self, images: Sequence[object], target: str) -> np.ndarray:
        if not images:
            raise ValueError("at least one image is required")
        prompt = canonical_target_prompt(target)
        inputs = self.processor(
            text=[prompt] * len(images), images=list(images), padding=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits
        return self._torch.sigmoid(logits).float().cpu().numpy().astype(np.float32)
