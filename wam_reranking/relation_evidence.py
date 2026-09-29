"""Deployable subject–anchor evidence from predicted and observed RGB.

The task binding is fixed from the instruction and current operation. No
simulator object pose, segmentation, intervention label, or success label is
used. Contact and support are image-space proxies, not physical-state claims.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelationBinding:
    subject: str
    anchor: str
    relation: str


# Frozen before D18-v18 training. The prompts correspond to the public task
# instruction and stage, never to a simulator object identifier at inference.
RELATION_BINDINGS: dict[tuple[int, int], RelationBinding] = {
    # Public LIBERO-10 pilot tasks.  These bindings are derived from the task
    # instruction rather than simulator object identifiers at decision time.
    (0, 0): RelationBinding("alphabet soup", "basket", "toward"),
    (1, 0): RelationBinding("cream cheese box", "basket", "toward"),
    (8, 0): RelationBinding("moka pot", "stove", "on"),
    (8, 1): RelationBinding("moka pot", "stove", "on"),
    (9, 0): RelationBinding("yellow mug", "microwave", "inside"),
    (9, 1): RelationBinding("microwave door", "microwave frame", "closed"),
    (20, 0): RelationBinding("stove button", "stove", "articulated"),
    (21, 0): RelationBinding("stove button", "stove", "articulated"),
    (21, 1): RelationBinding("frying pan", "stove", "on"),
    (23, 0): RelationBinding("bottom drawer", "cabinet frame", "closed"),
    (23, 1): RelationBinding("top drawer", "cabinet frame", "open"),
    (32, 0): RelationBinding("ketchup bottle", "top drawer", "toward"),
    (32, 1): RelationBinding("ketchup bottle", "top drawer", "inside"),
    (34, 0): RelationBinding("yellow white mug", "white mug", "toward"),
    (34, 1): RelationBinding("yellow white mug", "white mug", "in_front_of"),
    (44, 0): RelationBinding("stove button", "stove", "articulated"),
    (58, 0): RelationBinding("ketchup bottle", "tray", "toward"),
    (58, 1): RelationBinding("ketchup bottle", "tray", "inside"),
    (62, 0): RelationBinding("salad dressing bottle", "tray", "toward"),
    (62, 1): RelationBinding("salad dressing bottle", "tray", "inside"),
    (63, 0): RelationBinding("left black bowl", "right black bowl", "on"),
    (63, 1): RelationBinding("stack of black bowls", "tray", "inside"),
    (64, 0): RelationBinding("right black bowl", "left black bowl", "on"),
    (64, 1): RelationBinding("stack of black bowls", "tray", "inside"),
    (77, 0): RelationBinding("black book", "caddy", "toward"),
    (77, 1): RelationBinding("black book", "back caddy compartment", "inside"),
    (14, 0): RelationBinding("middle black bowl", "plate", "toward"),
    (14, 1): RelationBinding("middle black bowl", "plate", "on"),
    # Consumed regression tasks. These are not independent test tasks.
    (16, 0): RelationBinding("front black bowl", "middle black bowl", "toward"),
    (16, 1): RelationBinding("front black bowl", "middle black bowl", "on"),
    (17, 0): RelationBinding("middle black bowl", "back black bowl", "toward"),
    (17, 1): RelationBinding("middle black bowl", "back black bowl", "on"),
    (39, 0): RelationBinding("stove button", "stove", "articulated"),
    (39, 1): RelationBinding("stove button", "stove", "articulated"),
    (45, 0): RelationBinding("stove button", "stove", "articulated"),
    (45, 1): RelationBinding("frying pan", "stove", "on"),
    (73, 0): RelationBinding("black book", "caddy", "toward"),
    (73, 1): RelationBinding("black book", "caddy", "inside"),
    # Sealed confirmation bindings are determined from its frozen public task
    # protocol, without inspecting the confirmation observations or outcomes.
    (13, 0): RelationBinding("front black bowl", "plate", "toward"),
    (13, 1): RelationBinding("front black bowl", "plate", "on"),
    (37, 0): RelationBinding("white bowl", "plate", "toward"),
    (37, 1): RelationBinding("white bowl", "plate", "right_of"),
    (65, 0): RelationBinding("red coffee mug", "plate", "toward"),
    (65, 1): RelationBinding("red coffee mug", "left plate", "on"),
    # D19-D21 V5 independent gate. These bindings were frozen from the
    # public LIBERO-90 instructions before any validation observation or
    # outcome was generated.
    (26, 0): RelationBinding("wine bottle", "bottom drawer", "inside"),
    (55, 0): RelationBinding("alphabet soup", "tray", "inside"),
    (24, 0): RelationBinding("black bowl", "bottom drawer", "inside"),
    (70, 0): RelationBinding("chocolate pudding", "plate", "right_of"),
    (78, 0): RelationBinding("black book", "front caddy compartment", "inside"),
    (81, 0): RelationBinding("black book", "front caddy compartment", "inside"),
    # Preregistered candidate-coverage pool (2026-09-29).  Only public task
    # language and BDDL semantics are used; no rollout or outcome was viewed
    # when these bindings were frozen.
    (11, 0): RelationBinding("top drawer", "cabinet frame", "articulated"),
    (18, 0): RelationBinding("frying pan", "stove", "on"),
    (27, 0): RelationBinding("wine bottle", "wine rack", "on"),
    (35, 0): RelationBinding("microwave door", "microwave frame", "articulated"),
    (46, 0): RelationBinding("alphabet soup", "basket", "inside"),
    (57, 0): RelationBinding("cream cheese", "tray", "inside"),
    (68, 0): RelationBinding("yellow white mug", "right plate", "on"),
    (86, 0): RelationBinding("middle book", "cabinet shelf", "inside"),
}


def _distribution(relevance: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(relevance, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("relevance map must be a finite 2-D array")
    arr = np.clip(arr, 0.0, 1.0)
    mass = float(arr.mean())
    peak = float(np.quantile(arr, 0.99))
    total = float(arr.sum())
    dist = arr / total if total > 1e-8 else np.full_like(arr, 1.0 / arr.size)
    return dist, mass, peak


def _geometry(subject: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    from PIL import Image

    def resize(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        return np.asarray(Image.fromarray(arr, mode="F").resize((64, 64), Image.Resampling.BILINEAR))

    s, smass, speak = _distribution(resize(subject))
    a, amass, apeak = _distribution(resize(anchor))
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    sx, sy = float((s * xx).sum() / 63), float((s * yy).sum() / 63)
    ax, ay = float((a * xx).sum() / 63), float((a * yy).sum() / 63)
    dx, dy = sx - ax, sy - ay
    dist = float(np.hypot(dx, dy))
    # Affinity is a normalized soft overlap, useful as an image-space proxy
    # for contact/containment. It does not assert a physical contact event.
    affinity = float(np.minimum(s, a).sum() / max(np.maximum(s, a).sum(), 1e-8))
    sspread = float(np.sqrt(((xx / 63 - sx) ** 2 * s + (yy / 63 - sy) ** 2 * s).sum()))
    aspread = float(np.sqrt(((xx / 63 - ax) ** 2 * a + (yy / 63 - ay) ** 2 * a).sum()))
    return np.asarray([dx, dy, dist, affinity, sspread, aspread, smass, amass, speak, apeak], np.float32)


def relation_change_features(
    predicted_subject: np.ndarray,
    actual_subject: np.ndarray,
    predicted_anchor: np.ndarray,
    actual_anchor: np.ndarray,
) -> np.ndarray:
    """Return 16 image-space relation features for one camera.

    All relative quantities compare the same query's predicted and observed
    terminal frame. Absolute subject/anchor coordinates are intentionally
    omitted so the feature cannot identify an initial scene layout directly.
    """
    p = _geometry(predicted_subject, predicted_anchor)
    o = _geometry(actual_subject, actual_anchor)
    delta = o[:6] - p[:6]
    # The final four terms qualify evidence strength; a weak/missing anchor
    # should lower confidence, rather than provide false relation evidence.
    quality = np.asarray([
        min(p[6], o[6]), min(p[7], o[7]),
        min(p[8], o[8]), min(p[9], o[9]),
    ], np.float32)
    out = np.concatenate([delta, np.abs(delta[:4]), quality, np.asarray([p[3], o[3]], np.float32)])
    if out.shape != (16,) or not np.isfinite(out).all():
        raise RuntimeError("relation feature contract violated")
    return out


def relation_temporal_features(
    start_subject: np.ndarray,
    actual_subject: np.ndarray,
    start_anchor: np.ndarray,
    actual_anchor: np.ndarray,
) -> np.ndarray:
    """Eight deployable subject–anchor changes from query to real endpoint.

    This complements, rather than replaces, predicted-vs-real relation
    evidence. It can expose a small object displacement when the world model's
    terminal image has a much larger rendering error. The caller must use the
    same camera and the actual aligned block endpoints.
    """
    start = _geometry(start_subject, start_anchor)
    actual = _geometry(actual_subject, actual_anchor)
    delta = actual[:4] - start[:4]
    quality = np.asarray([
        min(start[8], actual[8]), min(start[9], actual[9]),
    ], np.float32)
    out = np.concatenate([delta, np.abs(delta[:2]), quality])
    if out.shape != (8,) or not np.isfinite(out).all():
        raise RuntimeError("temporal relation feature contract violated")
    return out


def instance_correspondence_maps(
    semantic_maps: np.ndarray,
    patch_tokens: np.ndarray,
    *,
    reference_index: int,
    prompt: str,
) -> np.ndarray:
    """Bind one language-selected instance across frames using a visual prototype.

    ``semantic_maps`` are open-vocabulary relevance maps and ``patch_tokens``
    are frozen visual tokens for the same ordered frames.  The reference map
    is spatially disambiguated by language when possible and otherwise uses a
    mild image-centre prior.  A prototype pooled from that reference selects
    the most appearance-consistent semantic evidence in every other frame.
    """
    from PIL import Image

    rel = np.asarray(semantic_maps, np.float32)
    tok = np.asarray(patch_tokens, np.float32)
    if rel.ndim != 3 or tok.ndim != 4 or rel.shape[0] != tok.shape[0]:
        raise ValueError("semantic maps and patch tokens must share a frame axis")
    n, height, width, channels = tok.shape
    resized = np.stack([
        np.asarray(Image.fromarray(m, mode="F").resize((width, height), Image.Resampling.BILINEAR), np.float32)
        for m in rel
    ])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / max(width - 1, 1)
    yn = yy / max(height - 1, 1)
    words = set(str(prompt).lower().replace("_", " ").split())
    cx, cy = 0.5, 0.58
    if "left" in words: cx = 0.28
    if "right" in words: cx = 0.72
    if "front" in words: cy = 0.72
    if "back" in words: cy = 0.34
    if "top" in words: cy = 0.32
    spatial = np.exp(-((xn-cx)**2+(yn-cy)**2)/(2*0.28**2)).astype(np.float32)
    ref_weight = np.clip(resized[reference_index], 0, 1) * (0.35 + 0.65*spatial)
    ref_weight /= max(float(ref_weight.sum()), 1e-8)
    normalized = tok / np.maximum(np.linalg.norm(tok, axis=-1, keepdims=True), 1e-8)
    prototype = (normalized[reference_index] * ref_weight[..., None]).sum((0, 1))
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    similarity = np.clip(np.einsum("nhwc,c->nhw", normalized, prototype), -1, 1)
    # Exponentiation sharpens instance identity while semantic relevance keeps
    # the match within the named object class.
    appearance = np.exp((similarity - similarity.max((1, 2), keepdims=True)) / 0.12)
    matched = np.clip(resized, 0, 1) * appearance
    matched /= np.maximum(matched.max((1, 2), keepdims=True), 1e-8)
    if not np.isfinite(matched).all():
        raise RuntimeError("instance correspondence produced non-finite maps")
    return matched.astype(np.float32)


def typed_relation_features(
    predicted_subject: np.ndarray,
    actual_subject: np.ndarray,
    predicted_anchor: np.ndarray,
    actual_anchor: np.ndarray,
    relation: str,
) -> np.ndarray:
    """Relation-specific evidence with a fixed 12-dimensional contract.

    The values are image-space proxies.  ``on`` emphasizes alignment/contact,
    ``inside`` containment, directional relations signed displacement, and
    articulated/open/closed relations internal relative geometry.  They do
    not assert physical contact or exact joint state.
    """
    p = _geometry(predicted_subject, predicted_anchor)
    o = _geometry(actual_subject, actual_anchor)
    relation = str(relation).lower()
    groups = ("directional", "on", "inside", "articulated")
    if relation in {"toward", "in_front_of", "right_of"}: group = 0
    elif relation == "on": group = 1
    elif relation == "inside": group = 2
    else: group = 3
    onehot = np.eye(len(groups), dtype=np.float32)[group]
    if relation == "right_of":
        before, after = p[0], o[0]
    elif relation == "in_front_of":
        before, after = p[1], o[1]
    elif relation == "on":
        before, after = abs(p[0]) + abs(p[1]) + (1-p[3]), abs(o[0]) + abs(o[1]) + (1-o[3])
    elif relation == "inside":
        before, after = p[2] + (1-p[3]), o[2] + (1-o[3])
    elif relation in {"open", "closed", "articulated"}:
        before, after = p[0] + p[1] + p[4] - p[5], o[0] + o[1] + o[4] - o[5]
    else:  # toward
        before, after = p[2], o[2]
    metrics = np.asarray([
        before, after, after-before, abs(after-before),
        p[3], o[3], min(p[8], o[8]), min(p[9], o[9]),
    ], np.float32)
    out = np.concatenate([onehot, metrics])
    if out.shape != (12,) or not np.isfinite(out).all():
        raise RuntimeError("typed relation feature contract violated")
    return out
