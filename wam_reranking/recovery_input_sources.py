"""Read-only content checks in addition to upstream X provenance audits.

A valid hash is not proof that a text file is RGB, that inline commands match
their source, or that a file was available before execution. This module
checks the first two; the whole-run producer audit must establish the third.
It never reads actual future/teacher files to build deployment features.
"""
from __future__ import annotations

import numpy as np


def _array(ref, reader, shape, inline, name):
    path = reader.path(ref)
    if path.suffix.lower() != ".npy":
        raise ValueError(name + " requires a standalone immutable NPY source")
    value = np.load(path, allow_pickle=False)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(name + " source shape/type/value mismatch")
    declared = np.asarray(inline, dtype=np.float64)
    if declared.shape != shape or not np.array_equal(value, declared):
        raise ValueError(name + " inline content differs from exact source")


def _rgb(ref, reader, name):
    path = reader.path(ref)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(name + " source must be uint8 HxWx3 RGB")
        if min(value.shape[:2]) < 2 or max(value.shape[:2]) > 4096:
            raise ValueError(name + " invalid source image dimensions")
        return
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError(name + " requires an actual decodable RGB image")
    from PIL import Image
    with Image.open(path) as img:
        if img.mode != "RGB" or img.format not in ("PNG", "JPEG"):
            raise ValueError(name + " source image format/mode mismatch")
        if min(img.size) < 2 or max(img.size) > 4096:
            raise ValueError(name + " invalid source image dimensions")
        img.verify()


def _json_value(ref, reader, inline, name):
    path = reader.path(ref)
    if path.suffix.lower() != ".json":
        raise ValueError(name + " requires an immutable canonical JSON source")
    from .joint_recovery_contract import digest
    # The producer emits canonical, narrowed before/predicted records, bound
    # to their original parent query/observation in its independent audit.
    # Broad teacher records are not accepted as canonical input records.
    value = reader.json(ref)
    if digest(value) != digest(inline):
        raise ValueError(name + " inline content differs from exact source")


def validate_input_source_contents(x, reader):
    """Require actual bytes and exact inline-source agreement, not flags."""
    before, predicted = x["actual_before"], x["candidate_predicted"]
    for name in ("primary_rgb", "wrist_rgb"):
        _rgb(before[name], reader, "actual-before " + name)
    for name in ("primary_endpoint_rgb", "wrist_endpoint_rgb"):
        _rgb(predicted[name], reader, "candidate-predicted " + name)
    _array(before["proprio_source"], reader, (9,), before["proprio"], "actual-before proprio")
    _array(x["planned_actions_source"], reader, (16, 7), x["planned_actions"], "planned actions")
    _json_value(before["quality_source"], reader, before["quality"], "deployable before quality")
    _json_value(predicted["visual_source"], reader, predicted["visual"], "predicted candidate visual")
    context = x["task_context"]
    binding = {k: v for k, v in context.items() if k != "entity_binding_source"}
    _json_value(context["entity_binding_source"], reader, binding, "frozen task binding")
    return True
