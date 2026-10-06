"""Offline negative CONTACT witnesses, never negative grasp or deployment X.

Private masks select annotation roles only. Both roles must have independently
registered, textured actual-RGB tracks. A visible, above-noise image gap plus
the immutable measured no-contact atom can supervise this finger's contact.
Unknown/hidden roles remain masked. All numerical thresholds inherit V1.
"""
from __future__ import annotations

import math
import numpy as np


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _registration(value):
    return (isinstance(value, dict) and value.get("passed") is True and
        all(_finite(value.get(k)) for k in ("mean_abs", "p95", "fraction_gt5", "psnr")) and
        0 <= value["mean_abs"] <= 2 and 0 <= value["p95"] <= 8 and
        0 <= value["fraction_gt5"] <= .06 and value["psnr"] >= 30)


def _boundary_witness(rgb, mask, floor):
    """Actual contrast across this role's own silhouette, not mask area."""
    h, w = mask.shape
    pixels = set()
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, h-dy), slice(0, w-dx))
        b = (slice(dy, h), slice(dx, w))
        contrast = np.max(np.abs(rgb[a] - rgb[b]), axis=-1)
        ys, xs = np.nonzero((mask[a] != mask[b]) & (contrast > floor))
        for y, x in zip(ys.tolist(), xs.tolist()):
            pixels.add((y, x) if mask[y, x] else (y+dy, x+dx))
    return sorted(pixels)


def _minimum_gap(first, second):
    # Boundary sets are small and this deterministic bounded implementation
    # avoids adding a CV dependency to the CPU annotation validator.
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    return min(float(np.sqrt(((chunk[:, None, :] - b[None, :, :])**2).sum(axis=2)).min())
               for chunk in np.array_split(a, max(1, math.ceil(len(a)/64))))


def _mask_boundary(mask):
    padded = np.pad(mask, 1)
    interior = padded[1:-1, 1:-1].copy()
    for region in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
        interior &= region
    return np.argwhere(mask & ~interior).tolist()


def negative_contact_witness(*, actual_rgb, target_mask, finger_mask,
                             target_registration, finger_registration,
                             target_identity_certified, finger_identity_certified,
                             target_repeat_noise_px, finger_repeat_noise_px,
                             target_repeat_rgb_mean_abs, finger_repeat_rgb_mean_abs,
                             fresh_geometry_registered, physical_contact, side):
    """Return conservative contact-interface input or None for no certificate.

    No-contact truth is a measured LABEL, not inferred from an open command.
    This function never changes a teacher atom, contact threshold or old mask.
    Callers must separately hash/link the actual arrays and candidate identity.
    """
    if side not in ("left", "right"):
        raise ValueError("A negative interface must bind one independent finger")
    if (physical_contact is not False or fresh_geometry_registered is not True or
        target_identity_certified is not True or finger_identity_certified is not True):
        return None
    rgb = np.asarray(actual_rgb)
    tm, fm = np.asarray(target_mask), np.asarray(finger_mask)
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or tm.shape != rgb.shape[:2] or fm.shape != tm.shape or
        tm.dtype != bool or fm.dtype != bool or not np.isfinite(rgb).all() or
        np.any(rgb < 0) or np.any(rgb > 255)):
        raise ValueError("Finite actual RGB and same-frame boolean role masks required")
    noises = (target_repeat_noise_px, finger_repeat_noise_px,
              target_repeat_rgb_mean_abs, finger_repeat_rgb_mean_abs)
    if not all(_finite(n) and n >= 0 for n in noises):
        return None
    if (not _registration(target_registration) or not _registration(finger_registration) or
        tm.sum() < 16 or fm.sum() < 16 or np.any(tm & fm)):
        return None
    # Background texture cannot certify a constant/unobserved role.
    if any(np.ptp(rgb[mask].astype(float), axis=0).max() <= 0 for mask in (tm, fm)):
        return None
    floor = max(2., 3. * max(target_repeat_rgb_mean_abs, finger_repeat_rgb_mean_abs))
    boundaries = [_boundary_witness(rgb.astype(float), mask, floor) for mask in (tm, fm)]
    if min(map(len, boundaries)) < 2:
        return None
    # Minimum gap uses the COMPLETE role boundaries, not a conveniently chosen
    # textured subset; a nearby untextured tip may invalidate the whole gap.
    gap = _minimum_gap(_mask_boundary(tm), _mask_boundary(fm))
    noise = max(target_repeat_noise_px, finger_repeat_noise_px)
    if gap < 2. or gap <= 3. * noise:
        return None
    registration = {k: max(target_registration[k], finger_registration[k])
                    for k in ("mean_abs", "p95", "fraction_gt5")}
    registration.update(psnr=min(target_registration["psnr"], finger_registration["psnr"]),
                        passed=True, measurement_basis="conservative_bounds_of_two_independent_role_registrations")
    return dict(physics_contact=False, other_surface_role="eef", own_finger_side=side,
        own_finger_identity_certified=True, actual_surface_gap_px=gap,
        separation_rgb_verified=True, surfaces_rgb_visible=True,
        actual_target_boundary_verified=True, actual_finger_boundary_verified=True,
        rgb_boundary_pixels=sum(map(len, boundaries)), local_rgb=registration,
        same_candidate_repeat_noise_px=noise, contrast_floor=floor,
        actual_boundary_coordinates=dict(target=boundaries[0], finger=boundaries[1]),
        boundary_measurement="actual_RGB_contrast_across_each_independent_role_silhouette",
        gap_measurement="minimum_distance_between_complete_typed_role_masks",
        mask_geometry_private_annotation_only=True,
        certifies_only_this_finger_contact=True, certified_contact_value=False,
        certifies_negative_grasp=False,
        deployment_allowed=False)
