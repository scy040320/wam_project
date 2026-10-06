"""CPU geometry for private, actual-input observability certificates.

The depth, role masks, camera state and world contact points are supervision
measurements, never deployment features. A visible interface is not a contact
force or force-closure certificate. Image comparison tolerances inherit the
frozen replay audit; 1 mm is a new surface-depth measurement tolerance.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np


OBSERVATION_GEOMETRY_CONTRACT = MappingProxyType(dict(
    rgb_mean_abs_max=2., rgb_p95_max=8., rgb_fraction_gt5_max=.06,
    rgb_psnr_min=30., minimum_role_pixels=16, contact_patch_radius_px=3.,
    surface_depth_tolerance_m=.001, raw_render_origin="bottom_left",
    camera_local_to_cv=[1., -1., -1.],
    texture_necessary_condition="nonconstant spatial RGB on each role's own patch pixels and shared patch",
    role="offline_supervision_only", physical_force_closure_certified=False))


def _image_shape(image_shape):
    if len(image_shape) not in (2, 3):
        raise ValueError("Image shape must contain height and width")
    h, w = image_shape[:2]
    if isinstance(h, (bool, np.bool_)) or isinstance(w, (bool, np.bool_)):
        raise ValueError("Image dimensions must be positive integers")
    if int(h) != h or int(w) != w or min(h, w) < 1:
        raise ValueError("Image dimensions must be positive integers")
    return int(h), int(w)


def saved_image_from_raw(image, *, image_convention=1):
    """Apply robosuite convention and the frozen saved-image vertical flip.

    ``image`` is the simulator's bottom-up raw render. Robosuite's mapping
    value, not its name, is required: +1 -> saved CV top-left; -1 -> saved
    bottom-left. RGB, segmentation and depth must use the identical transform.
    """
    if image_convention not in (-1, 1):
        raise ValueError("IMAGE_CONVENTION mapping must be +1 or -1")
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("A rendered image needs height and width")
    return np.flipud(array[::image_convention]).copy()


def project_world_points(points_m, *, camera_position_m,
                         camera_rotation_world_from_camera, fovy_degrees,
                         image_shape, image_convention=1):
    """Project without clipping, preserving off-screen coordinates.

    Rotation is camera-local -> world (MuJoCo ``cam_xmat``). OpenGL camera
    local coordinates become CV coordinates with diag(1,-1,-1). Intrinsics
    use fx=fy=height/(2*tan(fovy/2)), cx=width/2, cy=height/2. ``pixel_xy`` is
    in the *saved* image coordinates. Outputs always have leading length N,
    even for a single input point. Behind/nonfinite points get NaN pixels and
    explicit false validity; they must never be sampled from an image edge.
    """
    h, w = _image_shape(image_shape)
    if image_convention not in (-1, 1):
        raise ValueError("IMAGE_CONVENTION mapping must be +1 or -1")
    rotation = np.asarray(camera_rotation_world_from_camera, dtype=float)
    if rotation.size == 9:
        rotation = rotation.reshape(3, 3)
    origin = np.asarray(camera_position_m, dtype=float)
    if (rotation.shape != (3, 3) or origin.shape != (3,) or
            not np.isfinite(rotation).all() or not np.isfinite(origin).all()):
        raise ValueError("Camera pose must be finite with shapes (3,) and (3,3)")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0.)):
        raise ValueError("Camera rotation must be a proper orthonormal rotation")
    if not np.isfinite(fovy_degrees) or not 0. < fovy_degrees < 180.:
        raise ValueError("Camera vertical field of view must lie in (0,180)")
    points = np.asarray(points_m, dtype=float)
    if points.ndim == 1 and points.shape == (3,):
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("World points must have shape (N,3) or (3,)")
    camera = ((points - origin) @ rotation) * np.array([1., -1., -1.])
    finite = np.isfinite(camera).all(axis=1)
    depth = camera[:, 2]
    in_front = finite & (depth > 0.)
    pixels = np.full((len(points), 2), np.nan, dtype=float)
    focal = h / (2. * np.tan(np.deg2rad(float(fovy_degrees)) / 2.))
    pixels[in_front, 0] = focal * camera[in_front, 0] / depth[in_front] + w / 2.
    pixels[in_front, 1] = focal * camera[in_front, 1] / depth[in_front] + h / 2.
    if image_convention == -1:
        pixels[in_front, 1] = h - 1. - pixels[in_front, 1]
    in_bounds = (in_front & (pixels[:, 0] >= 0.) & (pixels[:, 0] < w)
                 & (pixels[:, 1] >= 0.) & (pixels[:, 1] < h))
    return dict(pixel_xy=pixels, camera_depth_m=depth, finite=finite,
                in_front=in_front, in_bounds=in_bounds, valid=in_bounds.copy(),
                image_shape=(h, w), image_convention=int(image_convention))


def _projection_arrays(projected):
    pixels = np.asarray(projected["pixel_xy"], dtype=float)
    depth = np.asarray(projected["camera_depth_m"], dtype=float)
    valid = np.asarray(projected["valid"], dtype=bool)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),) or valid.shape != depth.shape:
        raise ValueError("Projected point arrays must have consistent leading length N")
    valid = valid & np.isfinite(pixels).all(axis=1) & np.isfinite(depth) & (depth > 0.)
    return pixels, depth, valid


def depth_visibility(projected, depth_m, *, tolerance_m=.001):
    """Depth-test positive camera-z against nearest actual rendered surface.

    ``visible`` means not behind the measured surface. ``depth_match`` is
    stricter, requiring the point to lie on that surface within tolerance;
    contact witnesses require the latter. Depth must already be metric camera
    z in saved-image coordinates, not an OpenGL normalized depth buffer.
    """
    image = np.asarray(depth_m, dtype=float)
    if image.ndim != 2:
        raise ValueError("Metric surface depth must be a 2-D saved-image array")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0.:
        raise ValueError("Depth measurement tolerance must be finite and positive")
    if tuple(image.shape) != tuple(projected.get("image_shape", image.shape)):
        raise ValueError("Depth image and projection dimensions differ")
    pixels, point_depth, valid = _projection_arrays(projected)
    h, w = image.shape
    indices = np.full((len(pixels), 2), -1, dtype=np.int64)
    # Check continuous coordinates before any rounding; never clip an index.
    candidates = valid & (pixels[:, 0] >= 0.) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0.) & (pixels[:, 1] < h)
    indices[candidates] = np.floor(pixels[candidates] + .5).astype(np.int64)
    valid = candidates & (indices[:, 0] >= 0) & (indices[:, 0] < w) & (indices[:, 1] >= 0) & (indices[:, 1] < h)
    surface = np.full(len(pixels), np.nan, dtype=float)
    surface[valid] = image[indices[valid, 1], indices[valid, 0]]
    valid &= np.isfinite(surface) & (surface > 0.)
    residual = np.full(len(pixels), np.nan, dtype=float)
    residual[valid] = point_depth[valid] - surface[valid]
    return dict(valid=valid, visible=valid & (residual <= tolerance_m),
                occluded=valid & (residual > tolerance_m),
                depth_match=valid & (np.abs(residual) <= tolerance_m),
                residual_m=residual, sampled_depth_m=surface,
                sampled_pixel_xy=indices, tolerance_m=float(tolerance_m))


def _binary_mask(mask, image_shape):
    array = np.asarray(mask)
    if array.shape != image_shape or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError("ROI/role masks must be binary and match image dimensions")
    return array.astype(bool)


def local_rgb_registration(actual_rgb, rendered_rgb, roi_mask, *, texture_mask=None):
    """Frozen replay image metrics measured locally on explicit binary support.

    Empty/nonfinite support and spatially constant texture abstain. Texture is
    a necessary content condition, not a learned confidence/model threshold.
    A shared texture mask can include the interface while pixel error remains
    measured independently on each role's selected pixels.
    """
    actual = np.asarray(actual_rgb, dtype=float)
    rendered = np.asarray(rendered_rgb, dtype=float)
    if actual.ndim != 3 or actual.shape[-1] != 3 or rendered.shape != actual.shape:
        raise ValueError("Actual and rendered RGB must have identical HxWx3 shapes")
    roi = _binary_mask(roi_mask, actual.shape[:2])
    texture = roi if texture_mask is None else _binary_mask(texture_mask, actual.shape[:2])
    count = int(roi.sum()); texture_count = int(texture.sum())
    report = dict(mean_abs=None, p95=None, fraction_gt5=None, psnr=None,
                  roi_pixels=count, texture_pixels=texture_count,
                  actual_texture_std=None, rendered_texture_std=None,
                  actual_texture_range=None, rendered_texture_range=None,
                  measurement_valid=False, pixel_metrics_passed=False,
                  texture_sufficient=False, passed=False, reason="empty_roi")
    if not count or not texture_count:
        return report
    a, r = actual[roi], rendered[roi]
    ta, tr = actual[texture], rendered[texture]
    if (not all(np.isfinite(x).all() for x in (a, r, ta, tr))
            or any(np.any((x < 0.) | (x > 255.)) for x in (a, r, ta, tr))):
        report["reason"] = "invalid_rgb_measurement"
        return report
    delta = a - r; absolute = np.abs(delta); mse = float(np.mean(delta * delta))
    report.update(mean_abs=float(absolute.mean()), p95=float(np.quantile(absolute, .95)),
                  fraction_gt5=float(np.mean(absolute > 5.)),
                  psnr=99. if mse == 0. else float(20. * np.log10(255. / np.sqrt(mse))),
                  actual_texture_std=float(np.max(np.std(ta, axis=0))),
                  rendered_texture_std=float(np.max(np.std(tr, axis=0))),
                  actual_texture_range=float(np.max(np.ptp(ta, axis=0))),
                  rendered_texture_range=float(np.max(np.ptp(tr, axis=0))),
                  measurement_valid=True)
    c = OBSERVATION_GEOMETRY_CONTRACT
    report["pixel_metrics_passed"] = bool(report["mean_abs"] <= c["rgb_mean_abs_max"]
        and report["p95"] <= c["rgb_p95_max"]
        and report["fraction_gt5"] <= c["rgb_fraction_gt5_max"]
        and report["psnr"] >= c["rgb_psnr_min"])
    report["texture_sufficient"] = bool(report["actual_texture_range"] > 0. and report["rendered_texture_range"] > 0.)
    report["passed"] = report["pixel_metrics_passed"] and report["texture_sufficient"]
    report["reason"] = ("passed" if report["passed"] else
        "insufficient_local_texture" if not report["texture_sufficient"] else "local_rgb_mismatch")
    return report


def contact_interface_witness(projected, *, depth_m, target_mask, finger_mask,
                              actual_rgb, rendered_rgb):
    """Conservative one-point, one-camera interface witness with both roles.

    Both roles need >=16 visible pixels in the reference render, at least one
    pixel on the 3-pixel contact patch at matching camera depth, and separate
    local RGB registration with nonconstant texture on each role's own pixels.
    Shared full patch registration/texture also must pass; textured background
    cannot supply either role's missing evidence. Mere role co-occurrence
    elsewhere in the image cannot pass. The
    result supports visual interface observability only; it never proves
    physical contact, absence of contact, or a stable grasp.
    """
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("Contact depth must be a 2-D metric array")
    target = _binary_mask(target_mask, depth.shape)
    finger = _binary_mask(finger_mask, depth.shape)
    pixels, point_depth, valid = _projection_arrays(projected)
    if len(pixels) != 1:
        raise ValueError("A contact-interface witness requires exactly one projected point")
    if tuple(projected.get("image_shape", depth.shape)) != tuple(depth.shape):
        raise ValueError("Projection and contact depth image dimensions differ")
    if np.any(target & finger):
        raise ValueError("Rendered target and finger roles must be disjoint")
    c = OBSERVATION_GEOMETRY_CONTRACT
    radius = c["contact_patch_radius_px"]; tolerance = c["surface_depth_tolerance_m"]
    h, w = depth.shape; xy = pixels[0]
    full_patch = bool(valid[0] and 0. <= xy[0] - radius and xy[0] + radius <= w - 1.
                      and 0. <= xy[1] - radius and xy[1] + radius <= h - 1.)
    patch = np.zeros(depth.shape, dtype=bool)
    if full_patch:
        ys, xs = np.ogrid[:h, :w]
        patch = ((xs - xy[0]) ** 2 + (ys - xy[1]) ** 2) <= radius ** 2
    target_patch = patch & target; finger_patch = patch & finger
    matched_depth = np.isfinite(depth) & (depth > 0.)
    if valid[0]:
        matched_depth &= np.abs(depth - point_depth[0]) <= tolerance
    else:
        matched_depth[:] = False
    roles = dict(target=int(target.sum()), finger=int(finger.sum()))
    role_floor = all(n >= c["minimum_role_pixels"] for n in roles.values())
    counts = dict(target=int(target_patch.sum()), finger=int(finger_patch.sum()))
    matched = dict(target=int((target_patch & matched_depth).sum()),
                   finger=int((finger_patch & matched_depth).sum()))
    shared_reg = local_rgb_registration(actual_rgb, rendered_rgb, patch)
    target_reg = local_rgb_registration(actual_rgb, rendered_rgb, target_patch)
    finger_reg = local_rgb_registration(actual_rgb, rendered_rgb, finger_patch)
    center_test = depth_visibility(projected, depth, tolerance_m=tolerance)
    passed = bool(full_patch and role_floor and all(n > 0 for n in matched.values())
                  and center_test["depth_match"][0] and shared_reg["passed"]
                  and target_reg["passed"] and finger_reg["passed"])
    reasons = []
    if not full_patch: reasons.append("projected_contact_or_full_patch_outside_input")
    if not role_floor: reasons.append("insufficient_reference_role_pixels")
    if not all(n > 0 for n in counts.values()): reasons.append("roles_not_both_at_contact_patch")
    if not all(n > 0 for n in matched.values()): reasons.append("role_surface_depth_not_matched")
    if not center_test["depth_match"][0]: reasons.append("contact_center_depth_not_matched")
    if not shared_reg["passed"] or not target_reg["passed"] or not finger_reg["passed"]:
        reasons.append("local_input_registration_or_texture_insufficient")
    return dict(passed=passed, supports_interface_observability=passed,
                role="offline_supervision_only", physical_contact_certified=False,
                physical_force_closure_certified=False, full_patch_in_bounds=full_patch,
                contact_pixel_xy=[float(v) if np.isfinite(v) else None for v in xy],
                contact_depth_m=float(point_depth[0]) if np.isfinite(point_depth[0]) else None,
                role_visible_pixels=roles, role_reference_floor_passed=role_floor,
                role_patch_pixels=counts, role_depth_matched_patch_pixels=matched,
                shared_patch_registration=shared_reg, target_registration=target_reg,
                finger_registration=finger_reg, patch_radius_px=float(radius),
                surface_depth_tolerance_m=float(tolerance), reasons=reasons)


def track_rgb_role(start_rgb, end_rgb, start_mask, end_mask, projected_endpoints=None):
    """Witness actual RGB tracks, optionally tied to projected surface points.

    Requires >=3 corner tracks, successful forward/backward Lucas-Kanade,
    <=1 px forward/back error, and an endpoint in the same rendered role.
    Optional ``projected_endpoints`` is a dict with ``start_pixel_xy`` and
    ``end_pixel_xy`` Nx2 arrays and optional N ``valid`` flags; a callable may
    receive detected start corner pixels and return this dict, allowing exact
    per-corner depth/body reprojection by an external private teacher adapter.
    The optional Nx2 ``camera_only_end_pixel_xy`` projects the unchanged start
    world point into the end camera. It is returned as private per-track data
    so an adapter can compare actual flow against camera-only flow; an end
    projection alone does not constitute camera compensation.
    Start association and endpoint error must each be <=1 px, with unique
    surface-point associations. Geometry agreement supports correspondence,
    not metric world-motion magnitude by itself. Without geometry the result
    is explicitly only ``RGB_tracking_proxy``. No segmentation-centroid motion
    is substituted for actual-image optical flow. Missing cv2 abstains.
    """
    start = np.asarray(start_rgb, dtype=float); end = np.asarray(end_rgb, dtype=float)
    if start.ndim != 3 or start.shape[-1] != 3 or end.shape != start.shape:
        raise ValueError("RGB tracking requires identical HxWx3 actual images")
    sm = _binary_mask(start_mask, start.shape[:2]); em = _binary_mask(end_mask, start.shape[:2])
    geometry_supplied = projected_endpoints is not None
    report = dict(verified=False, count=0, detected_count=0, forward_backward_count=0,
                  role_endpoint_count=0, geometry_matched_count=0,
                  median_delta_px=None, error=None, median_endpoint_error_px=None,
                  geometry_motion_support=False, metric_world_motion_certified=False,
                  evidence_kind="actual_RGB_geometry_correspondence" if geometry_supplied else "RGB_tracking_proxy",
                  backend="opencv_LK", forward_backward_tolerance_px=1.,
                  projected_correspondence_tolerance_px=1., required_track_count=3,
                  role="offline_supervision_only", reason="empty_role_mask",
                  tracks=[])
    if not sm.any() or not em.any():
        return report
    if not np.isfinite(start).all() or not np.isfinite(end).all() or any(np.any((a < 0.) | (a > 255.)) for a in (start, end)):
        report["reason"] = "invalid_actual_rgb"
        return report
    if max(np.ptp(start[sm], axis=0)) == 0. or max(np.ptp(end[em], axis=0)) == 0.:
        report["reason"] = "insufficient_actual_role_texture"
        return report
    try:
        import cv2
    except ImportError:
        report["reason"] = "opencv_backend_unavailable"
        return report
    sgray = cv2.cvtColor(np.rint(start).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    egray = cv2.cvtColor(np.rint(end).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(sgray, maxCorners=100, qualityLevel=.01,
        minDistance=3., mask=sm.astype(np.uint8) * 255, blockSize=3)
    if corners is None:
        report["reason"] = "no_actual_rgb_corners"
        return report
    p0 = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    report["detected_count"] = len(p0)
    if len(p0) < 3:
        report["reason"] = "insufficient_actual_rgb_corners"
        return report
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)
    p1, sf, _ = cv2.calcOpticalFlowPyrLK(sgray, egray, p0.reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=3, criteria=criteria)
    if p1 is None or sf is None:
        report["reason"] = "forward_rgb_tracking_failed"
        return report
    p1 = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    if p1.shape != p0.shape or np.asarray(sf).size != len(p0):
        raise ValueError("OpenCV forward tracker returned mismatched arrays")
    if not np.isfinite(p1).all():
        report["reason"] = "nonfinite_forward_rgb_tracks"
        return report
    pback, sb, _ = cv2.calcOpticalFlowPyrLK(egray, sgray, p1.reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=3, criteria=criteria)
    if pback is None or sb is None:
        report["reason"] = "backward_rgb_tracking_failed"
        return report
    pback = np.asarray(pback, dtype=np.float32).reshape(-1, 2)
    if pback.shape != p0.shape or np.asarray(sb).size != len(p0):
        raise ValueError("OpenCV backward tracker returned mismatched arrays")
    fb = np.linalg.norm(pback - p0, axis=1)
    keep = np.asarray(sf).reshape(-1).astype(bool) & np.asarray(sb).reshape(-1).astype(bool)
    keep &= np.isfinite(pback).all(axis=1) & np.isfinite(fb) & (fb <= 1.)
    report["forward_backward_count"] = int(keep.sum())
    h, w = sm.shape

    def inside_role(points, mask):
        finite = np.isfinite(points).all(axis=1)
        inside = finite & (points[:, 0] >= 0.) & (points[:, 0] < w) & (points[:, 1] >= 0.) & (points[:, 1] < h)
        idx = np.full(points.shape, -1, np.int64)
        idx[inside] = np.floor(points[inside] + .5).astype(np.int64)
        inside &= (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
        selected = np.zeros(len(points), bool)
        selected[inside] = mask[idx[inside, 1], idx[inside, 0]]
        return selected

    keep &= inside_role(p0, sm) & inside_role(p1, em)
    report["role_endpoint_count"] = int(keep.sum())
    endpoint_error = np.full(len(p0), np.nan)
    geometry_index = np.full(len(p0), -1, dtype=np.int64)
    geometry_end = None; camera_only_end = None
    if geometry_supplied:
        ep = projected_endpoints(p0.copy()) if callable(projected_endpoints) else projected_endpoints
        if not isinstance(ep, dict) or not {"start_pixel_xy", "end_pixel_xy"}.issubset(ep):
            raise ValueError("Projected RGB endpoints require start_pixel_xy/end_pixel_xy")
        g0 = np.asarray(ep["start_pixel_xy"], dtype=float); g1 = np.asarray(ep["end_pixel_xy"], dtype=float)
        if g0.ndim != 2 or g0.shape[1] != 2 or g1.shape != g0.shape:
            raise ValueError("Projected endpoint arrays must be matching Nx2 arrays")
        geometry_end = g1
        if "camera_only_end_pixel_xy" in ep:
            camera_only_end = np.asarray(ep["camera_only_end_pixel_xy"], dtype=float)
            if camera_only_end.shape != g0.shape:
                raise ValueError("Camera-only endpoint information must match Nx2 projected endpoints")
        gv = np.asarray(ep.get("valid", np.ones(len(g0), bool)), dtype=bool).copy()
        if gv.shape != (len(g0),):
            raise ValueError("Projected endpoint validity must have length N")
        gv &= np.isfinite(g0).all(axis=1) & np.isfinite(g1).all(axis=1)
        gv &= inside_role(g0, sm) & inside_role(g1, em)
        matched = np.zeros(len(p0), bool)
        # Globally greedy unique associations avoid counting three RGB tracks
        # against one projected simulator point as three independent witnesses.
        pairs = []
        for i in np.flatnonzero(keep):
            for j in np.flatnonzero(gv):
                distance = float(np.linalg.norm(p0[i] - g0[j]))
                end_error = float(np.linalg.norm(p1[i] - g1[j]))
                if distance <= 1. and end_error <= 1.:
                    pairs.append((distance, end_error, int(i), int(j)))
        used = set()
        for _, err, i, j in sorted(pairs):
            if matched[i] or j in used:
                continue
            matched[i] = True; used.add(j); endpoint_error[i] = err; geometry_index[i] = j
        keep &= matched
        report["geometry_matched_count"] = int(keep.sum())
    report["count"] = int(keep.sum())
    if keep.any():
        report["median_delta_px"] = np.median(p1[keep] - p0[keep], axis=0).astype(float).tolist()
        report["error"] = float(np.max(fb[keep]))
        if geometry_supplied:
            report["median_endpoint_error_px"] = float(np.median(endpoint_error[keep]))
        report["tracks"] = [dict(start_pixel_xy=p0[i].astype(float).tolist(),
            end_pixel_xy=p1[i].astype(float).tolist(), forward_backward_error_px=float(fb[i]),
            projected_endpoint_error_px=float(endpoint_error[i]) if geometry_supplied else None,
            projected_end_pixel_xy=geometry_end[geometry_index[i]].astype(float).tolist() if geometry_supplied else None,
            camera_only_end_pixel_xy=(camera_only_end[geometry_index[i]].astype(float).tolist()
                if camera_only_end is not None and np.isfinite(camera_only_end[geometry_index[i]]).all() else None))
            for i in np.flatnonzero(keep)]
    report["verified"] = report["count"] >= 3
    report["geometry_motion_support"] = bool(report["verified"] and geometry_supplied)
    report["reason"] = ("passed_geometry_correspondence" if report["geometry_motion_support"] else
        "passed_rgb_tracking_proxy" if report["verified"] else "insufficient_verified_actual_rgb_tracks")
    return report
