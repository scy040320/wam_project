"""Frozen RGB-only re-extraction of the original 192 PRE candidate pools.

CURRENT maps are inferred once per pool from actual query RGB only. Each
candidate's FORECAST maps are inferred separately from its predicted RGB.
Shared CURRENT camera fusion replaces no frozen cache or model: this script
writes a versioned sidecar, with uncalibrated image-space proxies only.

Run a fixed two-TRAIN-pool engineering pilot first, then --phase full with its
passed --pilot-output. The pilot is chosen by identities, never outcomes. All
192 pools, including failed pools, are retained in the full phase. No Cosmos
query, environment action, fitting, selection or physical fact admission runs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np


SCHEMA = "shared_actual_CURRENT_own_FORECAST_visual_sidecar_v1"
HELPER = "research_runs/known_task_predicate_scoped_replay_20261008_v3_preserve_history/replay_evidence_scoped_routes.py"
ANCHOR = "outputs/known_task_evidence_interface_recheck_20261008_v4_final_boundaries_roots"
MODEL_CACHE = Path("/root/autodl-tmp/wam_project/model_cache/clipseg-rd64-refined")
MODEL_FILES = {
    "config.json": "c023375966d31b3b1392764f7bd91df47098ce19f62f11b0263d8eedcf708bcd",
    "model.safetensors": "d00ca85d6b859f9d07b7cfb8ef26fe9771cb275b34c9368f2ecf603139307f55",
    "preprocessor_config.json": "4fb09ebcfd7651205ca8299b993c30088e5535ef350a72d46c5c4580eeac0440",
    "tokenizer_config.json": "4c75d57fd9bd0be8478ad2d6f8b9cebdd4a45338eb108547329c2b6333476ca6",
    "special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
    "vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
}
EXPECTED_BINDINGS = {
    (0, 0): ("top drawer", "wooden cabinet frame", "articulated"),
    (9, 0): ("black bowl", "plate", "on"),
    (20, 0): ("stove button", "stove", "articulated"),
    (44, 0): ("stove button", "stove", "articulated"),
    (46, 0): ("alphabet soup", "basket", "inside"),
    (57, 0): ("cream cheese", "tray", "inside"),
}
ROLES = ("subject", "anchor", "gripper")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def inside(base, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Explicit relative source reference required")
    path = (Path(base) / relative).resolve()
    if not path.is_relative_to(Path(base).resolve()) or path == Path(base).resolve():
        raise ValueError("Source reference escapes its frozen directory")
    return path


def load_helper(root):
    path = root / HELPER
    spec = importlib.util.spec_from_file_location("shared_current_frozen_192_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.ROOT.resolve() != root:
        raise ValueError("Immutable helper root differs from explicitly requested root")
    return module


def load_shared_payload(helper, payload_path=None):
    """Load the new additive module from the versioned payload, never repo."""
    path = (Path(payload_path) if payload_path is not None else
            Path(__file__).resolve().parent / "shared_current_visual_baseline.py").resolve()
    if not path.is_file():
        raise ValueError("Shared CURRENT module missing from this isolated payload: " + str(path))
    module = helper.rt.load("wam_reranking.shared_current_visual_baseline", path)
    # rt.load registers sys.modules; also expose the module on the already
    # frozen package for consistent relative/package imports. No file is copied.
    setattr(sys.modules["wam_reranking"], "shared_current_visual_baseline", module)
    return module


def engineering_pilot_keys(pairs):
    """First pool in each of the first two TRAIN tasks; never inspect Y."""
    picked = {}
    for key in sorted(pairs):
        if pairs[key]["split"] == "train":
            picked.setdefault(key[0], key)
    if len(picked) < 2:
        raise ValueError("Two TRAIN tasks required for the fixed engineering pilot")
    return tuple(picked[task] for task in sorted(picked)[:2])


def strict_rgb(path):
    from PIL import Image
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Actual/forecast RGB source must be literal uint8 H x W x 3: " + str(path))
    return Image.fromarray(array, "RGB"), dict(shape=list(array.shape), dtype=str(array.dtype))


def verify_bindings(bindings):
    for key, expected in EXPECTED_BINDINGS.items():
        item = bindings.get(key)
        if item is None or (item.subject, item.anchor, item.relation) != expected:
            raise ValueError("Frozen LIBERO90 task/stage semantic binding changed; never use default LIBERO10")


def pin_identical_cache_copy(path, source_path, pins, expected):
    """Copy must equal the independently manifest-pinned authoritative source."""
    path, source_path = Path(path).resolve(), Path(source_path).resolve()
    if str(source_path) not in pins:
        raise ValueError("Authoritative source must be independently pinned before verifying cache copy")
    digest = sha(path)
    if digest != pins[str(source_path)]:
        raise ValueError("Frozen candidate cache/source file differs")
    if str(path) in expected and digest != expected[str(path)]:
        raise ValueError("Cache copy differs from its existing immutable audit hash")
    pins[str(path)] = digest
    return path


def infer_role_maps(localizer, images, prompts):
    maps = {}
    for role, prompt in zip(ROLES, prompts, strict=True):
        array = np.asarray(localizer.relevance(images, prompt), dtype=np.float32)
        if (array.ndim != 3 or array.shape[0] != len(images) or
                not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1)):
            raise ValueError("Frozen localizer returned nonfinite/misaligned relevance maps")
        maps[role] = array.copy()
        maps[role].setflags(write=False)
    return maps


def reextract_pool(*, localizer, current_images, forecast_images, binding, shared_module):
    """CURRENT-only three calls; own FORECASTs never enter those calls."""
    if len(current_images) != 2 or len(forecast_images) != 4 or any(len(v) != 2 for v in forecast_images):
        raise ValueError("Exactly dual CURRENT views and four dual-view FORECASTs required")
    prompts = (binding.subject, binding.anchor, "robot gripper")
    current = infer_role_maps(localizer, list(current_images), prompts)
    # FORECAST-only batching is separate from CURRENT. Candidate i has rows
    # 2i/2i+1 in this batch; no candidate can change CURRENT extraction/fusion.
    flat_forecasts = [image for views in forecast_images for image in views]
    forecast = infer_role_maps(localizer, flat_forecasts, prompts)

    def build(predicted):
        return shared_module.build_shared_current_visual_evidence(
            current_subject_maps=current["subject"], predicted_subject_maps=predicted["subject"],
            current_anchor_maps=current["anchor"], predicted_anchor_maps=predicted["anchor"],
            current_gripper_maps=current["gripper"], predicted_gripper_maps=predicted["gripper"],
            relation=binding.relation)

    # The shared baseline/weights are defined once using CURRENT only. The
    # public builder recomputes the same deterministic arithmetic per effect;
    # equality is checked rather than treating its candidate quality as PRE.
    _, baseline = build(current)
    evidences, diagnostics, candidate_maps = [], [], []
    for cid in range(4):
        predicted = {role: forecast[role][cid * 2:cid * 2 + 2] for role in ROLES}
        evidence, diagnostic = build(predicted)
        for name in ("before", "relation_weights", "contact_weights"):
            if diagnostic[name] != baseline[name]:
                raise ValueError("Candidate forecast altered the shared CURRENT baseline")
        evidences.append(evidence); diagnostics.append(diagnostic); candidate_maps.append(predicted)
    audit = shared_module.audit_shared_current_pool(evidences, diagnostics)
    candidates = []
    for cid, (evidence, diagnostic) in enumerate(zip(evidences, diagnostics, strict=True)):
        visual = asdict(evidence)
        candidates.append(dict(candidate_id=cid, candidate_visual_evidence=visual,
            forecast_expected_effects={name: value for name, value in visual.items()
                                       if name not in shared_module.BEFORE_FIELDS},
            forecast_quality=dict(relation_confidence=evidence.relation_confidence,
                contact_confidence=evidence.contact_confidence,
                visibility_confidence=evidence.visibility_confidence,
                cross_view_agreement=evidence.cross_view_agreement),
            fusion_diagnostic=diagnostic, source_role="own_candidate_forecast_with_shared_actual_PRE_baseline",
            physical_state_certified=False, predicted_effects_are_current_facts=False))
    return dict(current_maps=current, candidate_maps=candidate_maps, candidates=candidates,
        shared_current=dict(source_role="actual_query_CURRENT_only", before=baseline["before"],
            relation_weights=baseline["relation_weights"], contact_weights=baseline["contact_weights"],
            current_relation_quality=baseline["current_relation_quality"],
            current_contact_quality=baseline["current_contact_quality"],
            physical_state_certified=False, confidence_calibrated=False, predicate_facts=[]),
        audit=audit | dict(current_localizer_calls=3, forecast_localizer_calls=3,
            current_views_per_prompt=2, forecast_views_per_prompt=8,
            current_inference_repeated_per_candidate=False))


def source_index(helper, root):
    """Verify the complete frozen scope even during the two-pool pilot."""
    pins = {}
    expected = {}
    for name in ("source_sha256.json", "addition_sha256.json"):
        path = root / ANCHOR / name
        pins[str(path)] = sha(path)
        for source, digest in read(path).items():
            if source in expected and expected[source] != digest:
                raise ValueError("Frozen audit manifests disagree on source identity")
            expected[source] = digest

    def pin(path):
        path = Path(path).resolve()
        digest = sha(path)
        if expected.get(str(path)) != digest:
            raise ValueError("Source missing or different in frozen full-scope audit: " + str(path))
        pins[str(path)] = digest
        return path

    pair_path = pin(helper.CACHE / "paired_scenarios.json")
    pairs = helper.io.index_unique(read(pair_path))
    if len(pairs) != 192:
        raise ValueError("Original192 cache membership changed")
    if not read(pin(helper.SOURCE / "completion_audit.json"))["passed"]:
        raise ValueError("Frozen original source audit is not passed")
    raw = {}
    for result_path in sorted((helper.SOURCE / "outcomes").glob("*/results.json")):
        for row in read(pin(result_path)):
            key = helper.io.key(row)
            if key in raw:
                raise ValueError("Duplicate frozen source pool")
            raw[key] = (row, result_path.parent)
    if set(raw) != set(pairs):
        raise ValueError("Full original source/cache pool identities differ")
    verify_bindings(helper.bundle.FROZEN_RELATIONS)
    sources = {}
    for key in sorted(pairs):
        pair, (row, folder) = pairs[key], raw[key]
        if pair["split"] != ("train" if key[1] <= 5 else "val"):
            raise ValueError("Original task/state split changed")
        if [v["candidate_id"] for v in row["candidates"]] != list(range(4)):
            raise ValueError("Original K4 candidate membership changed")
        current = [pin(inside(folder, row[f"current_{view}_path"])) for view in ("primary", "wrist")]
        candidates = []
        for candidate in row["candidates"]:
            record = {"candidate_id": candidate["candidate_id"]}
            for name in ("actions_path", "predicted_primary_path", "predicted_wrist_path"):
                path = pin(inside(folder, candidate[name]))
                # Prior full-scope audits pin the authoritative source arrays;
                # immutable cache copies were compared bytewise, not all listed
                # independently. Require that same pinned-source equality here.
                pin_identical_cache_copy(inside(helper.CACHE / "candidate_outcomes" / folder.name,
                                                candidate[name]), path, pins, expected)
                record[name] = path
            plans = np.load(record["actions_path"], allow_pickle=False)
            if plans.shape != (16, 7) or plans.dtype.kind not in "fiu" or not np.isfinite(plans).all():
                raise ValueError("Original frozen planned actions are invalid")
            candidates.append(record)
        sources[key] = dict(current=current, candidates=candidates, split=pair["split"])
    if sum(value["split"] == "train" for value in sources.values()) != 144:
        raise ValueError("Original144/48 split changed")
    return pairs, sources, pins


def model_audit(cache):
    manifest_path = cache / "FROZEN_MANIFEST.json"
    manifest = read(manifest_path)
    declared = {item["path"]: item["sha256"] for item in manifest["files"]}
    if manifest.get("model_id") != "CIDAS/clipseg-rd64-refined" or declared != MODEL_FILES:
        raise ValueError("Frozen CLIPSeg seven-file identity changed")
    pins = {str(manifest_path): sha(manifest_path)}
    for name, expected in MODEL_FILES.items():
        path = cache / name
        if sha(path) != expected:
            raise ValueError("Frozen CLIPSeg source SHA256 mismatch: " + name)
        pins[str(path)] = expected
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/wam_project/cosmos-policy"))
    parser.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "outputs"):
        raise ValueError("Fresh additive outputs directory required")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    helper = load_helper(root)
    shared = load_shared_payload(helper)
    pairs, sources, pins = source_index(helper, root)
    pins.update(model_audit(args.model_cache.resolve()))
    for module in (helper, helper.bundle, shared):
        pins[str(Path(module.__file__).resolve())] = sha(module.__file__)
    # Transitive arithmetic must be pinned as well as the additive wrapper.
    # A wrapper hash alone would miss changed centroid/quality fusion code.
    for primitive in (shared._centroid_geometry, shared.build_candidate_visual_evidence):
        arithmetic_source = Path(inspect.getfile(primitive)).resolve()
        pins[str(arithmetic_source)] = sha(arithmetic_source)
    localizer_class = helper.bundle.CLIPSegTargetLocalizer
    constructor = inspect.signature(localizer_class).parameters
    if not {"device", "model_path", "local_files_only"}.issubset(constructor):
        raise ValueError("Frozen localizer constructor no longer supports strict offline cache binding")
    class_source = Path(inspect.getfile(localizer_class)).resolve()
    pins[str(class_source)] = sha(class_source)
    script_path = Path(__file__).resolve()
    pins[str(script_path)] = sha(script_path)
    pilot_keys = engineering_pilot_keys(pairs)
    selected = pilot_keys if args.phase == "pilot" else tuple(sorted(pairs))
    if args.phase == "full":
        if args.pilot_output is None:
            raise ValueError("Passed two-TRAIN-pool engineering pilot required before full")
        pilot = read(args.pilot_output / "completion_audit.json")
        if (pilot.get("passed") is not True or pilot.get("schema") != SCHEMA or
                pilot.get("phase") != "pilot" or pilot.get("keys") != [list(k) for k in pilot_keys] or
                pilot.get("source_sha256") != pins or pilot.get("device") != args.device):
            raise ValueError("Pilot identity/model/source/phase/device contract differs")
    output.mkdir(parents=True)
    protocol = dict(schema=SCHEMA, phase=args.phase, device=args.device,
        complete_input_scope=192, train=144, val=48, engineering_pilot_keys=[list(k) for k in pilot_keys],
        selected_output_pools=len(selected), pilot_selection_uses_outcomes=False,
        all_failed_pools_retained_by_membership=True, old_caches_modified=False,
        physical_state_certificates_created=0, fits=0, policy_queries=0, actions=0,
        localizer="frozen CLIPSeg RGB+language inference only",
        current_baseline="actual query RGB only, shared across K4; no forecast-conditioned camera weights",
        forecast_expected_effects="own predicted RGB relative to same shared CURRENT image-space proxy",
        new_schema_not_directly_injected_into_frozen_models=True,
        original_source_role="source/cache hashes bound; no new physical or per-query timestamp certification")
    dump(output / "protocol.json", protocol)
    dump(output / "source_sha256.json", pins)
    try:
        import torch
        torch.manual_seed(20261009)
        torch.use_deterministic_algorithms(True)
        localizer = localizer_class(device=args.device, model_path=str(args.model_cache), local_files_only=True)
        rows, output_pins = [], {}
        for number, key in enumerate(selected):
            source = sources[key]
            current, forecast, source_rgb = [], [], dict(current=[], forecast=[])
            for path in source["current"]:
                image, metadata = strict_rgb(path); current.append(image)
                source_rgb["current"].append(metadata | dict(path=str(path), sha256=pins[str(path)]))
            for candidate in source["candidates"]:
                images, refs = [], []
                for name in ("predicted_primary_path", "predicted_wrist_path"):
                    path = candidate[name]; image, metadata = strict_rgb(path); images.append(image)
                    refs.append(metadata | dict(path=str(path), sha256=pins[str(path)]))
                forecast.append(images); source_rgb["forecast"].append(refs)
            binding = helper.bundle.FROZEN_RELATIONS[(key[0], key[2])]
            extracted = reextract_pool(localizer=localizer, current_images=current,
                forecast_images=forecast, binding=binding, shared_module=shared)
            folder = output / ("pool_" + "_".join(map(str, key)))
            folder.mkdir()
            current_path = folder / "shared_actual_CURRENT_maps.npz"
            np.savez_compressed(current_path, **extracted.pop("current_maps"))
            output_pins[str(current_path.relative_to(output))] = sha(current_path)
            for cid, maps in enumerate(extracted.pop("candidate_maps")):
                path = folder / f"candidate_{cid}_own_FORECAST_maps.npz"
                np.savez_compressed(path, **maps)
                output_pins[str(path.relative_to(output))] = sha(path)
                extracted["candidates"][cid]["forecast_maps"] = str(path.relative_to(output))
                extracted["candidates"][cid]["planned_actions_source"] = dict(
                    path=str(source["candidates"][cid]["actions_path"]),
                    sha256=pins[str(source["candidates"][cid]["actions_path"])])
            record = extracted | dict(schema=SCHEMA, key=list(key), split=source["split"],
                suite="libero90", frozen_language_binding=asdict(binding), source_rgb=source_rgb,
                shared_current_maps=str(current_path.relative_to(output)),
                old_visual_cache_replaced=False, terminal_outcomes_used=False,
                actual_after_execution_images_used=False, physical_facts_created=[])
            path = folder / "visual_sidecar.json"; dump(path, record)
            output_pins[str(path.relative_to(output))] = sha(path)
            rows.append(dict(key=list(key), split=source["split"], candidates=4,
                visual_sidecar=str(path.relative_to(output)), before_candidate_invariant=True,
                CURRENT_source_separate_from_FORECAST=True, physical_facts_created=0))
            dump(output / "status.json", dict(phase="reextracting", completed_pools=number + 1,
                total_pools=len(selected), current_key=list(key), time_unix=time.time(),
                policy_queries=0, actions=0, fits=0))
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError("Frozen source/model/code mutated during extraction")
        report = dict(passed=True, schema=SCHEMA, phase=args.phase, device=args.device,
            keys=[list(k) for k in selected], pools=len(selected), candidates=4 * len(selected),
            train=sum(row["split"] == "train" for row in rows), val=sum(row["split"] == "val" for row in rows),
            full192_source_identity_matches=len(sources), before_invariance_passed_pools=len(rows),
            current_inference_calls=3 * len(selected), forecast_inference_calls=3 * len(selected),
            current_model_inference_per_candidate=False, source_sha256=pins,
            output_sha256=output_pins, rows=rows, physical_TRUE_facts_created=0,
            old_cache_or_frozen_models_modified=False, policy_queries=0, actions=0, fits=0,
            downstream_scoring_or_closedloop_not_run=True,
            new_feature_distribution_requires_training_support_audit=True)
        dump(output / "completion_audit.json", report)
        dump(output / "status.json", dict(phase="completed", completed_pools=len(rows),
            completed_candidates=len(rows) * 4, passed=True, policy_queries=0, actions=0, fits=0))
        print(json.dumps({name: value for name, value in report.items()
                          if name not in ("source_sha256", "output_sha256", "rows")}, ensure_ascii=False))
    except BaseException as error:
        dump(output / "startup_or_extraction_failure.json", dict(schema=SCHEMA,
            error_type=type(error).__name__, error=str(error), source_or_thresholds_changed=False,
            policy_queries=0, actions=0, fits=0))
        raise


if __name__ == "__main__":
    main()
