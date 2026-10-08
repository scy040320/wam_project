"""Read-only structural audit of the existing 160 actual calibration windows.

Reads only frozen source identities, local indices, array shapes and hashes;
no physics/label/outcome is converted to X, and no runtime fact is admitted.
These are AFTER-execution, offline calibration sources, not history available
at the original 192 PRE decisions. A query block number is deliberately not
converted into a supposedly captured absolute policy timestamp.

Default output is JSON on stdout. --output may save it in a fresh derived
directory, never in either immutable source directory. This script neither
loads Cosmos nor resets, renders, steps, fits or queries any environment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


DERIVED_SOURCE = "known_task_recovery_observation_audit_20261006_v5_motion_typed_support_full"
ORIGINAL_SOURCE = "known_task_recovery_v9_20261006_v5_paired_recovery"
SCHEMA = "offline_actual_window_source_structure_audit_v1"
ARRAYS = {
    "primary_sequence": ("primary.npy", (17, 256, 256, 3), "rgb"),
    "wrist_sequence": ("wrist.npy", (17, 256, 256, 3), "rgb"),
    "proprio_sequence": ("proprio.npy", (17, 9), "numeric"),
    "requested_actions": ("requested.npy", (16, 7), "numeric"),
    "applied_actions": ("applied.npy", (16, 7), "numeric"),
    "proprio_start": ("proprio_start.npy", (9,), "numeric"),
    "proprio_end": ("proprio_end.npy", (9,), "numeric"),
}
_ARRAY_HASH = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_sha(array):
    """Same canonical NPY hash as current_observation_evidence.array_sha256."""
    # The script can be uploaded alone beside frozen code. Import the existing
    # actual-source hash helper, without constructing a fake runtime snapshot.
    global _ARRAY_HASH
    if _ARRAY_HASH is None:
        try:
            from wam_reranking.current_observation_evidence import array_sha256
            _ARRAY_HASH = array_sha256
        except ModuleNotFoundError as error:
            if error.name != "wam_reranking.current_observation_evidence":
                raise
            # Isolated cloud experiments deliberately do not overwrite the
            # frozen repository package. Use the reviewed payload beside this
            # auditor, not an unrelated installed version or a fabricated hash.
            path = Path(__file__).resolve().with_name("current_observation_evidence.py")
            spec = importlib.util.spec_from_file_location("offline_actual_array_hash_payload", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            _ARRAY_HASH = module.array_sha256
    return _ARRAY_HASH(array)


def inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Only explicit relative source paths are accepted")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Source path escapes its manifest root")
    return path


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=invalid)


def manifest(path):
    result = {}
    for line in Path(path).read_text(encoding="utf8").splitlines():
        digest, name = line.split("  ", 1)
        if (name in result or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Duplicate reference or malformed SHA256 manifest")
        result[name] = digest
    return result


def pinned(root, relative, hashes, observed):
    path = inside(root, relative)
    expected = hashes.get(relative)
    if expected is None:
        raise ValueError("Source is not in the immutable hash manifest: " + relative)
    actual = sha(path)
    if actual != expected:
        raise ValueError("Immutable source SHA256 mismatch: " + relative)
    observed[str(path)] = actual
    return path


def array_record(path, shape, kind):
    array = np.load(path, allow_pickle=False)
    if array.shape != shape:
        raise ValueError("Actual array shape mismatch: " + str(path))
    if kind == "rgb" and array.dtype != np.uint8:
        raise ValueError("Actual RGB must be uint8: " + str(path))
    if kind == "numeric" and array.dtype.kind not in "fiu":
        raise ValueError("Actual numeric array must be finite real measurements: " + str(path))
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite actual array: " + str(path))
    return array, dict(shape=list(array.shape), dtype=str(array.dtype),
        canonical_array_sha256=canonical_array_sha(array))


def time_record(record, block_observation, pool_metadata, query_metadata):
    """Separate captured local order from absent absolute/runtime PRE binding."""
    length = record.get("executed_length")
    steps = record.get("step_indices")
    local = (type(length) is int and length == 16 and steps == list(range(17)) and
        all(type(step) is int for step in steps) and
        block_observation.get("executed_length") == length and
        block_observation.get("only_first_block") is True)
    if not local:
        raise ValueError("Recorded first-block local execution indices do not align")
    # Merely discovering a field does not certify it against a live recipient.
    absolute_names = ("policy_step_t", "capture_policy_steps", "actual_observation_policy_step",
                      "decision_policy_step", "use_policy_step", "runtime_episode_id")
    records = {"record": record, "block_observation": block_observation,
               "pool_metadata": pool_metadata, "query_metadata": query_metadata}
    explicit = {name: {key: value[key] for key in absolute_names if key in value}
                for name, value in records.items()}
    missing = ["independently_bound_recipient_SourceSnapshot", "per_frame_actual_SourceSnapshot",
               "PRE_current_entity_anchor_hash_binding"]
    if not any("capture_policy_steps" in value for value in records.values()):
        missing.append("captured_absolute_policy_steps")
    if not any("runtime_episode_id" in value for value in records.values()):
        missing.append("captured_runtime_episode_identity")
    if not any("decision_policy_step" in value or "use_policy_step" in value for value in records.values()):
        missing.append("captured_recipient_decision_use_time")
    return dict(local_step_indices=steps, local_transition_indices=list(range(16)),
        local_temporal_order_verified=True, captured_absolute_fields=explicit,
        query_block_diagnostic=pool_metadata.get("block"),
        absolute_policy_steps_synthesized=False,
        block_number_multiplied_by_chunk_length=False,
        captured_absolute_runtime_time_verified=False,
        runtime_PRE_binding_verified=False, missing_runtime_fields=missing)


def run(root):
    root = Path(root).resolve()
    source = root / "outputs" / DERIVED_SOURCE
    original = root / "outputs" / ORIGINAL_SOURCE
    observed = {}
    observed[str(Path(__file__).resolve())] = sha(__file__)
    hash_payload = Path(__file__).resolve().with_name("current_observation_evidence.py")
    if hash_payload.is_file():
        observed[str(hash_payload)] = sha(hash_payload)
    source_manifest_path = source / "SHA256SUMS.txt"
    hashes = manifest(source_manifest_path)
    observed[str(source_manifest_path)] = sha(source_manifest_path)
    records = read(pinned(source, "records.json", hashes, observed))
    protocol = read(pinned(source, "protocol.json", hashes, observed))
    original_array_hashes = read(pinned(source, "source_sha256.json", hashes, observed))
    if (protocol.get("source") != ORIGINAL_SOURCE or
            protocol.get("role") != "offline_supervision_only" or len(records) != 160):
        raise ValueError("Only the frozen existing 160 offline windows are in scope")
    old_hash_path = original / "derived_sha256.json"
    old_hashes = read(old_hash_path)
    observed[str(old_hash_path)] = sha(old_hash_path)
    original_protocol = read(pinned(original, "protocol.json", old_hashes, observed))
    pool_rows = read(pinned(original, "pools.json", old_hashes, observed))
    if (len(pool_rows) != 40 or original_protocol.get("candidate_first_blocks") != 160 or
            original_protocol.get("pools") != 40 or original_protocol.get("k") != 4):
        raise ValueError("Immutable original membership is not forty complete K4 pools")
    pools = {}
    for pool in pool_rows:
        if pool["pool_id"] in pools:
            raise ValueError("Duplicate original pool")
        pools[pool["pool_id"]] = pool
    seen, groups, rows = set(), {}, []
    verification_counts = Counter()
    missing_counts = Counter()
    for item in records:
        pool_id, cid, split = item["pool_id"], item["candidate_id"], item["split"]
        if (type(cid) is not int or cid not in range(4) or split not in ("train", "val") or
                (pool_id, cid) in seen or pool_id not in pools):
            raise ValueError("Duplicate, incomplete or unrecognized source membership")
        seen.add((pool_id, cid))
        pool = pools[pool_id]
        members = {candidate["candidate_id"]: candidate for candidate in pool["candidates"]}
        if set(members) != set(range(4)) or members[cid]["split"] != split or pool["split"] != split:
            raise ValueError("Original pool candidate/split identity mismatch")
        derived_relative = item["directory"] + "/temporal_teacher.json"
        teacher = read(pinned(source, derived_relative, hashes, observed))
        expected_folder = str(Path("outputs") / ORIGINAL_SOURCE / members[cid]["directory"])
        # Normalize separators only, never strip a source/dataset scope.
        if teacher.get("source_folder", "").replace("\\", "/") != expected_folder.replace("\\", "/"):
            raise ValueError("Derived source_folder does not identify this original candidate")
        folder = inside(root, teacher["source_folder"])
        expected_identity = dict(dataset=ORIGINAL_SOURCE, suite="libero90", task=pool["task"],
            state=pool["state"], candidate_id=cid, observed_block_id=pool_id + "/first_block")
        if (teacher.get("identity") != expected_identity or teacher.get("split") != split or
                teacher.get("role") != "offline_supervision_only"):
            raise ValueError("Derived source identity or teacher role mismatch")
        local_path = str(folder.relative_to(original)).replace("\\", "/")
        raw = read(pinned(original, local_path + "/record.json", old_hashes, observed))
        block = read(pinned(original, local_path + "/block_observation.json", old_hashes, observed))
        if (raw.get("identity") != expected_identity or raw.get("split") != split or
                raw.get("role") != "paired_recovery_supervision"):
            raise ValueError("Original capture identity or role mismatch")
        group = (expected_identity["suite"], pool["task"], pool["state"])
        if groups.setdefault(group, split) != split:
            raise ValueError("Original task/state group leaks between train and val")
        files, loaded = {}, {}
        for name, (filename, shape, kind) in ARRAYS.items():
            if raw["arrays"].get(name) != filename:
                raise ValueError("Original array reference differs from frozen actual schema")
            path = pinned(original, local_path + "/" + filename, old_hashes, observed)
            parent_reference = str(path.relative_to(root)).replace("\\", "/")
            expected_parent = original_array_hashes.get(parent_reference)
            if expected_parent is not None:
                if sha(path) != expected_parent:
                    raise ValueError("Derived original-source pin disagrees with actual array")
                verification_counts["derived_parent_array_hash_checks"] += 1
            loaded[name], summary = array_record(path, shape, kind)
            files[name] = summary | dict(path=parent_reference,
                file_sha256=observed[str(path)], original_manifest_verified=True,
                derived_parent_pin_available=expected_parent is not None)
        query_source = pool.get("query_source")
        query_metadata, query_pool = {}, {}
        query_files = {}
        # Keep query metadata diagnostic if its immutable capture binding is
        # unavailable; do not replace missing timestamps with seed or block.
        for key, rel in (("pool", "pool.json"), ("candidate", f"candidate_{cid}/query.json")):
            relative = query_source + "/" + rel if isinstance(query_source, str) else None
            if relative is None:
                query_files[key] = dict(present=False, immutable_hash_verified=False)
                continue
            path = inside(root, relative)
            if not path.is_file():
                query_files[key] = dict(path=relative, present=False, immutable_hash_verified=False)
                continue
            digest = sha(path)
            expected = pool.get("source_sha256", {}).get(relative, original_array_hashes.get(relative))
            is_pinned = expected is not None
            if is_pinned and digest != expected:
                raise ValueError("Frozen query reference SHA256 mismatch")
            value = read(path)
            observed[str(path)] = digest
            query_files[key] = dict(path=relative, present=True, file_sha256=digest,
                immutable_hash_verified=is_pinned, recorded_keys=sorted(value))
            if key == "pool":
                query_pool = value
            else:
                query_metadata = value
        timing = time_record(raw, block, query_pool, query_metadata)
        missing_counts.update(timing["missing_runtime_fields"])
        bindings = protocol.get("bindings", {}).get(str(pool["task"]))
        rows.append(dict(pool_id=pool_id, candidate_id=cid, split=split,
            source_identity=expected_identity,
            source_folder=teacher["source_folder"], source_role="offline_calibration_source",
            original_record_role=raw["role"], teacher_role=teacher["role"],
            bindings_recorded_in_frozen_protocol=bindings,
            observed_instance_identity_certified=False,
            array_records=files, timing=timing, query_files=query_files,
            per_frame_canonical_actual_hashes=[dict(local_step=step,
                primary_sha256=canonical_array_sha(loaded["primary_sequence"][step]),
                wrist_sha256=canonical_array_sha(loaded["wrist_sequence"][step]),
                proprio_sha256=canonical_array_sha(loaded["proprio_sequence"][step]))
                for step in range(17)],
            proprio_alias_diagnostic=dict(
                start_exact_equal=bool(np.array_equal(loaded["proprio_sequence"][0], loaded["proprio_start"])),
                end_exact_equal=bool(np.array_equal(loaded["proprio_sequence"][-1], loaded["proprio_end"]))),
            provenance_binding_is_physical_certificate=False,
            runtime_window_constructed=False, physical_fact_certified=False,
            PRE_root192_input_admitted=False, supervision_labels_read=False,
            physical_teacher_values_used=False))
        verification_counts["original_array_hash_checks"] += len(ARRAYS)
    if len(seen) != 160 or any({cid for p, cid in seen if p == pool} != set(range(4)) for pool in pools):
        raise ValueError("All original 160 candidates must remain; no selective filtering")
    splits = Counter(row["split"] for row in rows)
    if splits != {"train": 80, "val": 80}:
        raise ValueError("Existing calibration membership must remain train80/val80")
    return dict(schema=SCHEMA, scope="existing_160_offline_calibration_windows_only",
        structural_integrity_passed=True, candidates=160, pools=40, local_frames=2720,
        split_candidates=dict(splits), task_state_group_leakage=False,
        verification_counts=dict(verification_counts),
        absolute_runtime_PRE_binding_passed=False,
        missing_runtime_field_candidate_counts=dict(missing_counts),
        actual_runtime_windows_constructed=0, physical_TRUE_facts_created=0,
        source_or_labels_modified=False, original192_PRE_inputs_modified=False,
        new_queries=0, new_actions=0, new_fits=0, detectors_registered=0,
        observed_source_sha256=observed, rows=rows,
        limitations=[
            "Relative indices0..16 are recorded local order, not captured absolute policy timestamps.",
            "Query block/seed are diagnostic; no block-times-chunk-length timestamp synthesis.",
            "Offline AFTER-execution windows may calibrate an actual-only observer, but are not original192 PRE X.",
            "File/array hashes and frozen entity names do not certify current grasp, lift or instance tracking.",
            "Missing live recipient SourceSnapshot/time binding is reported, not fabricated.",
            "Teacher JSON identity/source references are read; physical values and labels are not used.",
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    # When uploaded alone, import frozen actual-array hashing from the repo.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    result = run(root)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    output = args.output.resolve()
    if (output.exists() or not output.is_relative_to(root / "outputs") or
            output.is_relative_to(root / "outputs" / DERIVED_SOURCE) or
            output.is_relative_to(root / "outputs" / ORIGINAL_SOURCE)):
        raise ValueError("Only a fresh separate derived output directory is allowed")
    output.mkdir(parents=True)
    (output / "actual_window_source_audit.json").write_text(payload, encoding="utf8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "observed_source_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
