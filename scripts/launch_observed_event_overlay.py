"""Load a frozen private code package and perform ONLY an immutable label join.

The old project package, input rows, label sources and frozen models are never
overwritten. A private sys.path prefix is set before any wam_reranking import;
PYTHONPATH alone is insufficient when python stdin starts from project ROOT.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import sys

FROZEN_PREPARER_SHA = "827ca0cce24063b55ef5b83d91042c3bc809ba61132c7c462a5e8f57a9cec7ec"
FROZEN_INPUT_SHA = "5f556eed5735a038ed3254c79fc4fcbc7ba17e53ab9853a6560a23e3b6ab5816"
MODULE_NAMES = ("wam_reranking", "wam_reranking.direct_recovery", "wam_reranking.recovery_contract")


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def private_path(code, name):
    code = Path(code).resolve()
    path = (code / name).resolve()
    if Path(name).is_absolute() or code not in path.parents:
        raise ValueError("Code manifest path leaves its private namespace")
    return path


def verify_frozen_code(code):
    code = Path(code).resolve()
    hashes = json.loads((code / "frozen_parent_code_sha256.json").read_text(encoding="utf-8"))
    if hashes.get("prepare_observed_event_overlay.py") != FROZEN_PREPARER_SHA:
        raise ValueError("Frozen parent preparer fingerprint changed")
    private_names = {p.relative_to(code).as_posix() for p in (code / "lib" / "wam_reranking").glob("*.py")}
    expected_private = {name for name in hashes if name.startswith("lib/wam_reranking/") and name.endswith(".py")}
    if not private_names or private_names != expected_private:
        raise ValueError("Private package completeness differs from its frozen parent")
    for name in sorted(expected_private | {"prepare_observed_event_overlay.py"}):
        if digest(private_path(code, name)) != hashes[name]:
            raise ValueError("Frozen private code hash changed: " + name)
    # New launchers are independently frozen by the deployment manifest.
    deployed = json.loads((code / "deployed_code_sha256.json").read_text(encoding="utf-8"))
    for name in ("launch_observed_event_overlay.py", "finish_observed_event_overlay.sh", "frozen_parent_code_sha256.json"):
        if digest(private_path(code, name)) != deployed.get(name):
            raise ValueError("Namespace launcher/deployment hash changed: " + name)
    return hashes


def load_private_namespace(code):
    lib = (Path(code).resolve() / "lib").resolve()
    # Do not evict/reuse a previously imported old package. A contaminated
    # interpreter must stop rather than silently mixing immutable versions.
    for name, module in tuple(sys.modules.items()):
        if name == "wam_reranking" or name.startswith("wam_reranking."):
            origin = getattr(module, "__file__", None)
            if origin is None or lib not in Path(origin).resolve().parents:
                raise ValueError("Foreign wam_reranking already loaded: " + name)
    sys.path.insert(0, str(lib))
    importlib.invalidate_caches()
    modules = tuple(importlib.import_module(name) for name in MODULE_NAMES)
    for name, module in tuple(sys.modules.items()):
        if name == "wam_reranking" or name.startswith("wam_reranking."):
            origin = getattr(module, "__file__", None)
            if origin is None or lib not in Path(origin).resolve().parents:
                raise ValueError("Module origin escaped private namespace: " + name)
    return {m.__name__: str(Path(m.__file__).resolve()) for m in modules}


def verify_sources(source, labels, input_path):
    if digest(input_path) != FROZEN_INPUT_SHA:
        raise ValueError("Immutable canonical input fingerprint changed")
    for directory, filename in ((Path(source), "completion_audit.json"),
                                (Path(labels), "observability_label_audit.json")):
        if (directory / "failure.json").exists():
            raise ValueError("Source failure must not be automatically resumed")
        report = json.loads((directory / filename).read_text(encoding="utf-8"))
        if report.get("passed") is not True or type(report.get("completed_candidates")) is not int or report["completed_candidates"] != 160:
            raise ValueError("Full immutable-source integrity gate failed")


def run_overlay(code, source, labels, input_path, output):
    code, source, labels, input_path, output = (Path(p).resolve() for p in (code, source, labels, input_path, output))
    if output.exists():
        raise ValueError("Refuse overwrite or restart of an existing overlay")
    verify_frozen_code(code)
    verify_sources(source, labels, input_path)
    origins = load_private_namespace(code)
    print(json.dumps({"private_namespace_origins": origins, "immutable_source_preflight_passed": True}), flush=True)
    spec = importlib.util.spec_from_file_location("frozen_observed_event_overlay_preparer", code / "prepare_observed_event_overlay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.prepare(input_path, labels, output)
    if report.get("passed") is not True or report.get("training_ready") is not False or report.get("training_started") is not False:
        raise ValueError("Overlay must not promote readiness or start training")
    print(json.dumps({"overlay_passed": True, "overlay_candidates": report["overlay_candidates"],
        "inherited_rank_rows": report["inherited_rank_rows"], "inherited_auxiliary_rows": report["inherited_auxiliary_rows"],
        "training_ready": False, "training_started": False}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "source", "labels", "input", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    run_overlay(args.code, args.source, args.labels, args.input, args.output)


if __name__ == "__main__":
    main()
