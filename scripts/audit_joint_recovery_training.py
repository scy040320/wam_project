"""Audit a new whole-run recovery bundle; never fits or collects anything."""
import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys


def load_gate(library_root, expected_sha=None):
    """Load one explicitly scoped package; never fall back to a frozen root."""
    root = Path(library_root).resolve()
    expected = root / "wam_reranking" / "joint_recovery_contract.py"
    if not expected.is_file():
        raise ValueError("Scoped whole-run contract module missing")
    actual_sha = sha256(expected.read_bytes()).hexdigest()
    if expected_sha is not None and expected_sha != actual_sha:
        raise ValueError("Frozen scoped contract source hash differs")
    sys.path.insert(0, str(root))
    module = importlib.import_module("wam_reranking.joint_recovery_contract")
    if Path(module.__file__).resolve() != expected:
        raise ValueError("Contract imported from a different package namespace")
    return module, actual_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-root", type=Path,
        help="Explicit package root, normally new immutable CODE/lib in cloud")
    parser.add_argument("--expected-module-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Independent new audit output required; no overwrite")
    code_root = Path(__file__).resolve().parents[1]
    library_root = args.library_root or (code_root / "lib" if (code_root / "lib").is_dir() else code_root)
    gate, module_sha = load_gate(library_root, args.expected_module_sha256)
    before = gate.file_sha(args.input)
    report = gate.audit_whole_training(json.loads(args.input.read_text(encoding="utf-8")), base_path=args.input.parent)
    if gate.file_sha(args.input) != before:
        raise ValueError("Source bundle changed during audit")
    report["source_bundle_file_sha256"] = before
    report["whole_run_contract_module_path"] = str(Path(gate.__file__).resolve())
    report["whole_run_contract_module_sha256"] = module_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "training_ready", "active_heads", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
