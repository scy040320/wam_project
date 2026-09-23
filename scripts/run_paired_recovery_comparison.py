#!/usr/bin/env python3
"""Run a preregistered native-16 A/B/C/D recovery comparison unattended."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # Keep manifest construction testable on non-POSIX hosts.
    fcntl = None

METHODS = ["A_binary_global", "B_uniform_subgraph", "C_attribution_global", "D_dependency_aware"]
CONDITIONS = ["normal", "visual_occlusion", "object_shift", "action_noise", "unknown"]
EXPECTED_THRESHOLD_SHA256 = "5dcac3d5e4bd6531ad52dfc1524f595096961d4cda8b8cde46a627de6c37cea6"
MIN_FREE_GIB = 30

def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_manifest() -> list[dict[str, str]]:
    """20 states/task are scheduled; seeds 30-31 are excluded from primary inference."""
    rows: list[dict[str, str]] = []
    job = cell = 0
    for task in range(4):
        for seed in range(30, 50):
            condition = CONDITIONS[(seed - 30 + task) % len(CONDITIONS)]
            shift = cell % len(METHODS)
            method_order = METHODS[shift:] + METHODS[:shift]
            analysis_set = "primary_heldout" if seed >= 32 else "supplemental_validation_overlap"
            cell_id = f"task{task}_seed{seed}_{condition}"
            for slot, method in enumerate(method_order):
                rows.append({"job_index": str(job), "cell_index": str(cell), "cell_id": cell_id,
                    "task": str(task), "seed": str(seed), "condition": condition, "approach": method,
                    "order_slot": str(slot), "analysis_set": analysis_set, "control_horizon": "16",
                    "phase": "approach", "intervention_block": "2"})
                job += 1
            cell += 1
    validate_manifest(rows)
    return rows

def validate_manifest(rows: list[dict[str, str]]) -> None:
    assert len(rows) == 320
    assert sum(r["analysis_set"] == "primary_heldout" for r in rows) == 288
    assert sum(r["analysis_set"] != "primary_heldout" for r in rows) == 32
    assert Counter(r["approach"] for r in rows) == Counter({m: 80 for m in METHODS})
    assert Counter(r["condition"] for r in rows) == Counter({c: 64 for c in CONDITIONS})
    cells: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        cells.setdefault(row["cell_id"], []).append(row)
        if row["analysis_set"] == "primary_heldout": assert int(row["seed"]) >= 32
    assert len(cells) == 80
    assert all({r["approach"] for r in rs} == set(METHODS) for rs in cells.values())
    assert Counter(rs[0]["condition"] for rs in cells.values()) == Counter({c: 16 for c in CONDITIONS})
    assert Counter(rs[0]["approach"] for rs in cells.values()) == Counter({m: 20 for m in METHODS})

def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def complete(output: Path) -> bool:
    required = [output / "online_decision_log.json", output / "full_episode_primary.mp4", output / "full_episode_wrist.mp4"]
    if not all(p.is_file() and p.stat().st_size > 0 for p in required): return False
    try:
        d = json.loads(required[0].read_text(encoding="utf-8"))
        return d.get("control_horizon") == 16 and isinstance(d.get("decisions"), list) and bool(d["decisions"])
    except (OSError, ValueError): return False

def append_progress(path: Path, row: dict[str, str], status: str, attempt: int, output: Path, note: str = "") -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        fields = ["timestamp", "status", "attempt", "job_index", "cell_id", "analysis_set", "approach", "task", "seed", "condition", "output", "note"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if new: w.writeheader()
        w.writerow({"timestamp": now(), "status": status, "attempt": attempt, "job_index": row["job_index"],
            "cell_id": row["cell_id"], "analysis_set": row["analysis_set"], "approach": row["approach"],
            "task": row["task"], "seed": row["seed"], "condition": row["condition"], "output": str(output), "note": note})
        f.flush(); os.fsync(f.fileno())

def quarantine_incomplete(output: Path) -> None:
    if not output.exists() or complete(output): return
    output.rename(output.with_name(output.name + f"_incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

def status_file(path: Path, **items: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated_at": now(), **items}, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", type=Path, default=Path.cwd()); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(); root = args.project_root.resolve()
    out_root = root / "outputs/paired_native16_recovery_v1"; run_root = out_root / "runs"; log_root = out_root / "job_logs"
    manifest = out_root / "manifest.csv"; progress = out_root / "progress.tsv"; status = out_root / "status.json"
    attention = out_root / "NEEDS_ATTENTION"; done = out_root / "DONE"; lock_path = out_root / ".orchestrator.lock"
    threshold = root / "outputs/native16_temporal_guard/native16_temporal_guard_frozen.yaml"
    checkpoint = root / "outputs/cfwam_v1_attributor_train_v1/attributor_best.pt"
    tensors = root / "outputs/cfwam_v1_training_tensors/train.pt"
    runner = root / "scripts/run_native16_temporal_guard.py"
    summary_script = root / "scripts/summarize_paired_recovery_comparison.py"; uv = shutil.which("uv")
    out_root.mkdir(parents=True, exist_ok=True); run_root.mkdir(exist_ok=True); log_root.mkdir(exist_ok=True)
    if fcntl is None and not args.dry_run:
        raise RuntimeError("paired recovery comparison requires a POSIX host with fcntl locking")
    lock = lock_path.open("w")
    try:
        if fcntl is not None: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: print("another paired-comparison orchestrator holds the lock", file=sys.stderr); return 3
    rows = build_manifest()
    if not manifest.exists(): write_manifest(rows, manifest)
    else:
        with manifest.open(newline="", encoding="utf-8") as f: existing = list(csv.DictReader(f))
        if existing != rows: raise RuntimeError("existing manifest differs from preregistered manifest")
    required = [threshold, checkpoint, tensors, runner, summary_script]
    missing = [str(p) for p in required if not p.is_file()]
    if missing or not uv:
        attention.write_text("preflight missing: " + ", ".join(missing + ([] if uv else ["uv"])), encoding="utf-8"); return 4
    threshold_hash = sha256(threshold)
    if threshold_hash != EXPECTED_THRESHOLD_SHA256:
        attention.write_text(f"threshold hash mismatch: {threshold_hash}\n", encoding="utf-8"); return 5
    free_gib = shutil.disk_usage(root).free / (1024 ** 3)
    if free_gib < MIN_FREE_GIB:
        attention.write_text(f"preflight free disk {free_gib:.1f} GiB < {MIN_FREE_GIB}\n", encoding="utf-8"); return 6
    protocol = {"protocol_version": "paired_native16_recovery_v1", "created_at": now(),
        "primary_population": "tasks 0-3, seeds 32-49; no training or validation states",
        "supplemental_population": "seeds 30-31; validation-overlap; excluded from primary inference",
        "total_jobs": 320, "primary_jobs": 288, "supplemental_jobs": 32,
        "pairing": "same task/seed/condition/intervention block/budget across A/B/C/D",
        "condition_assignment": "CONDITIONS[(seed-30+task) mod 5]", "method_order": "four-way cyclic Latin rotation by cell index",
        "threshold_sha256": threshold_hash, "checkpoint_sha256": sha256(checkpoint), "runner_sha256": sha256(runner),
        "manifest_sha256": sha256(manifest), "control_horizon": 16, "phase": "approach", "intervention_block": 2,
        "binary_mae_threshold": 13.5, "no_adaptation": "No thresholds, checkpoint, code, or recovery rules may change during the comparison."}
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run: print(json.dumps(protocol, indent=2, ensure_ascii=False)); return 0
    done.unlink(missing_ok=True); attention.unlink(missing_ok=True)
    complete_count = sum(complete(run_root / f'{r["cell_id"]}_{r["approach"]}') for r in rows); consecutive_failures = 0
    for row in rows:
        output = run_root / f'{row["cell_id"]}_{row["approach"]}'
        if complete(output): append_progress(progress, row, "skip_complete", 0, output); continue
        if shutil.disk_usage(root).free / (1024 ** 3) < MIN_FREE_GIB:
            msg = f"disk guard: free space below {MIN_FREE_GIB} GiB"; append_progress(progress, row, "halted", 0, output, msg)
            attention.write_text(msg + "\n", encoding="utf-8"); return 7
        succeeded = False
        for attempt in (1, 2):
            quarantine_incomplete(output); append_progress(progress, row, "started", attempt, output)
            status_file(status, state="running", completed=complete_count, total=len(rows), current=row, attempt=attempt)
            job_log = log_root / f'{int(row["job_index"]):03d}_{row["cell_id"]}_{row["approach"]}_attempt{attempt}.log'
            cmd = [uv, "run", "--extra", "cu128", "--group", "libero", "--python", "3.10", "python", str(runner),
                "--task-config", str(root / f'configs/libero_task_{row["task"]}.yaml'), "--checkpoint", str(checkpoint),
                "--thresholds", str(threshold), "--training-tensors", str(tensors), "--episode-id", row["seed"],
                "--phase", row["phase"], "--condition", row["condition"], "--intervention-block", row["intervention_block"],
                "--approach", row["approach"], "--dino-device", "cpu", "--output", str(output)]
            with job_log.open("w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()).returncode
            text = job_log.read_text(encoding="utf-8", errors="replace")
            if rc == 0 and complete(output):
                append_progress(progress, row, "complete", attempt, output); complete_count += 1; consecutive_failures = 0; succeeded = True; break
            oom = "out of memory" in text.lower() or "cuda error: out of memory" in text.lower()
            append_progress(progress, row, "failed", attempt, output, f"rc={rc}; complete={complete(output)}; oom={oom}")
            if oom:
                attention.write_text(f"OOM at {row}; see {job_log}\n", encoding="utf-8")
                status_file(status, state="halted_oom", completed=complete_count, total=len(rows), current=row); return 8
        if not succeeded:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                attention.write_text(f"three consecutive failed jobs; last={row}\n", encoding="utf-8")
                status_file(status, state="halted_failures", completed=complete_count, total=len(rows), current=row); return 9
        if complete_count % 10 == 0: print(f"[{now()}] paired comparison complete {complete_count}/{len(rows)}", flush=True)
    rc = subprocess.run([sys.executable, str(summary_script), "--root", str(out_root)], cwd=root).returncode
    if rc != 0: attention.write_text(f"summary failed rc={rc}\n", encoding="utf-8"); return 10
    final_complete = sum(complete(run_root / f'{r["cell_id"]}_{r["approach"]}') for r in rows)
    if final_complete != len(rows): attention.write_text(f"integrity audit {final_complete}/{len(rows)} complete\n", encoding="utf-8"); return 11
    status_file(status, state="complete", completed=final_complete, total=len(rows))
    done.write_text(f"completed_at={now()}\nmanifest_sha256={sha256(manifest)}\n", encoding="utf-8"); return 0

if __name__ == "__main__": raise SystemExit(main())
