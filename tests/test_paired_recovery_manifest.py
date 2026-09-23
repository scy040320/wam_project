import importlib.util
from collections import Counter
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_paired_recovery_comparison.py"
    spec = importlib.util.spec_from_file_location("paired_recovery_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_is_paired_and_leakage_tagged():
    module = _module()
    rows = module.build_manifest()
    assert len(rows) == 320
    assert sum(r["analysis_set"] == "primary_heldout" for r in rows) == 288
    assert sum(r["analysis_set"] == "supplemental_validation_overlap" for r in rows) == 32
    assert all(int(r["seed"]) >= 32 for r in rows if r["analysis_set"] == "primary_heldout")
    assert Counter(r["approach"] for r in rows) == Counter({m: 80 for m in module.METHODS})
    cells = {}
    for row in rows:
        cells.setdefault(row["cell_id"], []).append(row)
    assert len(cells) == 80
    assert all({r["approach"] for r in cell} == set(module.METHODS) for cell in cells.values())


def test_method_order_is_balanced():
    module = _module()
    rows = module.build_manifest()
    first = Counter(r["approach"] for r in rows if r["order_slot"] == "0")
    assert first == Counter({m: 20 for m in module.METHODS})
