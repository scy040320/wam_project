#!/usr/bin/env python3
"""Write an auditable progress snapshot for a candidate-pool queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--cells-total", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    completion = args.root / "completion.tsv"
    rows = completion.read_text().splitlines() if completion.exists() else []
    clean = sum(row.startswith("clean\t") for row in rows)
    stress = sum(row.startswith("stress\t") for row in rows)
    payload = {
        "phase": args.phase,
        "cells_total": args.cells_total,
        "clean_cells_complete": clean,
        "stress_cells_complete": stress,
        "k": args.k,
        "lanes": 3,
    }
    (args.root / "status.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
