"""Read-only stdout audit of historical carry/release observation sidecars."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_reranking.historical_replay_observation_contract import certify_historical_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scenario")
    args = parser.parse_args()
    manifest = json.loads((args.source/"frozen_membership.json").read_text(encoding="utf-8"))
    scenarios = [args.scenario] if args.scenario else [r["scenario"] for r in manifest["chains"]]
    certificates = [certify_historical_chain(args.root, args.source, s) for s in scenarios]
    print(json.dumps(dict(schema="historical_replay_obs_certificate_v1", certificates=certificates,
        training_ready=False, training_started=False, source_modified=False), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__": main()
