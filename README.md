# CF-WAM: Counterfactual Mismatch Attribution for World Action Models

> **Research code — Gate 1 prerelease.** This repository is under active development; it is not yet a paper reproduction package or a trained-method release.

CF-WAM studies a practical failure mode of world action models (WAMs): a difference between an imagined future and the subsequent observation does not, by itself, identify *why* the difference occurred. It may arise from a visual occlusion, an object displacement, action-execution noise, or an unknown event. A uniform “stop and replan” response can therefore be unnecessarily costly or unsafe.

This repository provides an external, dependency-aware attribution layer for a frozen WAM. Its first adapter uses [Cosmos Policy](https://github.com/nvlabs/cosmos-policy) with LIBERO. Cosmos Policy remains the action/world model; CF-WAM does **not** modify or redistribute Cosmos Policy, LIBERO, or their checkpoints.

## What CF-WAM does

At a fixed four-action control interval, the collector records:

1. current primary/wrist observations and proprioception;
2. frozen-WAM planned actions, imagined future images, and value;
3. executed actions and action-execution difference;
4. real primary/wrist observations after four actions; and
5. an online belief/provenance graph snapshot.

Simulator truth is written only as an **offline** cause and affected-node label for training and evaluation. It is never an online input. The intended causes are `normal`, `visual_occlusion`, `object_shift`, `action_noise`, and `unknown`.

The lightweight two-layer graph-attention prototype predicts a cause, an affected-node mask, and per-cause residual explanations. The recovery router then applies a transparent rule table. “Local rollback” means invalidating the affected **belief/provenance** subgraph and changing the next decision; it never claims to undo a physical simulator action.

## Gate 1 status

| Component | Status | Scope / caveat |
| --- | --- | --- |
| Task graph, online record contract, recovery router, and unit tests | Implemented | Four reviewed LIBERO task specifications. |
| Four-step paired collection interface | Implemented | Formal 2,400-record development collection is running separately; records are intentionally not versioned here. |
| Three controlled mismatch mechanisms | Implemented | Occlusion, object shift, and action noise; `unknown` is a mixed/OOD condition. |
| Frozen DINOv2-S feature extraction | Implemented | Offline only; configured to avoid concurrent WAM/DINO GPU residency. |
| Attribution GAT and loss | Prototype implemented | No trained checkpoint or performance claim is released yet. |
| D12 binary diagnostic baseline | Implemented, pending full evaluation | This is a custom diagnostic wrapper, **not** Cosmos Policy's native control logic. |
| Object-centric online state | In progress | The included slot/ROI audit is an engineering bridge, not a general detector or segmenter. |
| A/B/C/D end-to-end comparison | Not started | Requires the frozen development set and trained/calibrated model. |

## Repository layout

```text
cfwam/                 # task graph, record schema, GAT prototype, losses, abstain and recovery logic
configs/               # reviewed task graphs and frozen development-split protocol
scripts/               # collection, feature extraction, audits, training and statistics utilities
tests/                 # graph/protocol and model smoke tests
docs/                  # scope, reproducibility boundary and publication policy
```

## Installation

CF-WAM is an add-on, not a replacement runtime. First create a working Cosmos Policy + LIBERO environment from the upstream [Cosmos Policy setup instructions](https://github.com/nvlabs/cosmos-policy). Use a Python 3.10 environment for the current adapter.

```bash
git clone https://github.com/scy040320/wam_project.git cfwam
cd cfwam
python -m pip install -r requirements-gate1.txt
export PYTHONPATH="$PWD:${COSMOS_POLICY_ROOT}:${PYTHONPATH}"
pytest -q tests/test_cfwam_core.py tests/test_model_smoke.py
```

`COSMOS_POLICY_ROOT` must point to a separately installed upstream checkout with LIBERO and access to the official model assets. Do not commit Hugging Face tokens, checkpoints, cache directories, or simulator data.

## Reproduce the non-GPU protocol checks

The manifest freezes the 4 tasks × 3 phases × 5 conditions × 40 matched initial states = 2,400 planned records, with a grouped 24/8/8 initial-state split.

```bash
python scripts/build_development_manifest.py --root . --output outputs/protocol_v1
```

The output is deliberately ignored by Git. It records configuration/code hashes and split membership for experiment tracking.

## Collect one aligned record

Run this only after the upstream Cosmos Policy + LIBERO runtime and official model assets are available:

```bash
python scripts/collect_libero_counterfactuals.py \
  --task-config configs/libero_task_0.yaml \
  --episode-id 0 --phase approach --condition normal \
  --output outputs/cfwam_v1_records
```

Before scaling up, validate a complete 15-record task-0 smoke batch:

```bash
python scripts/audit_small_batch.py \
  --root outputs/small_batch_task0_seed00 \
  --report outputs/small_batch_task0_seed00_audit.json
```

## Training and evaluation boundary

`scripts/train_attributor.py` is a prototype trainer. It enforces that a matched initial state cannot appear in both train and validation inputs, but the training-tensor builder and final evaluation suite are intentionally not represented as completed results. Thresholds for `unknown/abstain` must be selected once on validation states 24–31 and then frozen before held-out testing.

The repository includes a paired-bootstrap summary utility, but it does not choose thresholds or alter raw records.

## What is deliberately not released

- raw observations, actions, labels, videos, DINO features, CSV results, and development manifests;
- model checkpoints, tokenizer files, Hugging Face cache or authentication material;
- LIBERO assets, Cosmos Policy source/weights, and cloud/desktop-specific launch scripts;
- personal plans, reports, screenshots, and unpublished claims.

These exclusions keep the repository reproducible without exposing credentials, third-party artifacts, or incomplete experimental evidence. Release of a dataset, checkpoints, final metrics, and a project license will accompany a later paper-ready version.

## Acknowledgements

This adapter is built around the public [Cosmos Policy](https://github.com/nvlabs/cosmos-policy) interface and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark. Please follow their respective licenses, access requirements, and citation instructions.
