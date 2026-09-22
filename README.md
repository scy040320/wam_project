# CF-WAM: Counterfactual Mismatch Attribution for World Action Models

> **Research code — Gate 1 prerelease.** D30 online qualification has passed; the D31 controlled comparison has not started. This is not yet a paper reproduction package or a trained-model release.

CF-WAM studies a practical failure mode of world action models (WAMs): a difference between an imagined future and the subsequent observation does not, by itself, identify *why* the difference occurred. It may arise from a visual occlusion, an object displacement, action-execution noise, or an unknown event. A uniform “stop and replan” response can therefore be unnecessarily costly or unsafe.

This repository provides an external, dependency-aware attribution layer for a frozen WAM. Its first adapter uses [Cosmos Policy](https://github.com/nvlabs/cosmos-policy) with LIBERO. Cosmos Policy remains the action/world model; CF-WAM does **not** modify or redistribute Cosmos Policy, LIBERO, or their checkpoints.

## Current status

- The counterfactual attribution model, dependency graph, recovery router, and `unknown/abstain` safety exit are implemented and unit-tested.
- Development attribution and localization results are complete. The corrected native-16 online control qualification passed its frozen D30 gate.
- A September 2026 protocol audit found that the released LIBERO checkpoint is trained for 16-action chunks. The online runner now preserves that native horizon and aligns predicted futures only with real observations at `t+16`.
- Method v1 is frozen after validation-only calibration, a 20-episode smoke test, and a 100-episode held-out qualification. D31 remains pending and no broader paper-level comparison claim is made here.

See [`docs/D30_PROTOCOL_CORRECTION.md`](docs/D30_PROTOCOL_CORRECTION.md) for the audit evidence, invalidated assumptions, frozen V6 rule, and D30 gate outcome.

## What CF-WAM does

For the corrected online controller, Cosmos decodes and executes its native 16-action chunk. At each aligned decision boundary, the runner records:

1. current primary/wrist observations and proprioception;
2. frozen-WAM planned actions, imagined future images, and value;
3. executed actions and action-execution difference;
4. real primary/wrist observations after 16 actions; and
5. an online belief/provenance graph snapshot.

Simulator truth is written only as an **offline** cause and affected-node label for training and evaluation. It is never an online input. The intended causes are `normal`, `visual_occlusion`, `object_shift`, `action_noise`, and `unknown`.

The lightweight two-layer graph-attention prototype predicts a cause, an affected-node mask, and per-cause residual explanations. The recovery router then applies a transparent rule table. “Local rollback” means invalidating the affected **belief/provenance** subgraph and changing the next decision; it never claims to undo a physical simulator action.

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
pytest -q tests/test_cfwam_core.py tests/test_model_smoke.py tests/test_d30_native16_calibration.py
```

`COSMOS_POLICY_ROOT` must point to a separately installed upstream checkout with LIBERO and access to the official model assets. Do not commit Hugging Face tokens, checkpoints, cache directories, or simulator data.

## Reproduce the non-GPU protocol checks

The development manifest freezes 4 tasks × 3 phases × 5 conditions × 40 matched initial states = 2,400 planned records, with a grouped 24/8/8 initial-state split. This v1 dataset is retained for attribution development; online control results are separately qualified under the native-16 protocol.

```bash
python scripts/build_development_manifest.py --root . --output outputs/protocol_v1
```

The output is deliberately ignored by Git. It records configuration/code hashes and split membership for experiment tracking.

## Collect one aligned development record

Run this only after the upstream Cosmos Policy + LIBERO runtime and official model assets are available:

```bash
python scripts/collect_libero_counterfactuals.py \
  --task-config configs/libero_task_0.yaml \
  --episode-id 0 --phase approach --condition normal \
  --output outputs/cfwam_v1_records
```

For the controlled action-execution diagnostic, retain the intervention scale in the record name and offline metadata:

```bash
python scripts/collect_libero_counterfactuals.py \
  --task-config configs/libero_task_0.yaml \
  --episode-id 0 --phase approach --condition action_noise \
  --noise-scale 0.5 --output outputs/d09_noise_strength_audit_v1
```

Before scaling up, validate a complete 15-record task-0 smoke batch:

```bash
python scripts/audit_small_batch.py \
  --root outputs/small_batch_task0_seed00 \
  --report outputs/small_batch_task0_seed00_audit.json
```

## Training and evaluation boundary

`scripts/train_attributor.py` is a prototype trainer. It enforces that a matched initial state cannot appear in both train and validation inputs. Thresholds for `unknown/abstain` must be selected once on validation states 24–31 and then frozen before held-out testing.

The repository includes a paired-bootstrap summary utility, but it does not choose thresholds or alter raw records.

## Diagnostic and online-repair runners

The D12 full-episode diagnostic uses a frozen image-MAE threshold to record the explicit custom decision `continue` or `global_refresh`. This is a measurement baseline rather than a claim about Cosmos Policy's native controller.

`scripts/run_d26_online_local_repair.py` is retained as the first four-step integration diagnostic. It must not be used for final control claims with the released 16-step checkpoint.

`scripts/run_d30_temporal_guard.py` is the corrected online runner. It keeps the frozen WAM at its trained 16-action horizon, writes strict `t+16` prediction–reality pairs, and applies the validation-frozen V6 guard. An isolated model-unknown signal first produces a guarded re-observation and requires a second consecutive confirmation before safe stop. Immediate safe stop is reserved for unknown evidence supported by action-execution evidence. Use only thresholds frozen on a separate validation set. `unknown/abstain` is a safety stop, not a successful recovery.

## D30 qualification snapshot

The frozen V6 configuration used 935 continuous online rows from task 0/1 validation only. Task 2/3 held-out results were not used to select thresholds.

| Check | Result | Gate |
|---|---:|---:|
| Full episodes completed | 100/100 | 100/100 |
| Known-intervention recovery | 59/60 (98.3%) | at least 90% |
| Unknown safe stop | 20/20 | at least 18/20 |
| Later false safety stop on known conditions | 2/60 (3.3%) | at most 20% |
| Clean false safety stop | 0/20 | 0 |
| Task 2 clean success, method / A-clean | 10/10 / 10/10 | decline at most 1/10 |
| Task 3 clean success, method / A-clean | 9/10 / 9/10 | decline at most 1/10 |

All required decision JSON files and both camera videos were non-empty. No OOM, dead loop, or runtime-fatal traceback occurred. Per-episode outputs and checkpoints remain excluded from Git.

## What is deliberately not released

- raw observations, actions, labels, videos, DINO features, CSV results, and development manifests;
- model checkpoints, tokenizer files, Hugging Face cache or authentication material;
- LIBERO assets, Cosmos Policy source/weights, and cloud/desktop-specific launch scripts;
- personal plans, reports, screenshots, and unpublished claims.

These exclusions keep the repository reproducible without exposing credentials, third-party artifacts, or large experimental artifacts. A paper-ready release will add the final comparison package, dataset/checkpoint policy, and project license.

## Acknowledgements

This adapter is built around the public [Cosmos Policy](https://github.com/nvlabs/cosmos-policy) interface and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark. Please follow their respective licenses, access requirements, and citation instructions.
