# Data protocols

## D18 attribution data

The current development model combines non-overlapping, audited legacy data
with the 4,800-branch expansion described in
`configs/d18_expansion_4800_protocol.json`. The expansion contains 4,200
formal train/validation samples and 600 paired `clean_b` QC branches.

The frozen observability contract routes pixel-equivalent physical changes to
`evidence_insufficient` and masks unsupported physical-factor supervision.
Raw records remain immutable; exclusions and schema migrations are stored as
separate audit manifests.

## D21 candidate-ranking data

The formal protocol is `configs/d21_formal_896_protocol.json`:

- 14 LIBERO-90 tasks;
- 8 frozen states per task;
- 2 action-block moments;
- clean, visual occlusion, object shift, and execution/contact deviation;
- 896 candidate pools and `K=4`, giving 3,584 executed candidate outcomes;
- states 0–5 train and states 6–7 validation within every task;
- grouping by `(task, state)` prevents leakage.

Each candidate record contains deployment-time attribution and belief inputs,
parsed candidate effects, official value, and post-execution success, steps,
WAM calls, latency, and risk proxies. The post-execution fields are targets and
metrics only.

Before the full dataset can run, a 112-pool pilot spanning all 14 tasks must
pass the frozen coverage and integrity gates. A failed pilot stops the queue;
the protocol does not lower thresholds or substitute tasks after outcomes are
known.
