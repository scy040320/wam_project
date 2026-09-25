# Public release boundaries

The active direction is paired recovery-treatment learning for a frozen WAM.
`recovery_decision/` releases lightweight manifest, alignment, stage-selection and
descriptive-analysis utilities, with unit tests. It does not yet release a portable
simulator collector, trained selector, online detector or end-to-end benchmark.

`cfwam/`, existing `configs/` and existing `scripts/` retain the earlier attribution
prototype and baseline tooling. Their interfaces are preserved; their metrics and
time-alignment assumptions must not be transferred to the new protocol. See
[historical notes](LEGACY_ATTRIBUTION.md) and the earlier protocol documents.

Excluded: credentials, personal reports, daily plans, private orchestration,
absolute cloud launch paths, checkpoints, observations, videos, datasets, caches
and third-party assets. New public files use semantic names, not daily task IDs.

No project-wide license has been selected. Do not advertise a fully licensed
open-source release or redistribute upstream code/weights without permission.
