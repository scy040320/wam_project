# Gate 1 release scope

This is a source-only prerelease for the CF-WAM research pipeline.

The public interface is intentionally narrow:

- `cfwam/` contains platform-neutral task-graph, attribution, abstention and recovery code.
- `configs/` contains human-reviewed semantic task graphs and a pre-registered development split.
- `scripts/` contains code to collect, audit, featurize and summarize records when the user provides a licensed upstream Cosmos Policy + LIBERO installation.

The repository does not publish an experimental conclusion. In particular, the GAT is a prototype implementation; it has no released trained weights, calibrated threshold file, or benchmark score. The custom binary diagnostic script exists to illustrate a stated baseline policy and is not evidence about an original Cosmos Policy decision rule.

Raw outputs remain private until their licenses, privacy, storage, and reproducibility conditions have been reviewed. A project license must be chosen before advertising a later paper-ready release as fully open source.
