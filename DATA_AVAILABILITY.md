# Data Availability

This repository includes public-release-safe experiment artifacts:

- graph schema and metadata;
- aggregate node/edge counts;
- aggregate experiment metrics, tables, audits, and figures;
- final report files and paper draft sections.

It intentionally does **not** include raw site-level data or files that can
expose website identities, support samples, prediction rows, or local collection
details.

## Included Data-Like Files

```text
data/node_feature_schema.json
data/edge_schema.json
data/node_stats.csv
data/edge_stats.csv
data/graphs/hetero_graph_v2_metadata.json
data/versions/*.json
data/versions/*.sha256
results/**/metrics/*.csv
results/**/tables/*.csv
results/**/plots/*
results/**/figures/*
```

## Excluded From GitHub

```text
raw site registries
processed node/edge CSVs with website-level rows
PyTorch graph/checkpoint binaries
raw prediction rows
few-shot support sample lists
split files with selected sample identifiers
training logs and temporary caches
```

## Rationale

The project studies illegal-vs-licensed website recognition and cross-region
transfer. Publishing raw website-level rows could create safety, privacy, and
misuse risks. The released package therefore focuses on auditable aggregate
evidence and code structure while keeping site-level materials out of the public
repository.

## Reproduction Note

Full numerical reruns require access to the original non-public graph inputs and
checkpoints from the canonical local workspace. Without those files, this
repository supports code review, claim audit, table/figure inspection, and
paper-writing verification, but not a complete end-to-end rerun from raw data.
