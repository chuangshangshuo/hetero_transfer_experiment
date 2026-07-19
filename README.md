# Heterogeneous Graph Transfer Experiment

This repository is a cleaned GitHub release package for a heterogeneous graph
transfer experiment on illegal-vs-licensed website recognition, regional
transfer, shortcut diagnostics, and few-shot target-domain calibration.

> 中文说明：本仓库按最终实验报告整理，不是原始工作区镜像。它保留代码、配置、论文草稿、聚合指标、图表和最终报告；默认不发布站点级原始数据、预测明细、few-shot support sample 清单、checkpoint 或临时缓存。

## Recommended Reading Order

1. [`reports/week11_final/Week11_Final_Experiment_Report.md`](reports/week11_final/Week11_Final_Experiment_Report.md)
2. [`reports/Week11_Final_Experiment_Report.docx`](reports/Week11_Final_Experiment_Report.docx)
3. [`docs/evidence_boundary_table.md`](docs/evidence_boundary_table.md)
4. [`paper/draft/sec5_honest_evidence_boundary_framework.md`](paper/draft/sec5_honest_evidence_boundary_framework.md)
5. [`results/week11/metrics/week11_acceptance_audit.csv`](results/week11/metrics/week11_acceptance_audit.csv)
6. [`results/week10/metrics/final_rq_evidence_table.csv`](results/week10/metrics/final_rq_evidence_table.csv)

## Repository Layout

```text
analysis/                 Analysis and report-generation scripts
configs/                  Week-level experiment configs and environment specs
data/                     Public schema, graph metadata, and aggregate stats only
docs/                     Design notes, diagnostic interpretation, reproducibility notes
figs/                     Paper figure outputs from the workspace
paper/draft/              Paper draft sections with safe-claim wording
reports/                  Final integrated Week11 report in Markdown and DOCX
results/                  Aggregate metrics, tables, plots, audits, and final evidence tables
scripts/                  Setup and release-packaging utilities
src/                      Data, model, transfer, training, and explanation code
tests/                    Unit tests for the pure-logic modules (run without private data)
reproducibility/          Project logs copied from the canonical workspace
```

## Main Evidence Boundary

The project does **not** claim that the model fully solves cross-region illegal
website detection. The final interpretation is deliberately bounded:

- The model can separate illegal websites from licensed controls in the
  constructed benchmark.
- High source-only AUC is not treated as proof of structural generalization.
- T3 France is treated as saturated and shortcut-sensitive, not as pure
  structural transfer evidence.
- Family hard-negative evaluation remains weak; the model is not claimed to
  perform reliable illegal-family attribution.
- Few-shot target calibration is target-dependent. It strongly helps hard but
  recoverable targets such as `T2_PH`, while adding little to saturated targets.
- All post-hoc diagnostic and few-shot conclusions must keep their post-hoc
  status visible.

## Key Result Files

- Week10 final RQ tables: [`results/week10/`](results/week10/)
- Week11 patch/audit metrics: [`results/week11/metrics/`](results/week11/metrics/)
- Week11 figures: [`results/week11/plots/`](results/week11/plots/)
- Final artifact audit: [`results/final_artifact_audit.csv`](results/final_artifact_audit.csv)
- Supplementary exact significance (n=5 sign-flip permutation / sign / exact
  Wilcoxon tests over the released seed-level deltas):
  [`analysis/supp_exact_significance.py`](analysis/supp_exact_significance.py)
  with outputs under
  [`reports/supplementary_exact_significance/`](reports/supplementary_exact_significance/).
  Headline: none of the 24 W7 method pairs survives Holm correction at the
  exact n=5 resolution (two-sided floor p=0.0625).
- Release file manifest: [`MANIFEST_RELEASE.csv`](MANIFEST_RELEASE.csv)

## What Is Intentionally Excluded

This GitHub package excludes:

- raw site registry rows and processed node/edge CSVs containing site-level data;
- PyTorch graph/checkpoint files (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`);
- training logs, raw prediction rows, support sample lists, and split sample dumps;
- local temporary folders (`tmp/`, `logs/`, `runs/`, `__pycache__/`);
- credentials, tokens, local `.env` files, and machine-specific settings.

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the data-release policy.

## Reproducibility

The repository keeps the code and aggregate outputs needed to audit the claims.
Full reruns require the private/non-public raw graph inputs and local conda
environment used in the original workspace. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Unit tests under [`tests/`](tests/) cover the pure-logic modules (paired
bootstrap, Holm correction, exact permutation tests, LogME scoring, family
pattern matching) and run against the released aggregate data only:

```bash
python -m pytest tests/
```

## Release Status

This package was prepared from a local research workspace and cleaned for public
release. It keeps the code, configs, aggregate metrics, figures, and final
reports; it intentionally excludes site-level data, checkpoints, prediction
rows, support sample lists, and other operational details. See
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the full release policy.

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for
details. Note that the released artifacts are research code and aggregate
results; they are not an operational tool. See [`SECURITY.md`](SECURITY.md) for
misuse notes.
