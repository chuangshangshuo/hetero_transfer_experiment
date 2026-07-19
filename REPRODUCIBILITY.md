# Reproducibility Notes

This GitHub-ready folder is a cleaned release package. It keeps the code,
configs, aggregate results, figures, and final reports, but omits raw site-level
data, graph binaries, checkpoints, and sample-level prediction files. Below,
`<workspace_root>` refers to the local working directory where the canonical
experiment was originally executed.

## Environment

The original local environment used a conda environment named
`hetero-transfer-v2`. Environment/config files are kept under:

```text
configs/
```

Useful files:

- `configs/environment_pyg_cpu.yml`
- `configs/environment_pyg_cpu_export.yml`
- `configs/environment_pyg_cuda118.yml`
- `configs/requirements-week3.txt`
- `configs/requirements-week3.lock.txt`

## Audit Commands

The following need only the released aggregate CSVs (no private data):

```powershell
python -m pytest tests/                       # 32 unit tests over pure-logic modules
python analysis/supp_exact_significance.py    # exact n=5 significance (sign-flip permutation)
```

When the private data/checkpoints are available, run from the canonical
workspace root:

```powershell
python analysis/final_artifact_audit.py
python -m compileall -q src analysis
```

Week10/Week11 aggregation commands:

```powershell
python analysis/week10_final_rq_tables.py
python analysis/week10_ablation_all.py
python analysis/week10_fewshot_final.py
python analysis/week10_acceptance_audit.py
python src/train/train_family_metric.py
python src/train/train_invariance_methods.py
python src/train/train_week11_fewshot.py
python analysis/week11_significance.py
python analysis/week11_acceptance.py
python analysis/week11_paper_updates.py
```

DOCX report generation:

```powershell
python analysis/build_week11_docx_report.py
```

GitHub release packaging:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_github_release.ps1
```

## Claim Audit

The most important audit files are:

- `results/week10/metrics/final_rq_evidence_table.csv`
- `results/week10/metrics/final_hypothesis_rollup.csv`
- `results/week11/metrics/week11_acceptance_audit.csv`
- `results/week11/metrics/week11_score_estimate.csv`
- `results/final_artifact_audit.csv`

These files are the preferred source of truth for paper-safe claims.
