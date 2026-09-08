# Reproducibility Checklist

This checklist records what can be reproduced from the current workspace and
what should be treated as saved evidence rather than a fresh claim.

## Environment

- Run from the repository root: `<workspace_root>`.
- Recommended conda environment: `hetero-transfer-v2`.
- Required runtime stack: Python 3.10, PyTorch 2.5.1, PyG 2.6.1, pandas,
  scikit-learn, numpy, matplotlib.
- Full Week 6-10 reruns require saved HeCo encoder checkpoints under
  `output/week5/checkpoints/` or `output/week6/checkpoints/`.
- MaxMind GeoLite2-ASN and other local-only data dependencies are not
  redistributed in this repository.

## Saved Result Packages

- Graph v2 data and audits:
  - `data/graphs/hetero_graph_v2.pt`
  - `data/graphs/hetero_graph_v2.npz`
  - `data/graphs/hetero_graph_v2_metadata.json`
  - `data/audits/*.csv`
- Week 4 baselines: `output/week4/`
- Week 5 HeCo pretraining/probes: `output/week5/`
- Week 6 fine-tuning/RQ1: `output/week6/`
- Week 7 transfer matrix: `output/week7/`
- Week 8 preregistered mechanism layer: `output/week8/`
- Week 8.5 correction layer: `output/week85/`
- Corrected E5 post-hoc diagnostics: `output/week8_patch/`
- Week 6 head-ablation diagnostics: `output/week6_head_ablation/` — this is where the
  reported main result lives (`mlp_64`, pooled AUC 0.971 ± 0.016). A bare
  `train_full.py` run uses the frozen `finetune.head_type: attention_pool_mlp_64`
  from `configs/week6_heco_finetune.yaml` and reproduces the 0.932 ± 0.072 ablation
  arm instead; pass `--head-ablation` for the six-head comparison.
- Week 9 few-shot and hard-negative audits: `output/week9/`
- Week 10 final RQ rollup: `output/week10/`

## Rerunnable Scripts

Use relative paths from the repository root:

```powershell
python src\data\sanity_check.py
python src\data\build_hetero_graph.py
python src\data\convert_npz_to_pyg.py
python src\train\train_single.py
python src\train\train_contrastive.py
python src\train\train_full.py
python src\train\train_full.py --head-ablation
python src\train\train_transfer.py --methods source_only
python src\train\train_transfer.py --methods dann
python src\train\train_transfer.py --methods strurw dann_strurw
python analysis\week8_acceptance.py
python analysis\week85_patch.py
python analysis\week8_patch_corrected_e5.py
python analysis\week9_fewshot_calibration.py --rebuild-from-artifacts
python analysis\week9_hard_negative_family_audit.py
python analysis\week10_final_rq_tables.py
python analysis\feature_bucket_transfer.py
python analysis\week10_ablation_all.py
python analysis\week10_fewshot_final.py
python analysis\week10_hard_negative_final.py
python analysis\week10_deviation_case.py
python analysis\week10_acceptance_audit.py
python analysis\final_artifact_audit.py
```

For the local Windows setup used during development, the same commands can be
run through:

```powershell
& "<user_home>\anaconda3\Scripts\conda.exe" run --no-capture-output -n hetero-transfer-v2 python <relative-script-path>
```

## Post-Hoc Outputs

The following outputs are correction or diagnostic evidence, not preregistered
Week 8 confirmations:

- `output/week85/metrics/week85_acceptance_audit.csv`
  - P1/P2 are correction-layer evidence.
  - P3/P4 are explicitly post-hoc interpretation revisions.
- `output/week8_patch/metrics/E5_corrected_acceptance_audit.csv`
- `output/week6_head_ablation/metrics/head_ablation_summary.csv`
- `output/week9/metrics/W9_E1_fewshot_curve.csv`
- `output/week9/metrics/week9_fewshot_rollup.csv`
- `output/week9/metrics/fewshot_acceptance_audit_v2.csv`
- `output/week9/metrics/hard_negative_family_acceptance.csv`
- `output/week10/metrics/final_hypothesis_rollup.csv` rows where
  `is_posthoc=True`

## Not Preregistered Evidence

Do not present these as preregistered Week 8 victories:

- Original Week 8 E3 family LOFO as family-level generalization evidence.
  Week 8.5 showed the original licensed-negative design mostly repeated
  illegal-vs-licensed discrimination.
- Week 8.5 P3/P4 direction-only revisions.
- Corrected E5 diagnostics under `output/week8_patch/`.
- Week 9 formal few-shot RQ4 under the W9_* tables. The older `fewshot_*` CSVs are legacy/preformal only.
- Week 9 hard-negative family audit.
- Week 6 head ablation.
- Any placeholder modules for SeHGNN, HINormer, IRM, EERM, or standalone
  StruRW entrypoints.

## Claim Boundaries

- RQ1 is mixed support: illegal-vs-licensed separation is strong, but
  illegal-family discrimination remains weak.
- RQ2 is lexical/ccTLD boundary evidence, especially for T3 France.
- RQ3 transferability is unreliable; DANN/StruRW and LogME should not be
  described as generally successful.
- RQ4 has target-dependent few-shot support: T2_PH improves strongly in the
  formal W9 rollup, while T3 is already saturated and shows no clear gain.
- One-class targets such as T2_ON must not be forced into ROC-AUC.

## Files Not Suitable for GitHub Commit

Do not commit local secrets or large artifacts:

- `.env`
- `.claude/settings.local.json`
- key files such as `*.pem`, `*.key`, `*.p12`, `*.pfx`
- local MaxMind databases
- large checkpoints: `*.pt`, `*.pth`, `*.ckpt`
- generated caches: `__pycache__/`, `.pytest_cache/`
- large generated output trees unless an explicit artifact-release policy is
  being followed

## Placeholder Modules

Several Python files exist only to reserve future module names. They now contain
top-level placeholder docstrings and must not be cited as implemented methods.
Current reported results are supported by the implemented scripts named in the
sections above.
