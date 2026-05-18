# Section 4.2 RQ1 Draft

Week 6 freezes the RQ1 main result on Graph v2. The final supervised label space is `licensed_baseline` versus `illegal_confirmed_official_single + illegal_confirmed_official_cross_verified`; gray candidates and legal-commercial controls are excluded from the main training label space. All reported Week 6 splits are group-aware and reuse the Week 4 grouping policy to reduce brand/operator leakage.

The strongest Week 4 graph-only baseline is HeteroGNN-only, with mean ROC-AUC `0.9277` over five seeds. After HeCo pretraining and discriminative learning-rate supervised finetuning, the Week 6 planned main model reaches mean ROC-AUC `0.9322` with standard deviation `0.0716`. This is a delta of `+0.0044` over the graph-only baseline. The uniform-LR fallback reaches mean ROC-AUC `0.9337` with standard deviation `0.0740`, a delta of `+0.0060` over the same baseline.

The Week 6 result should therefore be reported as a mixed integration result rather than a clean improvement over HeteroGNN-only. Seeds 42-45 are strong, but seed 46 is an outlier under the fixed group-aware split family; both the discriminative-LR and uniform-LR variants miss the predeclared pooled mean and variance targets. This does not invalidate the audit trail, but it means the final paper should avoid claiming that the integrated HeCo finetune is the locked best model until the variance issue is addressed.

The cross-verified evidence layer is evaluated separately because it is the core RQ1 primary-evidence tier rather than a separate model. Pooling the five test predictions and comparing `illegal_confirmed_official_cross_verified` against licensed-baseline test samples gives ROC-AUC `0.9798`. This number should be presented as the primary-layer robustness check, not as an additional training run.

Table 2 should use `output/week6/metrics/ablation_table.csv` as its source, and Figure 4 should use `figs/fig4_pooled_auc.pdf`.
