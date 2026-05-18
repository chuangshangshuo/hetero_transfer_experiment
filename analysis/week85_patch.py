from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.multi_scenario_eval import extract_frozen_embeddings, redirect_output_root
from src.train.train_transfer import add_transfer_groups, best_group_split, evaluate_binary, load_encoder_state
from src.train.utils import build_primary_task_frame, choose_threshold_by_youden, load_graph_bundle, save_dataframe
from src.transfer.logme_score import compute_logme
from src.transfer.transferability_analysis import compute_h_divergence, compute_sliced_wasserstein


def valid_source_regions(task_frame: pd.DataFrame, min_per_class: int = 2) -> list[str]:
    out = []
    for region, part in task_frame.groupby("jurisdiction"):
        counts = part["label"].value_counts()
        if counts.get(0, 0) >= min_per_class and counts.get(1, 0) >= min_per_class:
            out.append(str(region))
    return sorted(out)


def target_regions(task_frame: pd.DataFrame, min_rows: int = 2) -> list[str]:
    out = []
    for region, part in task_frame.groupby("jurisdiction"):
        if len(part) >= min_rows:
            out.append(str(region))
    return sorted(out)


def split_source(source: pd.DataFrame, seed: int, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = add_transfer_groups(source.reset_index(drop=True), list(config["splits"]["transfer"]["group_columns"]))
    train_idx, val_idx = best_group_split(
        grouped,
        test_fraction=float(config["splits"]["transfer"]["source_val_fraction"]),
        seed=seed,
        attempts=int(config["splits"]["transfer"].get("balance_attempts", 30)),
        require_two_classes_when_available=True,
    )
    return grouped.iloc[train_idx].copy(), grouped.iloc[val_idx].copy()


def target_primary_metric(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[str, float, dict[str, float]]:
    metrics = evaluate_binary(y_true, y_score, threshold)
    unique = np.unique(y_true)
    if unique.size == 2:
        return "roc_auc", metrics["roc_auc"], metrics
    if unique.size == 1 and int(unique[0]) == 1:
        return "illegal_recall_at_youden", metrics["illegal_consistency"], metrics
    if unique.size == 1 and int(unique[0]) == 0:
        return "one_minus_mean_pred_illegal", 1.0 - metrics["mean_pred_illegal"], metrics
    return "undefined", float("nan"), metrics


def run_e2_w85(config_path: Path) -> pd.DataFrame:
    bundle = load_graph_bundle(config_path)
    redirect_output_root(bundle, "week85")
    task_frame = build_primary_task_frame(bundle)
    sources = valid_source_regions(task_frame, min_per_class=2)
    targets = target_regions(task_frame, min_rows=2)
    rows: list[dict[str, Any]] = []
    for seed in [int(seed) for seed in bundle.config["seeds"]]:
        encoder_state, encoder_checkpoint = load_encoder_state(bundle, seed)
        embedding = extract_frozen_embeddings(bundle, encoder_state)
        for source_region in sources:
            source = task_frame[task_frame["jurisdiction"] == source_region].copy()
            source_train, source_val = split_source(source, seed, bundle.config)
            x_train = embedding[source_train["graph_node_index"].to_numpy(dtype=int)]
            y_train = source_train["label"].to_numpy(dtype=int)
            x_val = embedding[source_val["graph_node_index"].to_numpy(dtype=int)]
            y_val = source_val["label"].to_numpy(dtype=int)
            clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
            clf.fit(x_train, y_train)
            val_scores = clf.predict_proba(x_val)[:, 1]
            threshold = choose_threshold_by_youden(y_val, val_scores)
            for target_region in targets:
                if target_region == source_region:
                    continue
                target = task_frame[task_frame["jurisdiction"] == target_region].copy()
                if target.empty:
                    continue
                source_idx = source["graph_node_index"].to_numpy(dtype=int)
                target_idx = target["graph_node_index"].to_numpy(dtype=int)
                x_source = embedding[source_idx]
                y_source = source["label"].to_numpy(dtype=int)
                x_target = embedding[target_idx]
                y_target = target["label"].to_numpy(dtype=int)
                target_scores = clf.predict_proba(x_target)[:, 1]
                metric_name, metric_value, target_metrics = target_primary_metric(y_target, target_scores, threshold)
                rows.append(
                    {
                        "seed": seed,
                        "source_region": source_region,
                        "target_region": target_region,
                        "source_size": int(len(source)),
                        "target_size": int(len(target)),
                        "target_label_class_count": int(np.unique(y_target).size),
                        "target_metric_name": metric_name,
                        "target_quasi_metric": metric_value,
                        "target_roc_auc": target_metrics["roc_auc"],
                        "target_illegal_recall_at_youden": target_metrics["illegal_consistency"],
                        "target_mean_pred_illegal": target_metrics["mean_pred_illegal"],
                        "target_one_minus_mean_pred_illegal": 1.0 - target_metrics["mean_pred_illegal"],
                        "threshold": threshold,
                        "logme_source": compute_logme(x_source, y_source),
                        "logme_cross": compute_logme(np.vstack([x_source, x_target]), np.concatenate([y_source, y_target])),
                        "h_divergence": compute_h_divergence(x_source, x_target, seed=seed),
                        "wasserstein": compute_sliced_wasserstein(x_source, x_target, seed=seed),
                        "encoder_checkpoint": encoder_checkpoint,
                    }
                )
    frame = pd.DataFrame(rows)
    save_dataframe(frame, bundle.output_paths["metrics"] / "E2_w85_expanded_pair_scores.csv")
    metric_rows = []
    aligned = frame[np.isfinite(frame["target_quasi_metric"])].copy()
    for metric, invert in [
        ("logme_cross", False),
        ("h_divergence", True),
        ("wasserstein", True),
    ]:
        values = -aligned[metric] if invert else aligned[metric]
        rho, rho_p = spearmanr(values, aligned["target_quasi_metric"], nan_policy="omit")
        tau, tau_p = kendalltau(values, aligned["target_quasi_metric"], nan_policy="omit")
        metric_rows.append(
            {
                "metric": metric,
                "aligned_rows": int(len(aligned)),
                "ordered_pairs": int(aligned[["source_region", "target_region"]].drop_duplicates().shape[0]),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "kendall_tau": float(tau),
                "kendall_p": float(tau_p),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    save_dataframe(metrics, bundle.output_paths["metrics"] / "E2_w85_transferability_metrics.csv")
    plot_e2_w85(aligned, bundle.output_paths["plots"] / "E2_w85_logme_vs_quasi_metric.png")
    return frame


def plot_e2_w85(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 5))
    for metric_name, part in frame.groupby("target_metric_name"):
        axis.scatter(part["logme_cross"], part["target_quasi_metric"], s=28, alpha=0.72, label=metric_name)
    axis.set_xlabel("LogME cross score")
    axis.set_ylabel("Target quasi metric")
    axis.set_title("W8.5 E2 Expanded Transferability")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def build_w85_acceptance(config_path: Path) -> pd.DataFrame:
    bundle = load_graph_bundle(config_path)
    redirect_output_root(bundle, "week85")
    rows: list[dict[str, Any]] = []
    e3 = pd.read_csv(bundle.output_paths["metrics"] / "E3_w85_lofo_summary.csv")
    balanced = e3[e3["test_form"] == "balanced_licensed_negative"]
    hard = e3[e3["test_form"] == "hard_illegal_negative"]
    balanced_min = float(balanced["test_roc_auc_mean"].min())
    hard_mean = float(hard["test_roc_auc_mean_mean"].mean()) if "test_roc_auc_mean_mean" in hard.columns else float(hard["test_roc_auc_mean"].mean())
    hard_tsars = hard[hard["family_id"] == "tsars"]
    hard_tsars_auc = safe_float(hard_tsars["test_roc_auc_mean"].iloc[0]) if not hard_tsars.empty else float("nan")
    rows.append(
        {
            "patch": "P1",
            "hypothesis": "E3_balanced_licensed_negative",
            "status": "pass" if balanced_min >= 0.85 else "fail",
            "observed": balanced_min,
            "threshold": "min family AUC >= 0.85",
            "interpretation": "illegal-vs-licensed generalization remains strong under balanced test negatives",
        }
    )
    rows.append(
        {
            "patch": "P1",
            "hypothesis": "E3_hard_illegal_family_detection",
            "status": "fail" if hard_mean < 0.80 else "pass",
            "observed": hard_mean,
            "threshold": "mean hard-negative family AUC >= 0.80",
            "interpretation": "hard-negative audit measures family-level discrimination among illegal sites",
        }
    )
    rows.append(
        {
            "patch": "P1",
            "hypothesis": "E3_hard_tsars_family_detection",
            "status": "fail" if hard_tsars_auc < 0.80 else "pass",
            "observed": hard_tsars_auc,
            "threshold": "tsars hard-negative AUC >= 0.80",
            "interpretation": "critical tsars audit under hard illegal negatives",
        }
    )

    e2 = pd.read_csv(bundle.output_paths["metrics"] / "E2_w85_transferability_metrics.csv").set_index("metric")
    rho_logme = safe_float(e2.loc["logme_cross", "spearman_rho"])
    rho_h = abs(safe_float(e2.loc["h_divergence", "spearman_rho"]))
    rho_w = abs(safe_float(e2.loc["wasserstein", "spearman_rho"]))
    rows.append(
        {
            "patch": "P2",
            "hypothesis": "E2_expanded_LogME_predicts_quasi_metric",
            "status": "pass" if rho_logme >= 0.60 else "fail",
            "observed": rho_logme,
            "threshold": "Spearman rho >= 0.60",
            "interpretation": "expanded one-class-aware quasi metric, no GNN retraining",
        }
    )
    rows.append(
        {
            "patch": "P2",
            "hypothesis": "E2_expanded_LogME_beats_distance_baselines",
            "status": "pass" if abs(rho_logme) > rho_h and abs(rho_logme) > rho_w else "fail",
            "observed": abs(rho_logme),
            "threshold": "|rho_LogME| > |rho_H-div| and |rho_W|",
            "interpretation": f"|rho_H-div|={rho_h:.4f}, |rho_W|={rho_w:.4f}",
        }
    )

    week8_e4 = pd.read_csv(ROOT / "output" / "week8" / "metrics" / "E4_embedding_distances.csv")
    e4b_prime = int((week8_e4["d_illegal_control"] > week8_e4["d_illegal_licensed"]).sum())
    e4c_prime = int((week8_e4["d_licensed_control"] > week8_e4["d_illegal_licensed"]).sum())
    rows.append(
        {
            "patch": "P3_posthoc",
            "hypothesis": "H_E4b_prime_direction_only",
            "status": "pass" if e4b_prime >= 4 else "fail",
            "observed": e4b_prime,
            "threshold": ">=4/5 seeds",
            "interpretation": "post-hoc direction-only relaxation; not a prerun Week 8 pass",
        }
    )
    rows.append(
        {
            "patch": "P3_posthoc",
            "hypothesis": "H_E4c_prime_direction_only",
            "status": "pass" if e4c_prime >= 4 else "fail",
            "observed": e4c_prime,
            "threshold": ">=4/5 seeds",
            "interpretation": "post-hoc direction-only relaxation; not a prerun Week 8 pass",
        }
    )

    week8_e5 = pd.read_csv(ROOT / "output" / "week8" / "metrics" / "E5_structure_vs_lexical_summary.csv")
    t3 = week8_e5[week8_e5["transfer_id"] == "T3_DiagnoseFrance"].set_index("config_id")
    t2ph = week8_e5[week8_e5["transfer_id"] == "T2_PH"].set_index("config_id")
    t2on = week8_e5[week8_e5["transfer_id"] == "T2_ON"].set_index("config_id")
    e5_drop = safe_float(t3.loc["C1_full", "primary_metric_value_mean"]) - safe_float(t3.loc["C2_no_cctld", "primary_metric_value_mean"])
    t3_lex = safe_float(t3.loc["C5_lex_only", "primary_metric_value_mean"])
    t3_graph = safe_float(t3.loc["C4_graph_only", "primary_metric_value_mean"])
    t2ph_graph = safe_float(t2ph.loc["C4_graph_only", "primary_metric_value_mean"])
    t2ph_lex = safe_float(t2ph.loc["C5_lex_only", "primary_metric_value_mean"])
    t2on_graph = safe_float(t2on.loc["C4_graph_only", "primary_metric_value_mean"])
    t2on_lex = safe_float(t2on.loc["C5_lex_only", "primary_metric_value_mean"])
    rows.extend(
        [
            {
                "patch": "P4_posthoc",
                "hypothesis": "H_E5a_prime_cctld_key",
                "status": "pass" if e5_drop >= 0.05 else "fail",
                "observed": e5_drop,
                "threshold": "T3 full-minus-no-cctld >= 0.05",
                "interpretation": "post-hoc direction reversal: ccTLD is a key mechanism",
            },
            {
                "patch": "P4_posthoc",
                "hypothesis": "H_E5b_prime_lex_only_working",
                "status": "pass" if t3_lex >= 0.85 else "fail",
                "observed": t3_lex,
                "threshold": "T3 lex-only >= 0.85",
                "interpretation": "post-hoc direction reversal: lexical features work independently on T3",
            },
            {
                "patch": "P4_posthoc",
                "hypothesis": "H_E5c_prime_transfer_specific_dominance",
                "status": "pass" if (t3_lex > t3_graph and t2ph_graph > t2ph_lex and t2on_graph > t2on_lex) else "fail",
                "observed": float(sum([t3_lex > t3_graph, t2ph_graph > t2ph_lex, t2on_graph > t2on_lex])),
                "threshold": "3/3 specified directional checks",
                "interpretation": "post-hoc transfer-specific mechanism split",
            },
        ]
    )

    week8_e1 = pd.read_csv(ROOT / "output" / "week8" / "metrics" / "E1_edge_ablation_summary.csv")
    t2ph_e1 = week8_e1[week8_e1["transfer_id"] == "T2_PH"]
    negative_effect = t2ph_e1[(t2ph_e1["effect_mean"] < 0) & (t2ph_e1["p_value"] < 0.05)]
    t2on_e1 = week8_e1[week8_e1["transfer_id"] == "T2_ON"]
    t2on_sig = t2on_e1[t2on_e1["p_value"] < 0.05]
    rows.append(
        {
            "patch": "P4_posthoc",
            "hypothesis": "H_E1b_prime_T2PH_negative_transfer_edge",
            "status": "pass" if not negative_effect.empty else "fail",
            "observed": float(negative_effect["effect_mean"].min()) if not negative_effect.empty else float("nan"),
            "threshold": "at least one edge has effect<0 and p<0.05",
            "interpretation": "post-hoc direction reversal for PH channel-specific negative transfer",
        }
    )
    rows.append(
        {
            "patch": "P4_posthoc",
            "hypothesis": "H_E1c_prime_T2ON_any_significant_edge",
            "status": "pass" if not t2on_sig.empty else "fail",
            "observed": int(len(t2on_sig)),
            "threshold": "at least one edge p<0.05",
            "interpretation": "post-hoc significant influence criterion for ON one-class target",
        }
    )

    audit = pd.DataFrame(rows)
    save_dataframe(audit, bundle.output_paths["metrics"] / "week85_acceptance_audit.csv")
    write_w85_draft(bundle, audit)
    return audit


def write_w85_draft(bundle: Any, audit: pd.DataFrame) -> None:
    out = ROOT / "paper" / "draft" / "sec4_4_week85_patch.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = audit["status"].value_counts().to_dict()
    lines = [
        "# Week 8.5 Patch Audit",
        "",
        "Week 8.5 is a post-hoc patch audit. It does not overwrite the prerun Week 8 hypothesis table.",
        f"Patch outcomes: pass={counts.get('pass', 0)}, fail={counts.get('fail', 0)}.",
        "",
        "## Acceptance Table",
        "",
        "| patch | hypothesis | observed | threshold | status |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for _, row in audit.iterrows():
        observed = row["observed"]
        observed_text = f"{float(observed):.4f}" if isinstance(observed, (int, float)) and math.isfinite(float(observed)) else str(observed)
        lines.append(
            f"| {row['patch']} | {row['hypothesis']} | {observed_text} | {row['threshold']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The original Week 8 E3 LOFO is downgraded because it mainly tested illegal-vs-licensed discrimination.",
            "- The corrected balanced licensed-negative LOFO remains strong, but the hard illegal-negative audit shows weak family-level discrimination for several families.",
            "- P3/P4 are explicitly post-hoc direction changes and should be described as revised interpretation rather than prerun confirmations.",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config_path = ROOT / "configs" / "week8.yaml"
    run_e2_w85(config_path)
    audit = build_w85_acceptance(config_path)
    print(audit[["patch", "hypothesis", "observed", "threshold", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
