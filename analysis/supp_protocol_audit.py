"""Supplementary protocol audit: E1 Holm recomputation and P1 family-metric scoring audit.

This script is a post-hoc verification pass over already-released aggregate metrics.
It does not retrain anything and does not touch site-level data. It produces two
records that back the corrections made to the manuscript:

1. ``supp_holm_e1_20row_record.csv`` -- Holm-Bonferroni correction applied to the
   full 20-comparison E1 edge-ablation family. The E1 summary released with the
   repository reports uncorrected bootstrap p-values only; the manuscript
   previously implied the surviving effects were Holm-corrected. This file makes
   the correction explicit and reproducible.

2. ``supp_p1_family_metric_protocol_audit.csv`` -- per-run audit of the Week-11 P1
   family metric-learning experiment, recording which scoring function each arm
   actually used and what the AUC becomes under the scoring direction the metric
   head is actually optimised for. The A1 arm is scored by predicted illegal
   probability while every alpha>0 arm is scored by cosine similarity to seen-family
   centroids, so the arms are not directly comparable.

Both outputs are diagnostic (``is_posthoc=True``) and are not used to revise any
pre-registered hypothesis outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

E1_SUMMARY = ROOT / "results" / "week8" / "metrics" / "E1_edge_ablation_summary.csv"
P1_LOFO = ROOT / "results" / "week11" / "metrics" / "P1_family_metric_lofo.csv"
P1_ABLATION = ROOT / "results" / "week11" / "metrics" / "P1_alpha_ablation.csv"
OUT_DIR = ROOT / "reports" / "supplementary_exact_significance"

ALPHA = 0.05


def holm_correct(p_values: list[float], alpha: float = ALPHA) -> pd.DataFrame:
    """Apply Holm-Bonferroni step-down correction to a family of raw p-values.

    Returns one row per input comparison (original order preserved) carrying the
    Holm rank, the step-down threshold ``alpha / (m - k + 1)``, the reject decision
    under the step-down stopping rule, and the monotone adjusted p-value.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    rank = [0] * m
    threshold = [0.0] * m
    reject = [False] * m
    adjusted = [0.0] * m
    still_rejecting = True
    running_max = 0.0
    for k, idx in enumerate(order, start=1):
        thr = alpha / (m - k + 1)
        if still_rejecting and p_values[idx] > thr:
            still_rejecting = False
        running_max = max(running_max, min(1.0, (m - k + 1) * p_values[idx]))
        rank[idx] = k
        threshold[idx] = thr
        reject[idx] = still_rejecting
        adjusted[idx] = running_max
    return pd.DataFrame(
        {
            "holm_rank": rank,
            "holm_threshold": threshold,
            "holm_reject": reject,
            "p_holm_adjusted": adjusted,
        }
    )


def build_e1_holm_record() -> pd.DataFrame:
    """Recompute Holm correction over the full 20-row E1 edge-ablation family."""
    frame = pd.read_csv(E1_SUMMARY)
    if len(frame) != 20:
        raise ValueError(f"Expected 20 E1 comparisons, found {len(frame)}")
    corrected = holm_correct(frame["p_value"].astype(float).tolist())
    record = pd.concat([frame.reset_index(drop=True), corrected], axis=1)
    record["family_size"] = len(frame)
    record["alpha"] = ALPHA
    record["sign_consistent_across_seeds"] = (
        (record["effect_ci_lo"].astype(float) > 0) | (record["effect_ci_hi"].astype(float) < 0)
    )
    # The bootstrap p-value is a percentile inversion, so it collapses to 0 whenever
    # all five seed-level paired differences share a sign, independent of effect size.
    record["p_is_degenerate_zero"] = record["p_value"].astype(float) <= 0.0
    record["is_posthoc"] = True
    return record.sort_values("holm_rank").reset_index(drop=True)


def build_p1_protocol_audit() -> pd.DataFrame:
    """Audit the Week-11 P1 arms: which scoring function ran, and the corrected direction."""
    parts = []
    for path, source in [(P1_LOFO, "P1_family_metric_lofo"), (P1_ABLATION, "P1_alpha_ablation")]:
        part = pd.read_csv(path)
        part["source_file"] = source
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)

    keep = [
        "source_file",
        "family_id",
        "seed",
        "method",
        "alpha",
        "binary_weight",
        "shuffle_family_labels",
        "score_mode",
        "hard_negative_auc",
        "positive_score_mean",
        "negative_score_mean",
        "score_gap_pos_minus_neg",
        "heldout_positive_count",
        "negative_count",
    ]
    audit = frame[keep].copy()
    audit = audit.rename(columns={"hard_negative_auc": "auc_as_reported"})

    # Reported label: held-out (unseen) family = positive. The metric arms score by
    # similarity to centroids built from the *seen* families, so a higher score means
    # "more like a seen family" -- the opposite of the reported positive class. The
    # corrected-direction column reads the same scores against the label the metric is
    # actually optimised for (seen-family membership).
    audit["scoring_matches_label_direction"] = audit["score_mode"].eq("binary")
    audit["auc_seen_family_direction"] = audit.apply(
        lambda r: float(r["auc_as_reported"])
        if r["score_mode"] == "binary"
        else 1.0 - float(r["auc_as_reported"]),
        axis=1,
    )

    # Week-11 prepare_tensors() drops only the held-out family from the training pool,
    # so the sampled hard negatives (other Denmark illegal families) remain in training
    # and are also the rows that define the centroids. The Week-8.5 LOFO flow
    # (multi_scenario_eval.build_e3_w85_split) removed them from the training pool.
    audit["test_negatives_excluded_from_training"] = False
    audit["centroids_built_from_test_negative_families"] = audit["score_mode"].eq("metric")
    audit["is_posthoc"] = True
    return audit


def main() -> None:
    """Write both supplementary audit records."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    e1 = build_e1_holm_record()
    e1_path = OUT_DIR / "supp_holm_e1_20row_record.csv"
    e1.to_csv(e1_path, index=False, encoding="utf-8")
    n_reject = int(e1["holm_reject"].sum())
    print(f"[E1] wrote {e1_path.relative_to(ROOT)}  ({len(e1)} comparisons, {n_reject} survive Holm)")
    for _, row in e1[e1["p_value"].astype(float) > 0].head(3).iterrows():
        print(
            f"      {row['transfer_id']} x {row['edge_removed']}: "
            f"p={row['p_value']:.3f} rank={row['holm_rank']} "
            f"thr={row['holm_threshold']:.5f} reject={row['holm_reject']} "
            f"p_adj={row['p_holm_adjusted']:.4f}"
        )

    p1 = build_p1_protocol_audit()
    p1_path = OUT_DIR / "supp_p1_family_metric_protocol_audit.csv"
    p1.to_csv(p1_path, index=False, encoding="utf-8")
    print(f"[P1] wrote {p1_path.relative_to(ROOT)}  ({len(p1)} runs)")
    grouped = p1.groupby(["method", "score_mode"])[
        ["auc_as_reported", "auc_seen_family_direction"]
    ].mean()
    print(grouped.round(4).to_string())


if __name__ == "__main__":
    main()
