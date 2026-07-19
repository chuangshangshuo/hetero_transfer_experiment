"""RQ1: pooled evaluation on the cross-verified (n=33) subset."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.utils import ensure_directory, save_dataframe


SEEDS = [42, 43, 44, 45, 46]


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute metrics."""
    y_true = frame["label"].astype(int)
    y_score = frame["pred_prob_illegal"].astype(float)
    threshold = float(frame["threshold"].iloc[0]) if "threshold" in frame.columns else 0.5
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    """Command-line entry point."""
    metrics_dir = ROOT / "output" / "week6" / "metrics"
    audits_dir = ROOT / "output" / "week6" / "audits"
    ensure_directory(metrics_dir)
    ensure_directory(audits_dir)
    min_cv = 3
    config_path = ROOT / "configs" / "week6_heco_finetune.yaml"
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        min_cv = int(config["acceptance_thresholds"]["cross_verified_min_per_seed"])

    rows: list[dict[str, float | int | str]] = []
    audit_frames: list[pd.DataFrame] = []
    pooled_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        pred_path = ROOT / "output" / "week6" / "predictions" / f"heco_finetune__seed{seed}.csv"
        pred = pd.read_csv(pred_path)
        test = pred[pred["split"] == "test"].copy()
        target = test[test["sample_tier"].isin(["licensed_baseline", "illegal_confirmed_official_cross_verified"])].copy()
        num_cv = int((target["sample_tier"] == "illegal_confirmed_official_cross_verified").sum())
        if num_cv < min_cv:
            raise AssertionError(f"Seed {seed} has only {num_cv} cross_verified test rows; expected >= {min_cv}")
        metric = compute_metrics(target)
        rows.append(
            {
                "scope": f"seed_{seed}",
                "seed": seed,
                "num_rows": int(len(target)),
                "num_licensed": int((target["sample_tier"] == "licensed_baseline").sum()),
                "num_cross_verified": num_cv,
                **metric,
            }
        )
        target["rq1_eval_scope"] = "cross_verified_vs_licensed_test"
        audit_frames.append(target)
        pooled_frames.append(target)

    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled_metric = compute_metrics(pooled)
    rows.append(
        {
            "scope": "pooled_5_seed",
            "seed": -1,
            "num_rows": int(len(pooled)),
            "num_licensed": int((pooled["sample_tier"] == "licensed_baseline").sum()),
            "num_cross_verified": int((pooled["sample_tier"] == "illegal_confirmed_official_cross_verified").sum()),
            **pooled_metric,
        }
    )
    save_dataframe(pd.DataFrame(rows), metrics_dir / "cross_verified_pooled_eval.csv")
    save_dataframe(pd.concat(audit_frames, ignore_index=True), audits_dir / "cross_verified_eval_prediction_rows.csv")


if __name__ == "__main__":
    main()
