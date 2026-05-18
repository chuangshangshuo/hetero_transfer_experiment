from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.significance_testing import holm_bonferroni, paired_bootstrap_pvalue


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["workspace_root"])
    paths = {name: root / rel for name, rel in config["output"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2:
        return float("nan")
    try:
        return float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


def w7_method_pairwise(config: dict[str, Any]) -> pd.DataFrame:
    p4 = config["P4_significance"]
    raw = pd.read_csv(ROOT / config["inputs"]["week7_transfer_summary"])
    rows: list[dict[str, Any]] = []
    for transfer_id, part in raw.groupby("transfer_id"):
        methods = sorted(part["method"].dropna().unique().tolist())
        for a, b in itertools.combinations(methods, 2):
            fa = part[part["method"].eq(a)].sort_values("seed")
            fb = part[part["method"].eq(b)].sort_values("seed")
            paired = fa[["seed", "primary_metric_value"]].merge(
                fb[["seed", "primary_metric_value"]],
                on="seed",
                suffixes=(f"_{a}", f"_{b}"),
            )
            av = paired[f"primary_metric_value_{a}"].to_numpy(dtype=float)
            bv = paired[f"primary_metric_value_{b}"].to_numpy(dtype=float)
            boot = paired_bootstrap_pvalue(av, bv, B=int(p4["bootstrap_B"]), seed=42)
            rows.append(
                {
                    "comparison_family": "W7_method_pairwise",
                    "transfer_id": transfer_id,
                    "method_a": a,
                    "method_b": b,
                    "n_pairs": int(len(paired)),
                    "mean_a": float(np.nanmean(av)),
                    "mean_b": float(np.nanmean(bv)),
                    "mean_delta_a_minus_b": boot["mean_delta"],
                    "ci95_low": boot["ci95_low"],
                    "ci95_high": boot["ci95_high"],
                    "bootstrap_p": boot["bootstrap_p"],
                    "wilcoxon_p": wilcoxon_p(av, bv),
                    "is_prerun_frozen": True,
                    "is_posthoc": False,
                    "paper_safe_claim": "Paired tests quantify method differences; they do not by themselves prove transfer success.",
                    "paper_forbidden_claim": "Do not claim DANN/StruRW/IRM/EERM solves transfer from an isolated p-value.",
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_reject_bootstrap"] = holm_bonferroni(frame["bootstrap_p"].to_numpy(), alpha=float(p4["alpha"]))
    frame["holm_reject_wilcoxon"] = holm_bonferroni(frame["wilcoxon_p"].to_numpy(), alpha=float(p4["alpha"]))
    return frame


def w9_fewshot_significance(config: dict[str, Any]) -> pd.DataFrame:
    p4 = config["P4_significance"]
    raw = pd.read_csv(ROOT / config["inputs"]["week9_fewshot_curve"])
    raw = raw[raw["feature_condition"].eq("full") & raw["target_id"].isin(["T2_PH", "T3_DiagnoseFrance"])].copy()
    rows: list[dict[str, Any]] = []
    for target_id, part in raw.groupby("target_id"):
        zero = part[part["shot"].astype(int).eq(0)][["seed", "fewshot_auc"]].rename(columns={"fewshot_auc": "auc_0"})
        for shot in [1, 3, 5, 10]:
            fs = part[part["shot"].astype(int).eq(shot)][["seed", "fewshot_auc"]].rename(columns={"fewshot_auc": "auc_shot"})
            paired = fs.merge(zero, on="seed", how="inner")
            av = paired["auc_shot"].to_numpy(dtype=float)
            bv = paired["auc_0"].to_numpy(dtype=float)
            boot = paired_bootstrap_pvalue(av, bv, B=int(p4["bootstrap_B"]), seed=shot * 101)
            rows.append(
                {
                    "comparison_family": "W9_fewshot_vs_0shot",
                    "target_id": target_id,
                    "shot": shot,
                    "feature_condition": "full",
                    "n_pairs": int(len(paired)),
                    "mean_shot_auc": float(np.nanmean(av)),
                    "mean_0shot_auc": float(np.nanmean(bv)),
                    "mean_delta_auc": boot["mean_delta"],
                    "ci95_low": boot["ci95_low"],
                    "ci95_high": boot["ci95_high"],
                    "bootstrap_p": boot["bootstrap_p"],
                    "wilcoxon_p": wilcoxon_p(av, bv),
                    "is_prerun_frozen": True,
                    "is_posthoc": False,
                    "paper_safe_claim": "Few-shot significance is interpreted by target and delta versus source-only.",
                    "paper_forbidden_claim": "Do not claim few-shot universally improves transfer.",
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_reject_bootstrap"] = holm_bonferroni(frame["bootstrap_p"].to_numpy(), alpha=float(p4["alpha"]))
    frame["holm_reject_wilcoxon"] = holm_bonferroni(frame["wilcoxon_p"].to_numpy(), alpha=float(p4["alpha"]))
    return frame


def overall_significance(w7: pd.DataFrame, w9: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "scope": "W7_method_pairwise",
            "num_tests": int(len(w7)),
            "holm_bootstrap_reject_count": int(w7["holm_reject_bootstrap"].sum()),
            "holm_wilcoxon_reject_count": int(w7["holm_reject_wilcoxon"].sum()),
            "is_prerun_frozen": True,
            "paper_safe_claim": "W7 method comparisons are now auditable with paired uncertainty.",
        },
        {
            "scope": "W9_fewshot_vs_0shot",
            "num_tests": int(len(w9)),
            "holm_bootstrap_reject_count": int(w9["holm_reject_bootstrap"].sum()),
            "holm_wilcoxon_reject_count": int(w9["holm_reject_wilcoxon"].sum()),
            "is_prerun_frozen": True,
            "paper_safe_claim": "W9 few-shot gains are statistically summarized without changing the target-dependent claim.",
        },
    ]
    p1 = ROOT / config["output"]["metrics"] / "P1_family_metric_lofo.csv"
    p2 = ROOT / config["output"]["metrics"] / "P2_method_comparison.csv"
    p3 = ROOT / config["output"]["metrics"] / "P3_full_fewshot_matrix.csv"
    for scope, path in [("W11_P1_family_metric", p1), ("W11_P2_invariance", p2), ("W11_P3_full_fewshot", p3)]:
        rows.append(
            {
                "scope": scope,
                "num_tests": int(len(pd.read_csv(path))) if path.exists() else 0,
                "holm_bootstrap_reject_count": np.nan,
                "holm_wilcoxon_reject_count": np.nan,
                "is_prerun_frozen": True,
                "paper_safe_claim": "Included in Week11 acceptance by preregistered effect thresholds.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config(ROOT / "configs" / "week11.yaml")
    paths = output_dirs(config)
    w7 = w7_method_pairwise(config)
    w9 = w9_fewshot_significance(config)
    overall = overall_significance(w7, w9, config)
    w7.to_csv(paths["metrics"] / "P4_w7_method_pairwise.csv", index=False, encoding="utf-8")
    w9.to_csv(paths["metrics"] / "P4_w9_fewshot_significance.csv", index=False, encoding="utf-8")
    overall.to_csv(paths["metrics"] / "P4_w11_overall_significance.csv", index=False, encoding="utf-8")
    print(f"P4 w7_tests={len(w7)} w9_tests={len(w9)}")


if __name__ == "__main__":
    main()
