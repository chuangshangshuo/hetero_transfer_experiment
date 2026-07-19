"""RQ4: few-shot calibration evaluation and rollups."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_transfer import build_transfer_split
from src.train.utils import (
    compute_sample_counts,
    load_graph_bundle,
    make_common_manifest,
    record_run_manifest,
    save_dataframe,
    summarise_runs,
)
from src.transfer.few_shot_finetune import run_fewshot_finetune


def redirect_output(bundle: Any, root_name: str) -> None:
    """Redirect output."""
    workspace = Path(bundle.config["workspace_root"])
    for key in list(bundle.output_paths.keys()):
        subdir = key if key != "root" else ""
        path = workspace / "output" / root_name / subdir
        path.mkdir(parents=True, exist_ok=True)
        bundle.output_paths[key] = path
        bundle.config["output"][key] = str(Path("output") / root_name / subdir) if subdir else str(Path("output") / root_name)


def load_week7_split(bundle: Any, transfer_id: str, seed: int) -> pd.DataFrame:
    """Load week7 split."""
    path = Path(bundle.config["workspace_root"]) / "output" / "week7" / "splits" / f"{transfer_id}__seed{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    split_frame, _audit = build_transfer_split(bundle, transfer_id, seed)
    return split_frame


def eligible_transfers(bundle: Any) -> list[str]:
    """Eligible transfers."""
    configured = list(bundle.config.get("week9_fewshot", {}).get("transfers", []))
    if configured:
        return configured
    return ["T2_PH", "T3_DiagnoseFrance"]


def plot_fewshot(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot few-shot."""
    if summary.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for transfer_id, part in summary.groupby("transfer_id"):
        part = part.sort_values("shots_per_class")
        axis.plot(
            part["shots_per_class"],
            part["target_test_auc_mean"],
            marker="o",
            label=transfer_id,
        )
    axis.set_xlabel("Target shots per class")
    axis.set_ylabel("Target ROC-AUC")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Week 9 Few-Shot Target Adaptation")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_acceptance(raw: pd.DataFrame) -> pd.DataFrame:
    """Build acceptance."""
    rows = []
    for transfer_id, part in raw.groupby("transfer_id"):
        best = part.groupby("shots_per_class")["target_test_auc"].mean().sort_values(ascending=False)
        source_path = ROOT / "output" / "week7" / "metrics" / "transfer_method_summary.csv"
        source_auc = float("nan")
        if source_path.exists():
            w7 = pd.read_csv(source_path)
            source = w7[(w7["transfer_id"] == transfer_id) & (w7["method"] == "source_only")]
            if not source.empty and "target_test_auc_mean" in source.columns:
                source_auc = float(source["target_test_auc_mean"].iloc[0])
        best_shot = int(best.index[0])
        best_auc = float(best.iloc[0])
        rows.append(
            {
                "experiment": "Week9_fewshot",
                "transfer_id": transfer_id,
                "best_shots_per_class": best_shot,
                "best_fewshot_auc": best_auc,
                "week7_source_only_auc": source_auc,
                "delta_vs_source_only": best_auc - source_auc if pd.notna(source_auc) else float("nan"),
                "status": "posthoc_support" if pd.notna(source_auc) and best_auc >= source_auc + 0.02 else "no_clear_gain",
                "is_posthoc": True,
            }
        )
    return pd.DataFrame(rows)


def run_week9_fewshot(bundle: Any, smoke_test: bool = False) -> None:
    """Run week9 few-shot."""
    redirect_output(bundle, "week9_smoke" if smoke_test else "week9")
    bundle.config.setdefault("week9_fewshot", {})
    bundle.config["week9_fewshot"].setdefault("target_loss_weight", 1.0)
    transfers = eligible_transfers(bundle)
    shots = list(bundle.config["week9_fewshot"].get("shots_per_class", [1, 3, 5]))
    seeds = [int(seed) for seed in bundle.config["seeds"]]
    if smoke_test:
        transfers = transfers[:1]
        shots = shots[:1]
        seeds = seeds[:1]
    rows: list[dict[str, Any]] = []
    for transfer_id in transfers:
        for seed in seeds:
            split_frame = load_week7_split(bundle, transfer_id, seed)
            if split_frame[split_frame["split"] == "target_adapt_unlabeled"]["label"].nunique() < 2:
                continue
            for shot in shots:
                metrics, predictions, history = run_fewshot_finetune(
                    bundle=bundle,
                    graph_data=bundle.graph_data.cpu(),
                    split_frame=split_frame,
                    seed=seed,
                    shots_per_class=int(shot),
                    smoke_test=smoke_test,
                )
                suffix = f"fewshot_{transfer_id}__k{int(shot)}__seed{seed}"
                save_dataframe(predictions, bundle.output_paths["predictions"] / f"{suffix}_predictions.csv")
                save_dataframe(history, bundle.output_paths["logs"] / f"{suffix}_history.csv")
                manifest = make_common_manifest(
                    bundle=bundle,
                    task=f"week9_fewshot_{transfer_id}",
                    model="fewshot_finetune",
                    split_name=f"{transfer_id}__seed{seed}.csv",
                    sample_counts=compute_sample_counts(predictions),
                    seed=seed,
                )
                manifest["shots_per_class"] = int(shot)
                manifest["is_posthoc"] = True
                manifest["metrics"] = metrics
                record_run_manifest(bundle.output_paths["runs"] / f"{suffix}.json", manifest)
                rows.append(metrics)
                save_dataframe(pd.DataFrame(rows), bundle.output_paths["metrics"] / "fewshot_raw_runs.csv")
    raw = pd.DataFrame(rows)
    save_dataframe(raw, bundle.output_paths["metrics"] / "fewshot_raw_runs.csv")
    summary = summarise_runs(
        raw,
        ["transfer_id", "shots_per_class"],
        ["target_test_auc", "target_pr_auc", "target_balanced_acc", "target_macro_f1"],
    ) if not raw.empty else pd.DataFrame()
    save_dataframe(summary, bundle.output_paths["metrics"] / "fewshot_summary.csv")
    acceptance = build_acceptance(raw) if not raw.empty else pd.DataFrame()
    save_dataframe(acceptance, bundle.output_paths["metrics"] / "fewshot_acceptance_audit.csv")
    plot_fewshot(summary, bundle.output_paths["plots"] / "fewshot_auc_by_k.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Week 9 few-shot RQ4 target adaptation.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    run_week9_fewshot(bundle, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
