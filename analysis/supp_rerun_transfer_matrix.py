"""Rerun the full Week-7 transfer matrix, persisting per-sample target predictions.

The frozen research code does not save per-sample transfer predictions, so the impact of the
temperature-selection label overlap (review item M11) cannot be bounded from the released
aggregates alone. This driver reuses the frozen functions unchanged and only adds prediction
persistence, writing everything to a separate output tree so that output/week7 stays untouched.
"""
# NOTE: this script was run against the controlled research workspace, whose sample-level
# split files and model checkpoints are not part of the public release. WS and the output
# directory below are the paths used for the recorded run; set them to your own copies to
# re-execute. Only aggregate outputs are published, under reports/rerun_2026-09-08/.

import json
import os
import sys
import time
from pathlib import Path

WS = Path(r"D:\codex\hetero_transfer_experiment")
sys.path.insert(0, str(WS))
os.chdir(WS)

import pandas as pd
from src.train.train_transfer import (
    build_transfer_split,
    load_encoder_state,
    run_single_experiment,
)
from src.train.utils import load_graph_bundle

CONFIG = WS / "configs" / "week7_transfer.yaml"
SCRATCH = Path(r"<scratch_dir>")
OUT = SCRATCH / "rerun_out"
PRED = OUT / "predictions"

METHODS = ["source_only", "dann", "strurw", "dann_strurw"]


def main() -> None:
    """Run all transfer_id x seed x method combinations and persist metrics and predictions."""
    bundle = load_graph_bundle(str(CONFIG))
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    for name in list(bundle.output_paths):
        p = OUT / ("root" if name == "root" else name)
        p.mkdir(parents=True, exist_ok=True)
        bundle.output_paths[name] = p

    transfer_ids = list(bundle.config["transfer_pairs"].keys())
    seeds = [int(s) for s in bundle.config["seeds"]]
    rows = []
    total = len(transfer_ids) * len(seeds) * len(METHODS)
    done = 0
    started_all = time.time()

    for transfer_id in transfer_ids:
        for seed in seeds:
            split_frame, _ = build_transfer_split(bundle, transfer_id, seed)
            encoder_state, ckpt = load_encoder_state(bundle, seed)
            for method in METHODS:
                t0 = time.time()
                metrics, pred, _hist, _pseudo, _stru = run_single_experiment(
                    bundle=bundle, split_frame=split_frame, encoder_state=encoder_state,
                    seed=seed, transfer_id=transfer_id, method=method, smoke_test=False,
                )
                metrics["wall_runtime_s"] = float(time.time() - t0)
                metrics["encoder_checkpoint"] = Path(ckpt).name
                rows.append(metrics)
                keep = [c for c in ["node_id", "label", "split", "domain_role",
                                    "pred_prob_illegal", "pred_label"] if c in pred.columns]
                pred[keep].to_csv(
                    PRED / f"{transfer_id}__{method}__seed{seed}.csv", index=False, encoding="utf-8")
                done += 1
                print(f"[{done:>2}/{total}] {transfer_id:<20}{method:<14}seed{seed} "
                      f"primary={metrics['primary_metric_value']:.4f} "
                      f"({metrics['wall_runtime_s']:.1f}s)", flush=True)

    frame = pd.DataFrame(rows).sort_values(["transfer_id", "method", "seed"]).reset_index(drop=True)
    frame.to_csv(OUT / "rerun_transfer_summary.csv", index=False, encoding="utf-8")
    print(f"\n完成 {done} 次运行，总耗时 {time.time() - started_all:.0f}s")
    print(f"写出 {OUT / 'rerun_transfer_summary.csv'}")

    released = pd.read_csv(WS / "output" / "week7" / "metrics" / "transfer_summary.csv")
    merged = frame.merge(released, on=["transfer_id", "method", "seed"], suffixes=("_new", "_old"))
    merged["abs_diff"] = (merged["primary_metric_value_new"] - merged["primary_metric_value_old"]).abs()
    exact = int((merged["abs_diff"] < 1e-12).sum())
    print(f"\n与已发布结果逐行比对：{exact}/{len(merged)} 行完全一致，最大绝对差 {merged['abs_diff'].max():.2e}")
    merged[["transfer_id", "method", "seed", "primary_metric_value_new",
            "primary_metric_value_old", "abs_diff"]].to_csv(
        OUT / "reproduction_check.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
