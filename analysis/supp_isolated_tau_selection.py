"""Isolated hyper-parameter selection: choose the HeCo temperature with source-domain labels only.

The manuscript discloses that tau was selected by the pooled-primary validation ROC-AUC, and that
this validation split overlaps the transfer target_test splits. The remaining question the review
raises is whether that access changed anything. This script answers it directly for the seeds whose
temperature-grid encoders exist: for every transfer scenario it runs source_only under each
available temperature, selects tau by the *source_val* AUC alone -- a split that never leaves the
source domain -- and compares the resulting target metric against the tau=0.7 encoder the paper
actually used.

Outputs a per-(scenario, temperature) table plus the isolated-selection summary.
"""
import os
import sys
import time
from pathlib import Path

WS = Path(r"D:\codex\hetero_transfer_experiment")
sys.path.insert(0, str(WS))
os.chdir(WS)

import pandas as pd
import torch

import src.train.train_transfer as tt
from src.train.utils import load_graph_bundle

CONFIG = WS / "configs" / "week7_transfer.yaml"
GRID_DIR = WS / "output" / "week5" / "checkpoints"
OUT = Path(r"C:\Users\12092\AppData\Local\Temp\claude\D--hetero-transfer-experiment-release-hetero-transfer-experiment-release\4c0663aa-3a95-41dc-89a8-14a55e27d15d\scratchpad\m04b_out")
TEMPERATURES = [0.1, 0.3, 0.5, 0.7]
PAPER_TAU = 0.7


def available_seeds() -> list[int]:
    """Seeds for which every temperature in the grid has a stored encoder."""
    seeds = set()
    for path in GRID_DIR.glob("heco_encoder__tau*__seed*.pt"):
        seeds.add(int(path.stem.split("seed")[-1]))
    return sorted(s for s in seeds
                  if all((GRID_DIR / f"heco_encoder__tau{t}__seed{s}.pt").exists() for t in TEMPERATURES))


def main() -> None:
    """Run source_only under every temperature and select by source_val alone."""
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_graph_bundle(str(CONFIG))
    for name in list(bundle.output_paths):
        p = OUT / ("root" if name == "root" else name)
        p.mkdir(parents=True, exist_ok=True)
        bundle.output_paths[name] = p

    seeds = available_seeds()
    print(f"温度网格编码器齐备的种子：{seeds}")
    transfer_ids = list(bundle.config["transfer_pairs"].keys())

    rows = []
    total = len(transfer_ids) * len(seeds) * len(TEMPERATURES)
    done = 0
    for transfer_id in transfer_ids:
        for seed in seeds:
            split_frame, _ = tt.build_transfer_split(bundle, transfer_id, seed)
            for tau in TEMPERATURES:
                ckpt = GRID_DIR / f"heco_encoder__tau{tau}__seed{seed}.pt"
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                t0 = time.time()
                metrics, _pred, _h, _p, _s = tt.run_single_experiment(
                    bundle=bundle, split_frame=split_frame, encoder_state=state,
                    seed=seed, transfer_id=transfer_id, method="source_only", smoke_test=False,
                )
                rows.append({
                    "transfer_id": transfer_id, "seed": seed, "temperature": tau,
                    "source_val_auc": float(metrics["source_val_auc"]),
                    "primary_metric_value": float(metrics["primary_metric_value"]),
                    "target_test_auc": metrics.get("target_test_auc"),
                    "target_mean_pred_illegal": metrics.get("target_mean_pred_illegal"),
                    "selection_labels": "source_val_only",
                    "runtime_s": round(time.time() - t0, 1), "is_posthoc": True,
                })
                done += 1
                print(f"[{done:>2}/{total}] {transfer_id:<20}seed{seed} tau={tau} "
                      f"source_val={metrics['source_val_auc']:.4f} "
                      f"primary={metrics['primary_metric_value']:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "m04b_isolated_tau_grid.csv", index=False, encoding="utf-8")

    print("\n按源域验证集选优（不接触任何目标域或合并池标签）：")
    summary = []
    for (transfer_id, seed), g in frame.groupby(["transfer_id", "seed"]):
        best = g.loc[g["source_val_auc"].idxmax()]
        paper = g[g["temperature"] == PAPER_TAU].iloc[0]
        summary.append({
            "transfer_id": transfer_id, "seed": seed,
            "tau_selected_by_source_val": float(best["temperature"]),
            "tau_used_in_paper": PAPER_TAU,
            "same_choice": bool(float(best["temperature"]) == PAPER_TAU),
            "primary_at_selected_tau": float(best["primary_metric_value"]),
            "primary_at_paper_tau": float(paper["primary_metric_value"]),
            "delta": float(best["primary_metric_value"]) - float(paper["primary_metric_value"]),
            "source_val_auc_spread": float(g["source_val_auc"].max() - g["source_val_auc"].min()),
            "primary_spread_across_tau": float(g["primary_metric_value"].max() - g["primary_metric_value"].min()),
        })
        s = summary[-1]
        print(f"  {transfer_id:<20}seed{seed}  源域选优τ={s['tau_selected_by_source_val']}  "
              f"论文τ={PAPER_TAU}  一致={s['same_choice']}  "
              f"主指标 {s['primary_at_selected_tau']:.4f} vs {s['primary_at_paper_tau']:.4f} "
              f"(Δ={s['delta']:+.4f})  四温度间主指标极差={s['primary_spread_across_tau']:.4f}")
    pd.DataFrame(summary).to_csv(OUT / "m04b_isolated_selection_summary.csv", index=False, encoding="utf-8")
    print(f"\n写出 {OUT / 'm04b_isolated_selection_summary.csv'}")


if __name__ == "__main__":
    main()
