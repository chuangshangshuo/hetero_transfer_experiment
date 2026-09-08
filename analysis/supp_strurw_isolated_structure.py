"""M03: re-run StruRW with the structure weights estimated from source_train labels only.

The frozen implementation builds the CSBM block-label vector from every row whose
``domain_role`` is ``source`` (``train_transfer.build_source_label_inputs``), which includes
``source_val``. The validation split therefore takes part in fitting the edge reweighting and is
no longer a clean model-selection set.

This script monkey-patches that single function so the label vector is restricted to
``split == "source_train"``, leaves everything else in the frozen pipeline untouched, and re-runs
the StruRW and DANN+StruRW arms on all four scenarios and five seeds. The comparison against the
released numbers bounds how much the validation labels contributed.
"""
# NOTE: this script was run against the controlled research workspace, whose sample-level
# split files and model checkpoints are not part of the public release. WS and the output
# directory below are the paths used for the recorded run; set them to your own copies to
# re-execute. Only aggregate outputs are published, under reports/rerun_2026-09-08/.

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
OUT = Path(r"<scratch_dir>\m03_out")
METHODS = ["strurw", "dann_strurw"]

_ORIGINAL = tt.build_source_label_inputs


def build_source_label_inputs_train_only(bundle, split_frame):
    """Same as the frozen version but restricted to the source_train split."""
    labels = torch.full((len(bundle.website_frame),), -1, dtype=torch.long)
    source_mask = torch.zeros((len(bundle.website_frame),), dtype=torch.bool)
    source_frame = split_frame[split_frame["split"] == "source_train"]
    for _, row in source_frame.iterrows():
        node_idx = int(row["graph_node_index"])
        labels[node_idx] = int(row["label"])
        source_mask[node_idx] = True
    return labels, source_mask


def main() -> None:
    """Re-run the StruRW family with isolated structure estimation."""
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_graph_bundle(str(CONFIG))
    for name in list(bundle.output_paths):
        p = OUT / ("root" if name == "root" else name)
        p.mkdir(parents=True, exist_ok=True)
        bundle.output_paths[name] = p

    transfer_ids = list(bundle.config["transfer_pairs"].keys())
    seeds = [int(s) for s in bundle.config["seeds"]]

    probe_split, _ = tt.build_transfer_split(bundle, transfer_ids[0], seeds[0])
    _, m_all = _ORIGINAL(bundle, probe_split)
    _, m_tr = build_source_label_inputs_train_only(bundle, probe_split)
    print(f"结构估计标签范围：原实现 {int(m_all.sum())} 个源域节点 -> 仅 source_train {int(m_tr.sum())} 个"
          f"（差 {int(m_all.sum()) - int(m_tr.sum())} 个源验证样本）")

    tt.build_source_label_inputs = build_source_label_inputs_train_only

    rows = []
    total = len(transfer_ids) * len(seeds) * len(METHODS)
    done = 0
    for transfer_id in transfer_ids:
        for seed in seeds:
            split_frame, _ = tt.build_transfer_split(bundle, transfer_id, seed)
            encoder_state, ckpt = tt.load_encoder_state(bundle, seed)
            for method in METHODS:
                t0 = time.time()
                metrics, _pred, _h, _p, _s = tt.run_single_experiment(
                    bundle=bundle, split_frame=split_frame, encoder_state=encoder_state,
                    seed=seed, transfer_id=transfer_id, method=method, smoke_test=False,
                )
                metrics["wall_runtime_s"] = float(time.time() - t0)
                metrics["structure_labels"] = "source_train_only"
                rows.append(metrics)
                done += 1
                print(f"[{done:>2}/{total}] {transfer_id:<20}{method:<14}seed{seed} "
                      f"primary={metrics['primary_metric_value']:.4f} ({metrics['wall_runtime_s']:.1f}s)", flush=True)

    frame = pd.DataFrame(rows).sort_values(["transfer_id", "method", "seed"]).reset_index(drop=True)
    frame.to_csv(OUT / "m03_strurw_source_train_only.csv", index=False, encoding="utf-8")

    released = pd.read_csv(WS / "output" / "week7" / "metrics" / "transfer_summary.csv")
    rel = released[released["method"].isin(METHODS)]
    merged = frame.merge(rel, on=["transfer_id", "method", "seed"], suffixes=("_isolated", "_released"))
    merged["delta"] = merged["primary_metric_value_isolated"] - merged["primary_metric_value_released"]
    merged[["transfer_id", "method", "seed", "primary_metric_value_isolated",
            "primary_metric_value_released", "delta"]].to_csv(
        OUT / "m03_comparison.csv", index=False, encoding="utf-8")

    print("\n按情境×方法的均值对比（仅 source_train 估权 vs 已发布）：")
    g = merged.groupby(["transfer_id", "method"]).agg(
        isolated=("primary_metric_value_isolated", "mean"),
        released=("primary_metric_value_released", "mean"),
        delta=("delta", "mean"),
    ).reset_index()
    for _, r in g.iterrows():
        print(f"  {r['transfer_id']:<20}{r['method']:<14}"
              f"隔离={r['isolated']:.4f}  已发布={r['released']:.4f}  差={r['delta']:+.4f}")
    print(f"\n最大绝对差 {merged['delta'].abs().max():.4f}，平均绝对差 {merged['delta'].abs().mean():.4f}")
    print(f"写出 {OUT / 'm03_comparison.csv'}")


if __name__ == "__main__":
    main()
