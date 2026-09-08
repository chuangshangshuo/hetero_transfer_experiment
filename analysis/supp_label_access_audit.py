"""Label-access audit: which labelled samples were visible at each model-selection step.

The manuscript previously implied that no evaluation-set label was touched anywhere between
pre-training and final evaluation. That is not accurate: the HeCo contrastive temperature was
selected by the *pooled-primary validation* ROC-AUC (``train_contrastive.py`` picks
``frozen_linear_val_roc_auc.idxmax()``), and the pooled-primary validation split and the
per-scenario transfer ``target_test`` splits are drawn from the same 493-sample labelled pool.
This script quantifies the resulting overlap exactly, per transfer scenario and seed.

It reads sample-level split files, which are controlled material and are not part of the public
release, so it is written to run against a workspace root given on the command line and to emit
**counts only** -- no node ids, domains or other site identifiers ever reach the output.

Usage
-----
    python analysis/supp_label_access_audit.py --workspace D:/codex/hetero_transfer_experiment

Outputs ``reports/supplementary_exact_significance/supp_label_access_audit.csv`` plus a short
console summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "supplementary_exact_significance" / "supp_label_access_audit.csv"

# The temperature grid was run on seed 42 only; that single validation split is what selected
# tau=0.7, and the tau=0.7 encoder is then loaded by every downstream transfer run.
SELECTION_SPLIT = "output/week5/splits/heco_pooled_primary__seed42.csv"
TRANSFER_GLOB = "output/week7/splits/*__seed*.csv"
FINETUNE_GLOB = "output/week6/splits/*__seed*.csv"


def read_ids(path: Path, split_values: set[str], column: str = "split") -> set[str]:
    """Return the node_id set for the requested split values of one split file."""
    frame = pd.read_csv(path, usecols=lambda c: c in {"node_id", column})
    return set(frame.loc[frame[column].isin(split_values), "node_id"].astype(str))


def main() -> None:
    """Compute and write the label-access overlap table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Root of the original research workspace")
    args = parser.parse_args()
    ws = Path(args.workspace)

    sel_path = ws / SELECTION_SPLIT
    if not sel_path.exists():
        raise FileNotFoundError(f"temperature-selection split not found: {sel_path}")
    sel_val = read_ids(sel_path, {"val"})
    sel_train = read_ids(sel_path, {"train"})
    sel_test = read_ids(sel_path, {"test"})
    print(f"温度选优所用划分（seed42）：train={len(sel_train)} val={len(sel_val)} test={len(sel_test)}")

    rows = []
    for path in sorted(ws.glob(TRANSFER_GLOB)):
        stem = path.stem
        transfer_id, seed_part = stem.rsplit("__", 1)
        seed = int(seed_part.replace("seed", ""))
        target_test = read_ids(path, {"target_test"})
        source_train = read_ids(path, {"source_train"})
        source_val = read_ids(path, {"source_val"})
        rows.append({
            "transfer_id": transfer_id,
            "seed": seed,
            "target_test_size": len(target_test),
            "overlap_with_tau_selection_val": len(target_test & sel_val),
            "overlap_with_tau_selection_train": len(target_test & sel_train),
            "overlap_with_tau_selection_test": len(target_test & sel_test),
            "source_val_size": len(source_val),
            "source_val_overlap_with_tau_selection_val": len(source_val & sel_val),
            "source_train_size": len(source_train),
            "source_train_overlap_with_tau_selection_val": len(source_train & sel_val),
            "is_posthoc": True,
        })

    frame = pd.DataFrame(rows).sort_values(["transfer_id", "seed"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\n写出 {OUT.relative_to(ROOT)}  （{len(frame)} 行，仅计数）\n")

    pivot = frame.pivot_table(index="transfer_id", values=["target_test_size", "overlap_with_tau_selection_val"],
                              aggfunc=["min", "max", "sum"])
    print("按情境汇总（target_test 与温度选优验证集的交集）：")
    for tid, part in frame.groupby("transfer_id"):
        counts = part["overlap_with_tau_selection_val"].tolist()
        sizes = part["target_test_size"].tolist()
        print(f"  {tid:<20} target_test={sizes}  交集={counts}  "
              f"合计交集={sum(counts)}/{sum(sizes)}")

    total_overlap = int(frame["overlap_with_tau_selection_val"].sum())
    total_test = int(frame["target_test_size"].sum())
    print(f"\n全部 20 次运行：target_test 合计 {total_test} 条，其中 {total_overlap} 条曾出现在温度选优验证集中"
          f"（{total_overlap / total_test:.1%}）")

    # A same-pool draw would put roughly |val| / |pool| of any test split inside the val split.
    pool = len(sel_train | sel_val | sel_test)
    print(f"参照：温度选优划分覆盖 {pool} 个样本，其中验证集占 {len(sel_val) / pool:.1%}；"
          f"若两次划分相互独立，交集比例的期望即约为该值")

    ft_paths = sorted(ws.glob(FINETUNE_GLOB))
    if ft_paths:
        print(f"\n另有 {len(ft_paths)} 个阶段二微调划分文件，可用于分类头选择的标签访问核对")


if __name__ == "__main__":
    main()
