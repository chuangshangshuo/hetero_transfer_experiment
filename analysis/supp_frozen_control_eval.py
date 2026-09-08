"""M04: frozen-model external-control test.

The released 4.6 result (illegal vs commercial control, AUC 0.970+-0.013) comes from
multi_scenario_eval.run_e4, which rebuilds a train/val/test split per scenario and *retrains*
the classifier. It therefore measures "can the model separate the two classes after being
trained on them", not "does a fixed blacklist/whitelist classifier generalise to commercial
sites it has never seen".

This script runs the missing experiment: it loads the main-result mlp_64 checkpoints -- trained
on illegal vs licensed only, with the 82 control_legal_commercial sites never present in any
training, validation or test split -- freezes them, and evaluates three cohorts with no further
fitting of any kind.
"""
# NOTE: this script was run against the controlled research workspace, whose sample-level
# split files and model checkpoints are not part of the public release. WS and the output
# directory below are the paths used for the recorded run; set them to your own copies to
# re-execute. Only aggregate outputs are published, under reports/rerun_2026-09-08/.

import copy
import json
import os
import sys
from pathlib import Path

WS = Path(r"D:\codex\hetero_transfer_experiment")
sys.path.insert(0, str(WS))
os.chdir(WS)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.hetero_full_with_heco import HeCoFineTuneClassifier
from src.train.train_full import build_metapath_adjacency, make_heco_model
from src.train.utils import build_primary_task_frame, load_graph_bundle, resolve_device

CONFIG = WS / "configs" / "week6_heco_finetune.yaml"
CKPT_DIR = WS / "output" / "week6_head_ablation" / "checkpoints"
SPLIT_DIR = WS / "output" / "week6_head_ablation" / "splits"
OUT = Path(r"<scratch_dir>\m04_out")
SEEDS = [42, 43, 44, 45, 46]
HEAD = "mlp_64"

ILLEGAL_TIERS = {"illegal_confirmed_official_single", "illegal_confirmed_official_cross_verified"}


def predict_all(bundle, seed: int) -> np.ndarray:
    """Load the frozen mlp_64 finetune checkpoint for `seed` and score every Website node."""
    device = resolve_device(bundle.config)
    graph_data = bundle.graph_data.to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, device)
    encoder = make_heco_model(bundle).to(device)
    model = HeCoFineTuneClassifier(
        heco_encoder=encoder,
        head_type=HEAD,
        hidden_dim=int(bundle.config["heco"]["hidden_dim"]),
        dropout=float(bundle.config["finetune"]["dropout"]),
    ).to(device)
    state = torch.load(CKPT_DIR / f"heco_{HEAD}_head__seed{seed}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(graph_data, metapath_adjacency)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return probs


def main() -> None:
    """Evaluate the three cohorts under a strictly frozen model and write the comparison."""
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_graph_bundle(str(CONFIG))
    website = bundle.website_frame.copy()

    controls = website[website["sample_tier"] == "control_legal_commercial"]
    control_idx = controls["graph_node_index"].to_numpy(dtype=int)
    print(f"合法商业对照样本：{len(control_idx)} 条（全部从未进入阶段二任何划分）")

    rows = []
    for seed in SEEDS:
        probs = predict_all(bundle, seed)
        split = pd.read_csv(SPLIT_DIR / f"head_ablation__{HEAD}__seed{seed}_split.csv")
        test = split[split["split"] == "test"]
        test_ill = test[test["sample_tier"].isin(ILLEGAL_TIERS)]["graph_node_index"].to_numpy(dtype=int)
        test_lic = test[test["sample_tier"] == "licensed_baseline"]["graph_node_index"].to_numpy(dtype=int)

        train_ids = set(split.loc[split["split"].isin(["train", "val"]), "graph_node_index"].astype(int))
        assert not (set(control_idx) & train_ids), "控制样本不应出现在训练/验证集内"

        def auc(pos_idx, neg_idx):
            y = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]
            s = np.r_[probs[pos_idx], probs[neg_idx]]
            return float(roc_auc_score(y, s)), float(average_precision_score(y, s))

        s1_auc, s1_pr = auc(test_ill, test_lic)
        s2_auc, s2_pr = auc(test_ill, control_idx)
        s3_auc, s3_pr = auc(test_ill, np.r_[test_lic, control_idx])
        rows.append({
            "seed": seed,
            "n_test_illegal": len(test_ill),
            "n_test_licensed": len(test_lic),
            "n_control": len(control_idx),
            "S1_frozen_illegal_vs_licensed_auc": s1_auc,
            "S2_frozen_illegal_vs_control_auc": s2_auc,
            "S3_frozen_illegal_vs_both_auc": s3_auc,
            "S1_pr_auc": s1_pr, "S2_pr_auc": s2_pr, "S3_pr_auc": s3_pr,
            "mean_prob_test_illegal": float(probs[test_ill].mean()),
            "mean_prob_test_licensed": float(probs[test_lic].mean()),
            "mean_prob_control": float(probs[control_idx].mean()),
            "protocol": "frozen_main_task_model_no_refit",
            "is_posthoc": True,
        })
        print(f"  seed{seed}: S1={s1_auc:.4f}  S2(冻结外部对照)={s2_auc:.4f}  S3={s3_auc:.4f} "
              f"| 均值概率 非法={probs[test_ill].mean():.3f} 持牌={probs[test_lic].mean():.3f} "
              f"对照={probs[control_idx].mean():.3f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "m04_frozen_control_eval.csv", index=False, encoding="utf-8")
    print("\n跨种子汇总（均值±样本标准差）：")
    for col, name in [("S1_frozen_illegal_vs_licensed_auc", "S1 非法 vs 持牌 (冻结)"),
                      ("S2_frozen_illegal_vs_control_auc", "S2 非法 vs 商业对照 (冻结, 外部)"),
                      ("S3_frozen_illegal_vs_both_auc", "S3 非法 vs 两者合并 (冻结)")]:
        print(f"  {name:<34}{frame[col].mean():.4f} ± {frame[col].std():.4f}")

    released = WS / "output" / "week8" / "metrics" / "E4_three_scenario_summary.csv"
    if released.exists():
        print(f"\n已发布的重训版本（{released.relative_to(WS)}）：")
        print(pd.read_csv(released).to_string(index=False))
    print(f"\n写出 {OUT / 'm04_frozen_control_eval.csv'}")


if __name__ == "__main__":
    main()
