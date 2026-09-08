"""M02: run the real EERM (adversarial graph editors) alongside the K-means proxy.

Usage:
    python exp_m02_eerm_real.py --smoke      # one scenario, one seed, K=2, few epochs
    python exp_m02_eerm_real.py              # all scenarios x seeds x K in {2,4}
"""
# NOTE: this script was run against the controlled research workspace, whose sample-level
# split files and model checkpoints are not part of the public release. WS and the output
# directory below are the paths used for the recorded run; set them to your own copies to
# re-execute. Only aggregate outputs are published, under reports/rerun_2026-09-08/.

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

WS = Path(r"D:\codex\hetero_transfer_experiment")
sys.path.insert(0, str(WS))
os.chdir(WS)

import numpy as np
import pandas as pd
import torch
import yaml

from src.train.train_invariance_methods import index_tensor
from src.train.train_transfer import (
    build_label_vector,
    build_metapath_adjacency,
    build_transfer_split,
    evaluate_binary,
    load_encoder_state,
    make_finetune_model,
)
from src.train.utils import load_graph_bundle, set_random_seed

SCRATCH = Path(r"<scratch_dir>")
from src.transfer.eerm_adversarial import EdgeEditor, build_add_pool, eerm_step  # noqa: E402

CONFIG7 = WS / "configs" / "week7_transfer.yaml"
CONFIG11 = WS / "configs" / "week11.yaml"
OUT = SCRATCH / "m02_out"
BETA = 1.0
EDITOR_LR = 0.05
ADD_CANDIDATES = 200
# 50% keep probability gives the editors maximum freedom to diversify the environments;
# a sweep over init logit {2,0,-1} x editor lr {0.01,0.05,0.1} x beta {1,5} moved the achieved
# risk variance by less than a factor of two, so this setting is not a tuned optimum.
INIT_KEEP_LOGIT = 0.0


def run_one(bundle, split_frame, transfer_id, seed, K, p2, max_epochs, editor_lr=EDITOR_LR):
    """Train one real-EERM run and return its metrics plus the training history."""
    set_random_seed(seed)
    device = torch.device("cpu")
    data = copy.deepcopy(bundle.graph_data).to(device)
    metapath_adjacency = build_metapath_adjacency(bundle, bundle.graph_data, device)
    encoder_state, ckpt = load_encoder_state(bundle, seed)
    model = make_finetune_model(bundle, encoder_state, device)
    labels = build_label_vector(bundle, split_frame).to(device)
    train_idx = index_tensor(split_frame, "source_train", device)
    val_idx = index_tensor(split_frame, "source_val", device)
    test_idx = index_tensor(split_frame, "target_test", device)

    gen = torch.Generator().manual_seed(seed)
    editor = EdgeEditor(data, n_env=K, add_pool=build_add_pool(data, ADD_CANDIDATES, gen),
                        init_keep_logit=INIT_KEEP_LOGIT)
    editor_opt = torch.optim.Adam(editor.parameters(), lr=editor_lr)
    model_opt = torch.optim.Adam(
        [
            {"params": model.encoder_parameters(), "lr": float(p2["learning_rate_encoder"])},
            {"params": model.head_parameters(), "lr": float(p2["learning_rate_head"])},
        ],
        weight_decay=float(p2["weight_decay"]),
    )

    val_labels = split_frame.loc[split_frame["split"] == "source_val", "label"].to_numpy(int)
    best_state, best_score, best_epoch = None, -math.inf, 0
    patience = int(p2["patience"])
    min_epochs = int(p2["min_epochs_before_early_stop"])
    hist = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        stats = eerm_step(model, data, metapath_adjacency, labels, train_idx,
                          editor, editor_opt, model_opt, BETA)
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data, metapath_adjacency)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        v = probs[val_idx.cpu().numpy()]
        score = evaluate_binary(val_labels, v, 0.5)["roc_auc"] if np.unique(val_labels).size == 2 else -stats["loss"]
        hist.append({"epoch": epoch, "source_val_auc": float(score), **stats})
        if math.isfinite(score) and score >= best_score:
            best_score, best_epoch, best_state = float(score), epoch, copy.deepcopy(model.state_dict())
        elif epoch >= min_epochs and best_state is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data, metapath_adjacency)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    t_scores = probs[test_idx.cpu().numpy()]
    t_labels = split_frame.loc[split_frame["split"] == "target_test", "label"].to_numpy(int)
    target_auc = float("nan")
    if np.unique(t_labels).size == 2:
        target_auc = evaluate_binary(t_labels, t_scores, 0.5)["roc_auc"]
    # one-class targets (T1_DK) use illegal recall at the Youden threshold of the source val ROC
    from src.train.utils import choose_threshold_by_youden
    youden = float("nan")
    recall_at_youden = float("nan")
    if np.unique(val_labels).size == 2:
        youden = float(choose_threshold_by_youden(val_labels, probs[val_idx.cpu().numpy()]))
        ill = t_scores[t_labels == 1]
        recall_at_youden = float((ill >= youden).mean()) if ill.size else float("nan")
    return {
        "transfer_id": transfer_id, "seed": seed, "method": f"eerm_real_K{K}",
        "K": K, "beta": BETA, "editor_lr": editor_lr, "init_keep_logit": INIT_KEEP_LOGIT,
        "target_test_auc": target_auc,
        "target_illegal_recall_at_youden": recall_at_youden,
        "youden_threshold": youden,
        "target_mean_pred_illegal": float(t_scores.mean()),
        "source_val_auc": float(best_score) if math.isfinite(best_score) else float("nan"),
        "best_epoch": best_epoch, "trained_epochs": len(hist),
        "risk_var_last": hist[-1]["risk_var"] if hist else float("nan"),
        "runtime_s": time.time() - t0,
        "encoder_checkpoint": Path(ckpt).name,
        "implementation": "adversarial_edge_editors_reinforce_variance",
        "is_posthoc": True,
    }, pd.DataFrame(hist)


def main() -> None:
    """Run the smoke test or the full real-EERM grid."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_graph_bundle(str(CONFIG7))
    p2 = yaml.safe_load(CONFIG11.read_text(encoding="utf-8"))["P2_invariance_methods"]

    transfers = list(bundle.config["transfer_pairs"].keys())
    seeds = [int(s) for s in bundle.config["seeds"]]
    Ks = [2, 4]
    max_epochs = int(p2["max_epochs"])
    if args.smoke:
        transfers, seeds, Ks, max_epochs = ["T2_PH"], [42], [2], 15
    only = os.environ.get("EERM_ONLY")
    if only:
        transfers = only.split(",")

    rows, hists = [], []
    total = len(transfers) * len(seeds) * len(Ks)
    done = 0
    for tid in transfers:
        for seed in seeds:
            split_frame, _ = build_transfer_split(bundle, tid, seed)
            for K in Ks:
                m, h = run_one(bundle, split_frame, tid, seed, K, p2, max_epochs)
                rows.append(m)
                h["transfer_id"], h["seed"], h["K"] = tid, seed, K
                hists.append(h)
                done += 1
                print(f"[{done:>2}/{total}] {tid:<20}K={K} seed{seed} "
                      f"target_auc={m['target_test_auc']:.4f} val={m['source_val_auc']:.4f} "
                      f"risk_var={m['risk_var_last']:.2e} ({m['runtime_s']:.0f}s)", flush=True)

    suffix = "_smoke" if args.smoke else (os.environ.get("EERM_SUFFIX") or "")
    pd.DataFrame(rows).to_csv(OUT / f"m02_eerm_real{suffix}.csv", index=False, encoding="utf-8")
    pd.concat(hists, ignore_index=True).to_csv(OUT / f"m02_eerm_real_history{suffix}.csv", index=False, encoding="utf-8")
    print(f"\n写出 {OUT / f'm02_eerm_real{suffix}.csv'}")


if __name__ == "__main__":
    main()
