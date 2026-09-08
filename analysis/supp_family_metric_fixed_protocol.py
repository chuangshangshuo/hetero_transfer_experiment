"""M05: family metric learning re-run under a fixed evaluation protocol.

The Week-11 P1 experiment cannot be interpreted because three things were wrong at once:
the alpha switch also switched the scoring function (predicted illegal probability for alpha=0,
centroid cosine otherwise), the reported positive class (the unseen family) was scored by a
quantity that measures similarity to *seen* families, and the sampled hard negatives stayed in
both the training set and the centroid pool.

This re-run fixes all three and separates the tasks that were previously conflated:

* every arm is scored by the same function (cosine to seen-family centroids), so arms differ
  only in the training objective;
* **T-B, unseen-family detection** scores novelty as the negative max-similarity to seen
  centroids, which is the direction the metric is actually trained to produce; positives are the
  held-out family, negatives are held-out members of *seen* families;
* one member of each seen family is reserved as a probe and removed from the training set and
  from centroid construction, so no test item is ever fitted or used to define the score;
* **T-A, family attribution** asks whether a probe's nearest centroid is its own family, against
  a 1/6 chance level.
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
from sklearn.metrics import roc_auc_score

from src.models.family_metric_head import FamilyDualHead, supervised_contrastive_loss
from src.train.train_family_metric import (
    build_family_frame,
    eligible_dk_families,
    load_frozen_embeddings,
    load_week11_config,
)
from src.train.utils import load_graph_bundle, set_random_seed
import torch.nn.functional as F

CONFIG11 = WS / "configs" / "week11.yaml"
CONFIG7 = WS / "configs" / "week7_transfer.yaml"
OUT = Path(r"<scratch_dir>\m05_out")
SEEDS = [42, 43, 44, 45, 46]

ARMS = [
    {"name": "A1_no_supcon", "alpha": 0.0, "binary_weight": 1.0, "shuffle": False},
    {"name": "A2_dual", "alpha": 0.5, "binary_weight": 1.0, "shuffle": False},
    {"name": "A3_supcon_only", "alpha": 1.0, "binary_weight": 0.0, "shuffle": False},
    {"name": "A4_shuffled", "alpha": 0.5, "binary_weight": 1.0, "shuffle": True},
]
TIERS = ["licensed_baseline", "illegal_confirmed_official_single", "illegal_confirmed_official_cross_verified"]


def build_split(frame, heldout, seed, families):
    """Return (train_frame, probe_index_by_family) with probes excluded from training."""
    rng = np.random.default_rng(seed * 7919 + len(heldout))
    probes = {}
    for fam in families:
        if fam == heldout:
            continue
        members = frame.index[frame["p1_family_id"].astype(str).eq(fam)].to_numpy()
        probes[fam] = int(rng.choice(members))
    probe_rows = set(probes.values())
    keep = frame["sample_tier"].isin(TIERS) & ~frame["p1_family_id"].astype(str).eq(heldout)
    keep &= ~frame.index.isin(probe_rows)
    return frame[keep].copy(), probes


def train_arm(x, y, fam_labels, arm, base_cfg, seed):
    """Train the dual head for one arm; identical optimiser and schedule for every arm."""
    set_random_seed(seed)
    model = FamilyDualHead(embed_dim=x.shape[1], projection_dim=int(base_cfg["projection_dim"]),
                           hidden_dim=int(base_cfg["hidden_dim"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(base_cfg["learning_rate"]),
                           weight_decay=float(base_cfg["weight_decay"]))
    best_state, best_loss, best_epoch = None, float("inf"), 0
    for epoch in range(1, int(base_cfg["epochs"]) + 1):
        model.train(); opt.zero_grad()
        logits, z = model(x)
        ce = F.cross_entropy(logits, y)
        sup = supervised_contrastive_loss(z, fam_labels, temperature=float(base_cfg["temperature"]))
        loss = float(arm["binary_weight"]) * ce + float(arm["alpha"]) * sup
        loss.backward(); opt.step()
        cur = float(loss.detach())
        if cur < best_loss:
            best_loss, best_epoch, best_state = cur, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= int(base_cfg["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    """Run every held-out family x seed x arm under the corrected protocol."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg11 = load_week11_config(CONFIG11)
    base = cfg11["P1_family_metric"]
    bundle = load_graph_bundle(str(CONFIG7))
    frame = build_family_frame(bundle, cfg11)
    families = eligible_dk_families(frame, int(base["eligible_family_min_size"]))
    print(f"合格家族 {len(families)}：{families}")

    rows = []
    for seed in SEEDS:
        emb, ckpt = load_frozen_embeddings(bundle, seed)
        for heldout in families:
            train_frame, probes = build_split(frame, heldout, seed, families)
            # family labels for the SupCon objective, over training rows only
            counts = train_frame.loc[train_frame["p1_family_id"].notna(), "p1_family_id"].value_counts()
            valid = sorted(counts[counts >= 2].index.astype(str))
            fam_to_int = {f: i for i, f in enumerate(valid)}
            fl = np.full(len(train_frame), -1, dtype=np.int64)
            for i, f in enumerate(train_frame["p1_family_id"].astype(str)):
                if f in fam_to_int:
                    fl[i] = fam_to_int[f]
            x = torch.tensor(emb[train_frame["graph_node_index"].to_numpy(int)], dtype=torch.float32)
            y = torch.tensor(train_frame["label"].to_numpy(np.int64), dtype=torch.long)

            seen = [f for f in families if f != heldout]
            probe_idx = np.array([probes[f] for f in seen])
            probe_gni = frame.loc[probe_idx, "graph_node_index"].to_numpy(int)
            held_gni = frame.loc[frame["p1_family_id"].astype(str).eq(heldout), "graph_node_index"].to_numpy(int)

            for arm in ARMS:
                fl_arm = fl.copy()
                if arm["shuffle"]:
                    rng = np.random.default_rng(seed * 104729 + 17)
                    m = np.where(fl_arm >= 0)[0]
                    fl_arm[m] = rng.permutation(fl_arm[m])
                model = train_arm(x, y, torch.tensor(fl_arm, dtype=torch.long), arm, base, seed)
                with torch.no_grad():
                    _, z_all = model(torch.tensor(emb, dtype=torch.float32))
                z = torch.nn.functional.normalize(z_all, dim=-1).numpy()

                # centroids from TRAINING members of seen families only (probes excluded)
                cents, cent_fams = [], []
                for f in seen:
                    rows_f = train_frame[train_frame["p1_family_id"].astype(str).eq(f)]
                    if rows_f.empty:
                        continue
                    c = z[rows_f["graph_node_index"].to_numpy(int)].mean(axis=0)
                    cents.append(c / max(np.linalg.norm(c), 1e-12)); cent_fams.append(f)
                C = np.vstack(cents)

                # T-B: unseen-family detection. novelty = -max cosine to seen centroids.
                sim_held = (z[held_gni] @ C.T).max(axis=1)
                sim_probe = (z[probe_gni] @ C.T).max(axis=1)
                y_true = np.r_[np.ones(len(sim_held)), np.zeros(len(sim_probe))]
                novelty = np.r_[-sim_held, -sim_probe]
                tb_auc = float(roc_auc_score(y_true, novelty)) if len(set(y_true)) == 2 else float("nan")

                # T-A: family attribution on probes (nearest centroid == own family?)
                pred_f = [cent_fams[i] for i in (z[probe_gni] @ C.T).argmax(axis=1)]
                ta_acc = float(np.mean([p == f for p, f in zip(pred_f, seen)]))

                rows.append({
                    "seed": seed, "heldout_family": heldout, "arm": arm["name"],
                    "alpha": arm["alpha"], "binary_weight": arm["binary_weight"],
                    "shuffle_family_labels": arm["shuffle"],
                    "scoring": "cosine_to_seen_centroids (identical across arms)",
                    "TB_unseen_family_detection_auc": tb_auc,
                    "TA_family_attribution_top1": ta_acc,
                    "TA_chance": 1.0 / len(cent_fams),
                    "n_heldout": len(held_gni), "n_probes": len(probe_gni),
                    "n_train": len(train_frame), "n_centroids": len(cent_fams),
                    "mean_sim_heldout": float(sim_held.mean()),
                    "mean_sim_probe": float(sim_probe.mean()),
                    "probes_in_training": False, "probes_in_centroids": False,
                    "encoder_checkpoint": Path(ckpt).name, "is_posthoc": True,
                })
        print(f"  seed{seed} 完成", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "m05_family_fixed_protocol.csv", index=False, encoding="utf-8")
    print(f"\n写出 {OUT / 'm05_family_fixed_protocol.csv'}  ({len(out)} 行)\n")
    print("T-B 未见家族检测 AUC（越高越好，随机=0.5）与 T-A 家族归属 top-1（随机≈0.167）：")
    for arm in ARMS:
        p = out[out["arm"] == arm["name"]]
        print(f"  {arm['name']:<16} T-B={p['TB_unseen_family_detection_auc'].mean():.3f}±"
              f"{p['TB_unseen_family_detection_auc'].std():.3f}   "
              f"T-A={p['TA_family_attribution_top1'].mean():.3f}±{p['TA_family_attribution_top1'].std():.3f}")


if __name__ == "__main__":
    main()
