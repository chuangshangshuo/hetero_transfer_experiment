"""Editor-gradient diagnostic for the corrected adversarial EERM.

The v15 review asked for direct evidence that the editors receive a non-zero effective gradient
and that their parameters actually move, separately for K=2 and K=4, since the first
implementation's squared reward cancelled exactly at K=2. This script instruments one training
run per (K, scenario) and records, per iteration: the per-environment advantages, the editor
gradient norm, the parameter displacement, and the resulting risk variance.
"""
import os
import sys
from pathlib import Path

WS = Path(r"D:\codex\hetero_transfer_experiment")
sys.path.insert(0, str(WS))
os.chdir(WS)
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))

import pandas as pd
import torch
import torch.nn.functional as F

from eerm_real import EdgeEditor, apply_edits, build_add_pool
from src.train.train_transfer import build_transfer_split, load_encoder_state
from src.train.utils import load_graph_bundle
from src.models.hetero_full_with_heco import HeCoFineTuneClassifier
from src.train.train_full import build_metapath_adjacency, make_heco_model

CONFIG = WS / "configs" / "week7_transfer.yaml"
OUT = SCRATCH / "m02_out"
ITERS = 25


def diagnose(bundle, transfer_id: str, seed: int, K: int) -> list[dict]:
    """Instrument ITERS editor steps and return per-iteration diagnostics."""
    torch.manual_seed(seed)
    split = build_transfer_split(bundle, transfer_id, seed)[0]
    enc_state, _ = load_encoder_state(bundle, seed)
    device = torch.device("cpu")
    data = bundle.graph_data.to(device)
    mp = build_metapath_adjacency(bundle, device)
    encoder = make_heco_model(bundle).to(device)
    encoder.load_state_dict(enc_state)
    model = HeCoFineTuneClassifier(
        heco_encoder=encoder, head_type="mlp_64",
        hidden_dim=int(bundle.config["heco"]["hidden_dim"]),
        dropout=float(bundle.config["finetune"]["dropout"]),
    ).to(device)

    train_rows = split[split["split"] == "source_train"]
    idx = torch.tensor(train_rows["graph_node_index"].to_numpy(int))
    labels = torch.tensor(train_rows["label"].to_numpy(int), dtype=torch.long)

    gen = torch.Generator().manual_seed(seed)
    add_pool = build_add_pool(data, 200, gen)
    editor = EdgeEditor(data, n_env=K, add_pool=add_pool, init_keep_logit=0.0)
    opt = torch.optim.Adam(editor.parameters(), lr=0.05)

    rows = []
    for it in range(1, ITERS + 1):
        before = torch.cat([p.detach().flatten().clone() for p in editor.parameters()])
        log_probs, risks = [], []
        for k in range(K):
            keep, add, logp = editor.sample(k)
            edited = apply_edits(data, editor, keep, add)
            with torch.no_grad():
                logits = model(edited, mp)[0]
                risks.append(F.cross_entropy(logits[idx], labels))
            log_probs.append(logp)
        rv = torch.stack(risks)
        reward = (rv - rv.mean()).detach()
        loss = -torch.stack([lp * r for lp, r in zip(log_probs, reward)]).mean()
        opt.zero_grad()
        loss.backward()
        gnorm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in editor.parameters() if p.grad is not None))
        opt.step()
        after = torch.cat([p.detach().flatten().clone() for p in editor.parameters()])
        rows.append({
            "transfer_id": transfer_id, "seed": seed, "K": K, "iteration": it,
            "risks": ";".join(f"{float(r):.6f}" for r in rv),
            "advantage_abs_max": float(reward.abs().max()),
            "advantage_all_zero": bool(float(reward.abs().max()) < 1e-12),
            "editor_grad_norm": float(gnorm),
            "param_l2_step": float((after - before).norm()),
            "risk_variance": float(rv.var(unbiased=False)),
        })
    return rows


def main() -> None:
    """Diagnose one run per (scenario, K) and summarise."""
    bundle = load_graph_bundle(str(CONFIG))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for transfer_id in ["T1_Nordic", "T2_PH", "T2_ON", "T3_DiagnoseFrance"]:
        for K in (2, 4):
            rows += diagnose(bundle, transfer_id, 42, K)
            print(f"  {transfer_id} K={K} 完成", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "m02_editor_gradient_diagnostic.csv", index=False, encoding="utf-8")
    print(f"\n写出 {OUT / 'm02_editor_gradient_diagnostic.csv'}  ({len(frame)} 行)\n")
    print(f"{'情境':<20}{'K':>3}{'零优势迭代':>11}{'|优势|中位':>12}{'梯度范数中位':>13}{'参数步长中位':>13}")
    for (t, K), g in frame.groupby(["transfer_id", "K"]):
        print(f"{t:<20}{K:>3}{int(g.advantage_all_zero.sum()):>11}"
              f"{g.advantage_abs_max.median():>12.3e}{g.editor_grad_norm.median():>13.3e}"
              f"{g.param_l2_step.median():>13.3e}")
    print(f"\n全部 {len(frame)} 次迭代中优势恒为零的次数：{int(frame.advantage_all_zero.sum())}")


if __name__ == "__main__":
    main()
