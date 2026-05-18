from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.builder_common import assert_no_all_zero_columns
else:
    from .builder_common import assert_no_all_zero_columns


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Convert Graph v2 NPZ package into PyG HeteroData.")
    parser.add_argument("--npz", type=Path, default=root / "data" / "graphs" / "hetero_graph_v2.npz")
    parser.add_argument("--builder-state", type=Path, default=root / "data" / "versions" / "feature_builder_state.json")
    parser.add_argument("--out", type=Path, default=root / "data" / "graphs" / "hetero_graph_v2.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch_geometric.data import HeteroData
    except Exception as exc:
        raise SystemExit(
            "torch and torch_geometric are required. Install the Week 3 PyG environment first. "
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc

    if not args.builder_state.exists():
        raise SystemExit(f"Missing feature builder state: {args.builder_state}")

    package = np.load(args.npz, allow_pickle=True)
    builder_state = json.loads(args.builder_state.read_text(encoding="utf-8"))
    node_builders = builder_state["node_feature_builders"]
    edge_type_triples = {edge_type: tuple(triple) for edge_type, triple in builder_state["edge_type_triples"].items()}

    data = HeteroData()
    for node_type, state in node_builders.items():
        key = f"x__{node_type}"
        if key not in package:
            raise SystemExit(f"Missing node feature array: {key}")
        matrix = package[key]
        assert_no_all_zero_columns(matrix, state["active_feature_columns"], node_type)
        data[node_type].x = torch.tensor(matrix, dtype=torch.float32)
        node_id_key = f"node_id__{node_type}"
        if node_id_key in package:
            data[node_type].node_id = package[node_id_key].astype(str).tolist()

    for edge_type, triple in edge_type_triples.items():
        edge_index_key = f"edge_index__{edge_type}"
        edge_weight_key = f"edge_weight__{edge_type}"
        if edge_index_key not in package:
            raise SystemExit(f"Missing edge index array: {edge_index_key}")
        if edge_weight_key not in package:
            raise SystemExit(f"Missing edge weight array: {edge_weight_key}")
        data[triple].edge_index = torch.tensor(package[edge_index_key], dtype=torch.long)
        data[triple].edge_weight = torch.tensor(package[edge_weight_key], dtype=torch.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    print(f"pyg_export_complete={args.out}")


if __name__ == "__main__":
    main()
