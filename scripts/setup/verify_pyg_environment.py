from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
import torch_geometric


MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    "networkx",
    "tqdm",
    "maxminddb",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
]


def main() -> None:
    versions: dict[str, str] = {
        "python_env": "hetero-transfer-v2",
        "torch": torch.__version__,
        "torch_cuda_available": str(torch.cuda.is_available()),
        "torch_geometric": torch_geometric.__version__,
    }
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        versions[module_name] = getattr(module, "__version__", "installed")

    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "versions" / "pyg_environment_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(versions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
