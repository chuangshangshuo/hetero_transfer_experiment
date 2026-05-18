from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "week10_final.yaml"
OUT = ROOT / "output" / "week10"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def input_path(config: dict[str, Any], group: str, key: str) -> Path:
    return resolve(config["inputs"][group][key])


def output_path(config: dict[str, Any], group: str, name: str) -> Path:
    root = resolve(config["outputs"][group])
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ["root", "metrics", "tables", "figures", "audits"]:
        resolve(config["outputs"][key]).mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> pd.DataFrame:
    full = resolve(path)
    if not full.exists():
        return pd.DataFrame()
    return pd.read_csv(full)


def read_input(config: dict[str, Any], group: str, key: str) -> pd.DataFrame:
    return read_csv(input_path(config, group, key))


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    full = resolve(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(full, index=False, encoding="utf-8")


def first_float(frame: pd.DataFrame, column: str, default: float = float("nan")) -> float:
    if frame.empty or column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.iloc[0])


def first_text(frame: pd.DataFrame, column: str, default: str = "") -> str:
    if frame.empty or column not in frame.columns:
        return default
    value = frame[column].dropna()
    if value.empty:
        return default
    return str(value.iloc[0])


def with_claim_columns(
    row: dict[str, Any],
    status: str,
    is_posthoc: bool,
    safe: str,
    forbidden: str,
) -> dict[str, Any]:
    row["status"] = status
    row["is_posthoc"] = bool(is_posthoc)
    row["paper_safe_claim"] = safe
    row["paper_forbidden_claim"] = forbidden
    return row


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in text.to_numpy(dtype=str)]
    return "\n".join([header, sep, *body])


def canonical_feature_condition(value: str) -> str:
    mapping = {
        "C1_full": "full",
        "C2_no_cctld": "no_cctld",
        "C3_no_website_lexical": "no_website_lexical",
        "C4_graph_only": "graph_only",
        "C5_lex_only": "lexical_only",
        "C6_no_edges": "no_edges",
        "C9_infra_edges_only": "infra_edges_only",
        "no-ccTLD": "no_cctld",
        "no-website-lexical": "no_website_lexical",
        "graph-only": "graph_only",
    }
    return mapping.get(str(value), str(value))
