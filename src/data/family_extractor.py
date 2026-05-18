from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.utils import GraphBundle, build_primary_task_frame, load_graph_bundle, save_dataframe


def _normalise_domain(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^www\.", "", text)
    return text


def _compact_domain(value: str) -> str:
    return re.sub(r"[^a-z]+", "", value.lower())


def _pattern_family(root_domain: str, patterns: dict[str, list[str]]) -> tuple[str | None, str | None]:
    root = _normalise_domain(root_domain)
    compact = _compact_domain(root)
    for family, raw_patterns in patterns.items():
        for raw_pattern in raw_patterns:
            pattern = str(raw_pattern).strip().lower()
            if not pattern:
                continue
            compact_pattern = _compact_domain(pattern)
            if pattern in root or (compact_pattern and compact_pattern in compact):
                return family, f"pattern:{pattern}"
    return None, None


def _hosted_on_components(bundle: GraphBundle, candidate_indices: set[int]) -> list[set[int]]:
    if not candidate_indices:
        return []
    edge_key = ("Website", "hosted_on", "IP")
    if edge_key not in bundle.graph_data.edge_types:
        return []
    edge_index = bundle.graph_data[edge_key].edge_index.detach().cpu().numpy()
    ip_to_websites: dict[int, list[int]] = defaultdict(list)
    for website_idx, ip_idx in zip(edge_index[0], edge_index[1]):
        website_int = int(website_idx)
        if website_int in candidate_indices:
            ip_to_websites[int(ip_idx)].append(website_int)

    adjacency: dict[int, set[int]] = {idx: set() for idx in candidate_indices}
    for websites in ip_to_websites.values():
        unique = sorted(set(websites))
        if len(unique) < 2:
            continue
        for idx in unique:
            adjacency[idx].update(other for other in unique if other != idx)

    components: list[set[int]] = []
    seen: set[int] = set()
    for start in sorted(candidate_indices):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        component: set[int] = set()
        seen.add(start)
        while queue:
            current = queue.popleft()
            component.add(current)
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        if len(component) >= 2:
            components.append(component)
    return components


def extract_denmark_families(bundle: GraphBundle) -> pd.DataFrame:
    """Rebuild Denmark family labels using domain stems plus hosted-on components."""
    primary = build_primary_task_frame(bundle)
    denmark = primary[primary["jurisdiction"] == "Denmark"].copy().reset_index(drop=True)
    patterns = dict(bundle.config.get("E3_lofo", {}).get("family_patterns", {}))
    if not patterns:
        patterns = {
            "tsars": ["tsars"],
            "icecasino": ["icecasino", "icecazino"],
            "verdecasino": ["verdecasino"],
            "ggbet": ["ggbet", "gbett", "ggbbbet", "ggonline", "gg54", "i-ggbet"],
            "bcgame": ["bcgame", "bc-game"],
            "bcdot": ["bc."],
            "freshbet": ["freshbet", "fresh."],
        }

    family_ids: list[str] = []
    family_sources: list[str] = []
    for _, row in denmark.iterrows():
        family, source = _pattern_family(str(row["root_domain"]), patterns)
        if family is None:
            family_ids.append("")
            family_sources.append("")
        else:
            family_ids.append(str(family))
            family_sources.append(str(source))
    denmark["week8_family_id"] = family_ids
    denmark["week8_family_source"] = family_sources

    unassigned = set(
        denmark.loc[denmark["week8_family_id"].eq(""), "graph_node_index"].astype(int).tolist()
    )
    components = _hosted_on_components(bundle, unassigned)
    graph_index_to_row = {
        int(row.graph_node_index): int(row.Index)
        for row in denmark[["graph_node_index"]].itertuples()
    }
    for component_id, component in enumerate(components, start=1):
        sorted_rows = sorted(graph_index_to_row[idx] for idx in component if idx in graph_index_to_row)
        if len(sorted_rows) < 2:
            continue
        family_id = f"hosted_on_component_{component_id:02d}"
        for row_idx in sorted_rows:
            if denmark.at[row_idx, "week8_family_id"]:
                continue
            denmark.at[row_idx, "week8_family_id"] = family_id
            denmark.at[row_idx, "week8_family_source"] = "hosted_on_component"

    for row_idx, row in denmark[denmark["week8_family_id"].eq("")].iterrows():
        root = _normalise_domain(row["root_domain"])
        denmark.at[row_idx, "week8_family_id"] = f"singleton::{root}"
        denmark.at[row_idx, "week8_family_source"] = "singleton"

    sizes = denmark["week8_family_id"].value_counts().to_dict()
    denmark["week8_family_size"] = denmark["week8_family_id"].map(lambda value: int(sizes[value]))
    denmark["week8_lofo_eligible"] = denmark["week8_family_size"] >= int(
        bundle.config.get("E3_lofo", {}).get("min_family_size_for_lofo", 2)
    )
    return denmark.sort_values(["week8_family_id", "root_domain"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Week 8 Denmark family assignments.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "week8.yaml"))
    parser.add_argument(
        "--output",
        default="output/week8/audits/E3_dk_family_assignments.csv",
        help="Workspace-relative output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_graph_bundle(args.config)
    assignments = extract_denmark_families(bundle)
    output_path = Path(bundle.config["workspace_root"]) / args.output
    save_dataframe(assignments, output_path)
    print(assignments["week8_family_id"].value_counts().to_string())


if __name__ == "__main__":
    main()
