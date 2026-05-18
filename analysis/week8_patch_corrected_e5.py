from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain.structure_vs_lexical import make_split_group_overlap_audit
from src.train.utils import load_graph_bundle, save_dataframe


OUT = ROOT / "output" / "week8_patch"


def _load_raw() -> pd.DataFrame:
    path = OUT / "metrics" / "E5_structure_vs_lexical.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_delta_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for transfer_id, part in raw.groupby("transfer_id"):
        full = part[part["config_id"] == "C1_full"].set_index("seed")["primary_metric_value"]
        direction = str(part["primary_metric_direction"].dropna().iloc[0])
        for config_id, cfg in part.groupby("config_id"):
            values = cfg.set_index("seed")["primary_metric_value"]
            aligned = pd.concat([full.rename("full"), values.rename("config")], axis=1).dropna()
            if aligned.empty:
                continue
            raw_delta = aligned["config"] - aligned["full"]
            if direction == "lower_is_better":
                effect = raw_delta
            else:
                effect = aligned["full"] - aligned["config"]
            rows.append(
                {
                    "transfer_id": transfer_id,
                    "config_id": config_id,
                    "primary_metric_direction": direction,
                    "full_mean": float(aligned["full"].mean()),
                    "config_mean": float(aligned["config"].mean()),
                    "delta_config_minus_full": float(raw_delta.mean()),
                    "effect_full_minus_config_when_higher_is_better": float(effect.mean()),
                    "n_pairs": int(len(aligned)),
                    "is_posthoc": True,
                }
            )
    return pd.DataFrame(rows)


def build_acceptance(delta: pd.DataFrame) -> pd.DataFrame:
    t3 = delta[delta["transfer_id"] == "T3_DiagnoseFrance"].set_index("config_id")
    rows = []
    def value(config_id: str, column: str) -> float:
        return float(t3.loc[config_id, column])

    no_cctld_drop = value("C2_no_cctld", "effect_full_minus_config_when_higher_is_better")
    zero_website_auc = value("C3_no_website_lexical", "config_mean")
    graph_only_auc = value("C4_graph_only", "config_mean")
    lex_only_auc = value("C5_lex_only", "config_mean")
    no_edges_auc = value("C6_no_edges", "config_mean")
    no_cert_drop = value("C7_no_cert_edge", "effect_full_minus_config_when_higher_is_better")
    no_reg_drop = value("C8_no_registrar_edge", "effect_full_minus_config_when_higher_is_better")
    infra_auc = value("C9_infra_edges_only", "config_mean")

    rows.append(
        {
            "hypothesis": "W8P_E5_ccTLD_sensitivity",
            "observed_metric": "T3_full_minus_no_cctld_auc",
            "observed_value": no_cctld_drop,
            "interpretation": "ccTLD/local-TLD removal remains a material post-hoc diagnostic effect",
            "status": "diagnostic_support" if no_cctld_drop >= 0.05 else "weak_effect",
            "is_posthoc": True,
        }
    )
    rows.append(
        {
            "hypothesis": "W8P_E5_website_lexical_removed",
            "observed_metric": "T3_C3_no_website_lexical_auc",
            "observed_value": zero_website_auc,
            "interpretation": "zeroing Website lexical features substantially weakens T3",
            "status": "diagnostic_support" if zero_website_auc < 0.85 else "weak_effect",
            "is_posthoc": True,
        }
    )
    rows.append(
        {
            "hypothesis": "W8P_E5_lex_only_vs_graph_only",
            "observed_metric": "T3_lex_only_minus_graph_only_auc",
            "observed_value": lex_only_auc - graph_only_auc,
            "interpretation": "Website lexical-only outperforms graph-only on T3",
            "status": "diagnostic_support" if lex_only_auc > graph_only_auc else "opposite",
            "is_posthoc": True,
        }
    )
    rows.append(
        {
            "hypothesis": "W8P_E5_no_edges_equivalent_to_lex_only",
            "observed_metric": "T3_no_edges_minus_lex_only_auc",
            "observed_value": no_edges_auc - lex_only_auc,
            "interpretation": "C6 no_edges checks that lex-only semantics match empty-edge behavior",
            "status": "consistent" if abs(no_edges_auc - lex_only_auc) < 1e-9 else "check",
            "is_posthoc": True,
        }
    )
    rows.append(
        {
            "hypothesis": "W8P_E5_edge_sensitivity",
            "observed_metric": "max_T3_drop_no_cert_or_no_registrar",
            "observed_value": max(no_cert_drop, no_reg_drop),
            "interpretation": "certificate/registrar edges have post-hoc diagnostic contribution",
            "status": "diagnostic_support" if max(no_cert_drop, no_reg_drop) >= 0.05 else "weak_effect",
            "is_posthoc": True,
        }
    )
    rows.append(
        {
            "hypothesis": "W8P_E5_infra_only",
            "observed_metric": "T3_infra_only_auc",
            "observed_value": infra_auc,
            "interpretation": "infra-only removes referenced_by and matches full when ExtRef is not needed",
            "status": "diagnostic_support" if infra_auc >= 0.95 else "weak_effect",
            "is_posthoc": True,
        }
    )
    return pd.DataFrame(rows)


def build_split_audit(bundle: Any, raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, run in raw[["transfer_id", "config_id", "seed"]].drop_duplicates().iterrows():
        split_path = ROOT / "output" / "week7" / "splits" / f"{run.transfer_id}__seed{int(run.seed)}.csv"
        split_frame = pd.read_csv(split_path)
        rows.append(make_split_group_overlap_audit(split_frame, str(run.config_id), str(run.transfer_id), int(run.seed)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def patch_manifest_audit_reference() -> None:
    for path in (OUT / "runs").glob("E5_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["is_posthoc"] = True
        payload["split_group_overlap_audit"] = "E5_split_group_overlap_audit.csv"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    bundle = load_graph_bundle(ROOT / "configs" / "week8.yaml")
    raw = _load_raw()
    delta = build_delta_table(raw)
    save_dataframe(delta, OUT / "metrics" / "E5_corrected_delta_vs_full.csv")
    acceptance = build_acceptance(delta)
    save_dataframe(acceptance, OUT / "metrics" / "E5_corrected_acceptance_audit.csv")
    split_audit = build_split_audit(bundle, raw)
    save_dataframe(split_audit, OUT / "audits" / "E5_split_group_overlap_audit.csv")
    patch_manifest_audit_reference()
    print(acceptance.to_string(index=False))
    print()
    print(split_audit.groupby(["transfer_id", "domain_role"])["num_overlapping_groups"].max().reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
