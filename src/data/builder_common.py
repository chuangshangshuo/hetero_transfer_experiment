from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "source_materials" / "data"
OUT_ROOT = ROOT / "data"
PROCESSED_ROOT = OUT_ROOT / "processed"
NODE_ROOT = PROCESSED_ROOT / "nodes"
EDGE_ROOT = PROCESSED_ROOT / "edges"
GRAPH_ROOT = OUT_ROOT / "graphs"
VERSION_ROOT = OUT_ROOT / "versions"
AUDIT_ROOT = OUT_ROOT / "audits"
MAXMIND_DB = OUT_ROOT / "external" / "maxmind" / "GeoLite2-ASN.mmdb"
LOCAL_DEPS = ROOT / ".deps" / "python"

if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))


MULTIPART_SUFFIXES = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "com.br",
    "com.cn",
    "com.co",
    "com.hk",
    "com.mx",
    "com.my",
    "com.pe",
    "com.ph",
    "com.pl",
    "com.sg",
    "com.tr",
    "com.tw",
    "co.il",
    "co.jp",
    "co.kr",
    "co.nz",
    "co.za",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def short_hash(value: str, n: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_text_token(value: Any) -> str:
    text = clean_value(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_as_bool(value: Any) -> bool:
    return clean_value(value).lower() in {"1", "true", "yes", "success"}


def _normalize_url_like(value: Any) -> str:
    text = clean_value(value).lower()
    if not text:
        return ""
    text = (
        text.replace("[.]", ".")
        .replace("(.)", ".")
        .replace("hxxps://", "https://")
        .replace("hxxp://", "http://")
    )
    for prefix in ("domain::", "host::", "url::"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip(" .,:;|()[]{}<>\"'")


def _extract_host(value: Any, *, strip_port: bool) -> str:
    text = _normalize_url_like(value)
    if not text:
        return ""
    if "://" in text:
        parsed = urlparse(text)
    else:
        parsed = urlparse(f"http://{text}")
    host = parsed.netloc or parsed.path.split("/")[0]
    host = host.split("@")[-1]
    host = host.lstrip("*.").removeprefix("www.")
    if strip_port:
        host = host.rsplit(":", 1)[0]
        host = re.sub(r"[^a-z0-9.-]", "", host)
    else:
        host = re.sub(r"[^a-z0-9.:-]", "", host)
    host = host.strip(".")
    if not host or "." not in host:
        return ""
    return host


def normalize_domain(value: Any) -> str:
    return _extract_host(value, strip_port=True)


def coarse_redirect_host(value: Any) -> str:
    return _extract_host(value, strip_port=False)


def coarse_redirect_site_with_port(value: Any) -> str:
    host = coarse_redirect_host(value)
    if not host:
        return ""
    host_without_port = host
    port = ""
    if host.count(":") == 1:
        candidate_host, candidate_port = host.rsplit(":", 1)
        if candidate_port.isdigit():
            host_without_port = candidate_host
            port = candidate_port
    site = registrable_domain(host_without_port)
    if not site:
        return ""
    return f"{site}:{port}" if port else site


def registrable_domain(value: Any) -> str:
    host = normalize_domain(value)
    if not host:
        return ""
    labels = host.split(".")
    if len(labels) < 2:
        return host
    suffix2 = ".".join(labels[-2:])
    if suffix2 in MULTIPART_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def domain_features(domain: str) -> dict[str, float]:
    root = domain.split(".")[0] if domain else ""
    suffix = domain.split(".")[-1] if "." in domain else ""
    return {
        "domain_len": float(len(domain)),
        "root_len": float(len(root)),
        "domain_segment_count": float(domain.count(".") + 1 if domain else 0),
        "digit_count": float(sum(ch.isdigit() for ch in domain)),
        "hyphen_count": float(domain.count("-")),
        "is_dot_com": float(suffix == "com"),
        "is_local_tld": float(suffix in {"fr", "de", "it", "be", "uk", "ca", "es", "se", "ph", "dk", "nl"}),
    }


def parse_san_values(value: Any) -> list[str]:
    text = clean_value(value)
    if not text:
        return []
    raw_parts = re.split(r"[|,;]+", text)
    values: list[str] = []
    for raw in raw_parts:
        token = clean_value(raw)
        if token:
            values.append(token)
    return values


def stable_top_categories(values: pd.Series, max_size: int) -> list[str]:
    cleaned = values.map(clean_value)
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return []

    def sort_key(item: tuple[str, int]) -> tuple[int, int | str]:
        value, count = item
        secondary: int | str = int(value) if value.isdigit() else value
        return (-count, secondary)

    counts = cleaned.value_counts().to_dict()
    ordered = sorted(counts.items(), key=sort_key)
    return [value for value, _ in ordered[:max_size]]


def zscore(values: pd.Series, epsilon: float) -> tuple[np.ndarray, dict[str, float | bool]]:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if arr.size == 0:
        return arr, {"mean": 0.0, "std": 0.0, "epsilon": epsilon, "dropped": True}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std <= epsilon:
        return np.zeros_like(arr, dtype=float), {"mean": mean, "std": std, "epsilon": epsilon, "dropped": True}
    return (arr - mean) / (std + epsilon), {"mean": mean, "std": std, "epsilon": epsilon, "dropped": False}


def maybe_add_feature_column(
    frame: pd.DataFrame,
    feature_name: str,
    values: np.ndarray,
    active_columns: list[str],
    dropped_columns: list[dict[str, Any]],
    *,
    source_column: str,
    feature_kind: str,
    drop_all_zero: bool = True,
) -> None:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) != len(frame):
        raise ValueError(f"Feature {feature_name} has invalid shape {arr.shape} for frame of length {len(frame)}.")
    if drop_all_zero and (arr.size == 0 or np.allclose(arr, 0.0)):
        dropped_columns.append(
            {
                "feature_name": feature_name,
                "source_column": source_column,
                "feature_kind": feature_kind,
                "drop_reason": "all_zero",
            }
        )
        return
    frame[feature_name] = arr.astype(float)
    active_columns.append(feature_name)


def assert_no_all_zero_columns(matrix: np.ndarray, feature_columns: list[str], node_type: str) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"{node_type}.x must be 2D, got shape={matrix.shape}.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{node_type}.x has zero active feature columns.")
    zero_columns = [feature_columns[idx] for idx in range(matrix.shape[1]) if np.allclose(matrix[:, idx], 0.0)]
    if zero_columns:
        raise ValueError(f"{node_type}.x still contains all-zero columns: {zero_columns}")


def join_unique(values: pd.Series) -> str:
    items = sorted({clean_value(v) for v in values.tolist() if clean_value(v)})
    return "|".join(items)


def edge_count_to_weight(count: int) -> float:
    return float(np.log1p(count))


def aggregate_edge_pairs(
    raw_df: pd.DataFrame,
    edge_type: str,
    src_index_map: dict[str, int],
    dst_index_map: dict[str, int],
) -> pd.DataFrame:
    columns = [
        "edge_id",
        "src_id",
        "dst_id",
        "src_index",
        "dst_index",
        "edge_count_raw",
        "edge_weight_base_log1p",
        "edge_weight",
        "source_dataset_set",
    ]
    if raw_df.empty:
        return pd.DataFrame(columns=columns)

    work = raw_df.copy()
    work["src_id"] = work["src_id"].map(clean_value)
    work["dst_id"] = work["dst_id"].map(clean_value)
    work = work[(work["src_id"] != "") & (work["dst_id"] != "")].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        work.groupby(["src_id", "dst_id"], sort=True, dropna=False)
        .agg(edge_count_raw=("src_id", "size"), source_dataset_set=("source_dataset", join_unique))
        .reset_index()
    )
    grouped["src_index"] = grouped["src_id"].map(src_index_map)
    grouped["dst_index"] = grouped["dst_id"].map(dst_index_map)
    grouped = grouped.dropna(subset=["src_index", "dst_index"]).copy()
    grouped["src_index"] = grouped["src_index"].astype(int)
    grouped["dst_index"] = grouped["dst_index"].astype(int)
    grouped["edge_weight_base_log1p"] = grouped["edge_count_raw"].map(edge_count_to_weight)
    grouped["edge_weight"] = grouped["edge_weight_base_log1p"]
    grouped = grouped.sort_values(["src_index", "dst_index", "src_id", "dst_id"]).reset_index(drop=True)
    grouped.insert(0, "edge_id", [f"{edge_type.upper()}_{idx:07d}" for idx in range(1, len(grouped) + 1)])
    return grouped[columns]


def archive_file(path: Path, archive_root: Path) -> Path | None:
    if not path.exists():
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / path.name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.move(str(path), str(target))
    return target


class NodeStore:
    def __init__(self, prefix: str, value_column: str) -> None:
        self.prefix = prefix
        self.value_column = value_column
        self.value_to_id: dict[str, str] = {}
        self.rows: list[dict[str, Any]] = []

    def get(self, value: Any, **attrs: Any) -> str:
        cleaned = clean_value(value).lower()
        if not cleaned:
            return ""
        if cleaned not in self.value_to_id:
            node_id = f"{self.prefix}_{short_hash(cleaned)}"
            row = {"node_id": node_id, self.value_column: cleaned}
            row.update(attrs)
            self.value_to_id[cleaned] = node_id
            self.rows.append(row)
        else:
            node_id = self.value_to_id[cleaned]
            for row in self.rows:
                if row["node_id"] == node_id:
                    for key, val in attrs.items():
                        if key not in row or clean_value(row.get(key)) == "":
                            row[key] = val
                    break
        return self.value_to_id[cleaned]

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        return pd.DataFrame(self.rows).sort_values("node_id").reset_index(drop=True)


def load_maxmind_reader() -> Any:
    if not MAXMIND_DB.exists():
        return None
    try:
        import maxminddb

        return maxminddb.open_database(str(MAXMIND_DB))
    except Exception:
        return None


def lookup_asn(reader: Any, ip_value: str) -> tuple[str, str, str, str]:
    try:
        ip_obj = ipaddress.ip_address(ip_value)
    except ValueError:
        return "", "", "0", "0"
    is_private = "1" if ip_obj.is_private else "0"
    is_global = "1" if ip_obj.is_global else "0"
    if reader is None:
        return "", "", is_private, is_global
    try:
        hit = reader.get(ip_value) or {}
    except Exception:
        hit = {}
    asn = clean_value(hit.get("autonomous_system_number", ""))
    org = clean_value(hit.get("autonomous_system_organization", ""))
    return asn, org, is_private, is_global
