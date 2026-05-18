# Heterogeneous Graph Schema v2

## Nodes

- `Website`
- `IP`
- `Certificate`
- `NameServer`
- `Registrar`
- `ExternalReference`

Excluded from the main graph:

- `PromotionAccount`: archived after weak X/Telegram collection yield.
- `App`: sparse jurisdiction-specific evidence only.
- `Brand`: folded into Website attributes.
- `ASN`: folded into IP attributes via MaxMind GeoLite2-ASN.

## Semantic Relations

- `hosted_on`: `Website -> IP`
- `uses_cert`: `Website -> Certificate`
- `uses_ns`: `Website -> NameServer`
- `registered_via`: `Website -> Registrar`
- `referenced_by`: `Website -> ExternalReference`

`redirects_to` is audit-only and does not enter the formal graph.

## Materialized PyG Edge Types

- `hosted_on` / `rev_hosted_on`
- `uses_cert` / `rev_uses_cert`
- `uses_ns` / `rev_uses_ns`
- `registered_via` / `rev_registered_via`
- `referenced_by` / `rev_referenced_by`

## Metapaths

Core:

- `M1 = W-IP-W`
- `M2 = W-Cert-W`
- `M3 = W-NS-W`
- `M4 = W-ExtRef-W`

Extended:

- `M5 = W-Registrar-W`
- `M6 = W-IP-NS-IP-W`
