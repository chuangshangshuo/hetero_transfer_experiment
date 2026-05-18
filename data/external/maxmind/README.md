# MaxMind GeoLite2-ASN

Week 3 graph construction expects:

- `GeoLite2-ASN.mmdb`
- `GeoLite2-ASN.sha256`

The database is not bundled because MaxMind requires a free account and license key.

## Download

1. Create a free MaxMind account.
2. Generate a GeoLite2 license key.
3. Run:

```powershell
$env:MAXMIND_LICENSE_KEY="YOUR_LICENSE_KEY"
powershell -ExecutionPolicy Bypass -File <workspace_root>\scripts\setup\download_maxmind_geolite2_asn.ps1
```

The script downloads the current GeoLite2-ASN tarball, extracts `GeoLite2-ASN.mmdb`, and writes a SHA256 checksum.

## Reproducibility

Record the download date and checksum in `data/versions/version_log.md` before building `hetero_graph_v2.pt`.
