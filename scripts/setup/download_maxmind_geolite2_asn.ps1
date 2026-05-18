param(
    [string]$LicenseKey = $env:MAXMIND_LICENSE_KEY,
    [string]$OutputDir = "<workspace_root_v2>\data\external\maxmind"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($LicenseKey)) {
    throw "Set MAXMIND_LICENSE_KEY or pass -LicenseKey. GeoLite2-ASN requires a free MaxMind account and license key."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$tmp = Join-Path $OutputDir "GeoLite2-ASN.tar.gz"
$url = "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-ASN&license_key=$LicenseKey&suffix=tar.gz"

Invoke-WebRequest -Uri $url -OutFile $tmp -TimeoutSec 300
tar -xzf $tmp -C $OutputDir

$db = Get-ChildItem -Path $OutputDir -Recurse -Filter "GeoLite2-ASN.mmdb" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $db) {
    throw "Download completed but GeoLite2-ASN.mmdb was not found."
}

Copy-Item -LiteralPath $db.FullName -Destination (Join-Path $OutputDir "GeoLite2-ASN.mmdb") -Force

$hash = Get-FileHash -LiteralPath (Join-Path $OutputDir "GeoLite2-ASN.mmdb") -Algorithm SHA256
$hash.Hash | Set-Content -LiteralPath (Join-Path $OutputDir "GeoLite2-ASN.sha256") -Encoding ASCII

Write-Output "GeoLite2-ASN ready: $(Join-Path $OutputDir 'GeoLite2-ASN.mmdb')"
