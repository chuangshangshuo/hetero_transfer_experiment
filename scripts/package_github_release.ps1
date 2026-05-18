param(
    [string]$SourceRoot = "<workspace_root>",
    [string]$RepoRoot = "<release_root>",
    [string]$ReportPackage = "<user_home>\Desktop\files.zip"
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath([string]$PathValue) {
    if (Test-Path -LiteralPath $PathValue) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    $parent = Split-Path -Parent $PathValue
    $leaf = Split-Path -Leaf $PathValue
    $resolvedParent = (Resolve-Path -LiteralPath $parent).Path
    return (Join-Path $resolvedParent $leaf)
}

function Ensure-Directory([string]$PathValue) {
    if (-not (Test-Path -LiteralPath $PathValue)) {
        New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
    }
}

function Copy-OneFile([string]$RelativePath, [string]$DestinationRelativePath = $null) {
    $src = Join-Path $SourceRootFull $RelativePath
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        return
    }
    $destRel = if ($DestinationRelativePath) { $DestinationRelativePath } else { $RelativePath }
    $dest = Join-Path $RepoRootFull $destRel
    Ensure-Directory (Split-Path -Parent $dest)
    Copy-Item -LiteralPath $src -Destination $dest -Force
}

function Copy-TreeByExtension([string]$RelativeDir, [string[]]$Extensions, [string]$DestinationRelativeDir = $null) {
    $srcDir = Join-Path $SourceRootFull $RelativeDir
    if (-not (Test-Path -LiteralPath $srcDir -PathType Container)) {
        return
    }
    $destRelDir = if ($DestinationRelativeDir) { $DestinationRelativeDir } else { $RelativeDir }
    $files = Get-ChildItem -LiteralPath $srcDir -Recurse -File -Force | Where-Object {
        $Extensions -contains $_.Extension.ToLowerInvariant() -and
        $_.FullName -notmatch '\\__pycache__\\' -and
        $_.FullName -notmatch '\\tmp\\' -and
        $_.FullName -notmatch '\\logs\\' -and
        $_.FullName -notmatch '\\runs\\' -and
        $_.FullName -notmatch '\\checkpoints\\' -and
        $_.FullName -notmatch '\\predictions\\' -and
        $_.Name -notmatch '\.pyc$'
    }
    foreach ($file in $files) {
        $rel = $file.FullName.Substring($srcDir.Length).TrimStart('\')
        $dest = Join-Path (Join-Path $RepoRootFull $destRelDir) $rel
        Ensure-Directory (Split-Path -Parent $dest)
        Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
    }
}

function Copy-OutputLeaf([string]$Week, [string]$LeafName) {
    $relative = Join-Path (Join-Path "output" $Week) $LeafName
    Copy-TreeByExtension $relative @(".csv", ".json", ".md", ".txt", ".png", ".pdf") (Join-Path (Join-Path "results" $Week) $LeafName)
}

$SourceRootFull = Resolve-FullPath $SourceRoot
$RepoRootFull = Resolve-FullPath $RepoRoot

if ($SourceRootFull -ne "<workspace_root>") {
    throw "Refusing unexpected SourceRoot: $SourceRootFull"
}
if ($RepoRootFull -ne "<release_root>") {
    throw "Refusing unexpected RepoRoot: $RepoRootFull"
}
if (-not (Test-Path -LiteralPath (Join-Path $RepoRootFull ".git") -PathType Container)) {
    throw "RepoRoot must already contain .git: $RepoRootFull"
}

$backupRoot = "<workspace_root_legacy_backup>_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
Ensure-Directory $backupRoot

Get-ChildItem -LiteralPath $RepoRootFull -Force | Where-Object { $_.Name -ne ".git" } | ForEach-Object {
    Move-Item -LiteralPath $_.FullName -Destination (Join-Path $backupRoot $_.Name) -Force
}

foreach ($dir in @(
    "analysis",
    "src",
    "scripts",
    "configs",
    "docs",
    "paper",
    "figs",
    "data",
    "results",
    "reports",
    "reproducibility"
)) {
    Ensure-Directory (Join-Path $RepoRootFull $dir)
}

Copy-TreeByExtension "analysis" @(".py", ".md")
Copy-TreeByExtension "src" @(".py", ".md")
Copy-TreeByExtension "scripts" @(".ps1", ".py", ".md")
Copy-TreeByExtension "configs" @(".yaml", ".yml", ".txt")
Copy-TreeByExtension "docs" @(".md", ".txt", ".csv")
Copy-TreeByExtension "paper" @(".md")
Copy-TreeByExtension "figs" @(".pdf", ".png", ".md")
Copy-TreeByExtension "source_materials\project" @(".md") "reproducibility/project_logs"

Copy-OneFile "README.md" "docs/workspace_README.md"
Copy-OneFile "MANIFEST.md" "docs/workspace_MANIFEST.md"
Copy-OneFile "MANIFEST_file_inventory.csv" "docs/workspace_MANIFEST_file_inventory.csv"
Copy-OneFile ".env.example" ".env.example"

foreach ($file in @(
    "data\node_feature_schema.json",
    "data\edge_schema.json",
    "data\node_stats.csv",
    "data\edge_stats.csv",
    "data\graphs\hetero_graph_v2_metadata.json",
    "data\versions\feature_builder_state.json",
    "data\versions\hetero_graph_v2.sha256",
    "data\versions\pyg_environment_verification.json",
    "data\versions\version_log.md",
    "data\external\maxmind\README.md",
    "data\external\maxmind\GeoLite2-ASN.sha256"
)) {
    Copy-OneFile $file
}

foreach ($week in @("week4", "week5", "week6", "week6_head_ablation", "week7", "week8", "week8_patch", "week85", "week9")) {
    foreach ($leaf in @("metrics", "tables", "plots", "figures", "audits")) {
        Copy-OutputLeaf $week $leaf
    }
}
foreach ($week in @("week10", "week11")) {
    foreach ($leaf in @("metrics", "tables", "plots", "figures", "audits")) {
        Copy-OutputLeaf $week $leaf
    }
}

$week11AuditDir = Join-Path $RepoRootFull "results\week11\audits"
    if (Test-Path -LiteralPath $week11AuditDir) {
    Get-ChildItem -LiteralPath $week11AuditDir -File -Force | Where-Object {
        $_.Name -match '^P3_fewshot_.*_split\.csv$' -or $_.Name -eq "P3_oneclass_support_samples.csv"
    } | Remove-Item -Force
}

$sampleLevelPatterns = @(
    "cross_verified_eval_prediction_rows.csv",
    "W9_fewshot_support_samples.csv",
    "fewshot_raw_runs.csv",
    "fewshot_summary.csv",
    "fewshot_acceptance_audit.csv"
)
foreach ($pattern in $sampleLevelPatterns) {
    Get-ChildItem -LiteralPath (Join-Path $RepoRootFull "results") -Recurse -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $pattern } |
        Remove-Item -Force
}

foreach ($supersededDraft in @(
    "paper\draft\sec4_3_week7_transfer.md",
    "paper\draft\sec4_4_week8_mechanism.md"
)) {
    $target = Join-Path $RepoRootFull $supersededDraft
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Force
    }
}

Copy-OneFile "output\final_artifact_audit.csv" "results\final_artifact_audit.csv"
Copy-TreeByExtension "output\doc" @(".docx") "reports"

$packageDir = Join-Path $SourceRootFull "tmp\week11_report_package"
if ((-not (Test-Path -LiteralPath (Join-Path $packageDir "Week11_Final_Experiment_Report.md"))) -and (Test-Path -LiteralPath $ReportPackage)) {
    Ensure-Directory $packageDir
    Expand-Archive -LiteralPath $ReportPackage -DestinationPath $packageDir -Force
}
if (Test-Path -LiteralPath (Join-Path $packageDir "Week11_Final_Experiment_Report.md")) {
    Ensure-Directory (Join-Path $RepoRootFull "reports\week11_final")
    Copy-Item -LiteralPath (Join-Path $packageDir "Week11_Final_Experiment_Report.md") -Destination (Join-Path $RepoRootFull "reports\week11_final\Week11_Final_Experiment_Report.md") -Force
    $figureDest = Join-Path $RepoRootFull "reports\week11_final\figs"
    Ensure-Directory $figureDest
    Get-ChildItem -LiteralPath $packageDir -Filter "fig*.png" -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $figureDest $_.Name) -Force
    }
}

$manifestRows = Get-ChildItem -LiteralPath $RepoRootFull -Recurse -File -Force |
    Where-Object { $_.FullName -notmatch '\\.git\\' } |
    Sort-Object FullName |
    ForEach-Object {
        [PSCustomObject]@{
            path = $_.FullName.Substring($RepoRootFull.Length).TrimStart('\')
            bytes = $_.Length
            last_write_time = $_.LastWriteTime.ToString("s")
        }
    }

$manifestPath = Join-Path $RepoRootFull "MANIFEST_RELEASE.csv"
$manifestRows | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8

Write-Host "Packaged GitHub release folder: $RepoRootFull"
Write-Host "Legacy backup: $backupRoot"
Write-Host "Manifest: $manifestPath"
