param(
    [string]$CondaExe = "<user_home>\anaconda3\Scripts\conda.exe",
    [string]$EnvName = "hetero-transfer-v2",
    [string]$Workspace = "<workspace_root>"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "Conda executable not found: $CondaExe"
}

$buildScript = Join-Path $Workspace "src\data\build_hetero_graph.py"
$verifyScript = Join-Path $Workspace "scripts\setup\verify_pyg_environment.py"
$sanityScript = Join-Path $Workspace "src\data\sanity_check.py"

& $CondaExe run -n $EnvName python $verifyScript
& $CondaExe run -n $EnvName python $buildScript
& $CondaExe run -n $EnvName python $sanityScript

Write-Output "week3_conda_build_complete"
