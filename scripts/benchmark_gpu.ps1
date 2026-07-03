param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,
    [string]$Output = "work\ocr-benchmark"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Tool = Join-Path $PSScriptRoot "benchmark_ocr.py"
if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
    $ProjectRoot = $PSScriptRoot
    $Tool = Join-Path $ProjectRoot "tools\benchmark_ocr.py"
}
$YomiPython = Join-Path $ProjectRoot ".venv-yomi-gpu\Scripts\python.exe"
if (-not (Test-Path $YomiPython)) {
    $YomiPython = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
}
$PaddlePython = Join-Path $ProjectRoot ".venv-paddle-gpu\Scripts\python.exe"
& $YomiPython $Tool ([System.IO.Path]::GetFullPath($Dataset)) `
    --output ([System.IO.Path]::GetFullPath($Output)) `
    --paddle-python $PaddlePython `
    --yomitoku-python $YomiPython
