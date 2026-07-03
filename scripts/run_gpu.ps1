param(
    [ValidateSet("paddle", "yomitoku", "yomitoku-lite")]
    [string]$Provider = "paddle",
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
    $ProjectRoot = $PSScriptRoot
}
$WorkerRoot = $ProjectRoot
$WorkspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if (
    -not (Test-Path (Join-Path $WorkerRoot ".venv-paddle-gpu\Scripts\python.exe")) -and
    (Test-Path (Join-Path $WorkspaceRoot "pyproject.toml"))
) {
    $WorkerRoot = $WorkspaceRoot
}
if ($Provider -eq "paddle") {
    $WorkerPython = Resolve-Path (Join-Path $WorkerRoot ".venv-paddle-gpu\Scripts\python.exe")
}
else {
    $candidate = Join-Path $WorkerRoot ".venv-yomi-gpu\Scripts\python.exe"
    if (-not (Test-Path $candidate)) { $candidate = Join-Path $WorkerRoot ".venv-gpu\Scripts\python.exe" }
    $WorkerPython = Resolve-Path $candidate
}
$env:VISUAL_MEMORY_OCR_PROVIDER = $Provider
$env:VISUAL_MEMORY_OCR_DEVICE = "cuda"
$env:VISUAL_MEMORY_OCR_WORKER_PYTHON = $WorkerPython
if ($DataDir) { $env:VISUAL_MEMORY_DATA_DIR = [System.IO.Path]::GetFullPath($DataDir) }
$MainExecutable = Join-Path $ProjectRoot "visual-memory\visual-memory.exe"
if (Test-Path $MainExecutable) {
    & $MainExecutable
}
else {
    $MainPython = Resolve-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    & $MainPython -m visual_memory
}
