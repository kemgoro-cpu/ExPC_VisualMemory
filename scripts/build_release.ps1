param(
    [ValidateSet("Both", "Lite", "Full")]
    [string]$Profile = "Both",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OutputRoot = "outputs",
    [string]$ModelBundle = "work\model-bundle",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $ProjectRoot
try {
    $Version = & $Python -c "from visual_memory import __version__; print(__version__)"
    if (-not $Version) { throw "Unable to determine the application version" }

    function Build-Profile([string]$Name) {
        $LowerName = $Name.ToLowerInvariant()
        $Target = Join-Path $OutputRoot "external-pc-visual-memory-$LowerName-$Version"
        $Stage = Join-Path "work" "release-$LowerName-$Version"
        if ((Test-Path $Target) -and -not $Rebuild) {
            throw "Release target already exists: $Target (use -Rebuild to replace it)"
        }
        if ((Test-Path $Stage) -and -not $Rebuild) {
            throw "Build stage already exists: $Stage (use -Rebuild to reuse it)"
        }
        if ($Rebuild) {
            $RootPath = [IO.Path]::GetFullPath($ProjectRoot.Path) + [IO.Path]::DirectorySeparatorChar
            foreach ($Generated in @($Target, (Join-Path $Stage "dist"))) {
                if (Test-Path $Generated) {
                    $GeneratedPath = [IO.Path]::GetFullPath($Generated)
                    if (-not $GeneratedPath.StartsWith($RootPath, [StringComparison]::OrdinalIgnoreCase)) {
                        throw "Refusing to replace a path outside the project: $GeneratedPath"
                    }
                    Remove-Item -LiteralPath $GeneratedPath -Recurse -Force
                }
            }
        }

        $env:VISUAL_MEMORY_BUILD_PROFILE = $LowerName
        if ($LowerName -eq "full") {
            if (-not (Test-Path $ModelBundle)) { throw "Model bundle is missing: $ModelBundle" }
            $env:VISUAL_MEMORY_MODEL_BUNDLE = (Resolve-Path $ModelBundle)
        }
        else {
            Remove-Item Env:VISUAL_MEMORY_MODEL_BUNDLE -ErrorAction SilentlyContinue
        }

        $Dist = Join-Path $Stage "dist"
        $Build = Join-Path $Stage "build"
        & $Python -m PyInstaller --noconfirm --distpath $Dist --workpath $Build visual-memory.spec
        & $Python -m PyInstaller --noconfirm --distpath $Dist --workpath $Build visual-memory-mcp.spec

        New-Item -ItemType Directory -Path $Target | Out-Null
        Copy-Item (Join-Path $Dist "visual-memory") $Target -Recurse
        Copy-Item (Join-Path $Dist "visual-memory-mcp") $Target -Recurse
        Copy-Item "README.md" $Target
        Copy-Item "docs\FIRST_RUN.md" $Target
        Copy-Item "docs\MCP_SETUP.md" $Target
        if ($LowerName -eq "full") {
            Copy-Item "scripts\setup_gpu.ps1" $Target
            Copy-Item "scripts\run_gpu.ps1" $Target
            Copy-Item "scripts\benchmark_gpu.ps1" $Target
            $Packages = Join-Path $Target "packages"
            New-Item -ItemType Directory -Path $Packages | Out-Null
            & $Python -m pip wheel . --no-deps --wheel-dir $Packages
        }
        Get-FileHash (Join-Path $Target "visual-memory\visual-memory.exe") -Algorithm SHA256 |
            Select-Object Algorithm, Hash, Path |
            ConvertTo-Json | Set-Content (Join-Path $Target "SHA256.json") -Encoding utf8
        Write-Host "Created $Target"
    }

    if ($Profile -in @("Both", "Lite")) { Build-Profile "Lite" }
    if ($Profile -in @("Both", "Full")) { Build-Profile "Full" }
}
finally {
    Remove-Item Env:VISUAL_MEMORY_BUILD_PROFILE -ErrorAction SilentlyContinue
    Pop-Location
}
