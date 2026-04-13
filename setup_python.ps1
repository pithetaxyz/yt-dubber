# setup_python.ps1
# Installs Python 3.12 via winget and adds it to the user PATH.
# Run in PowerShell (no admin required).

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── 1. Install Python 3.12 ────────────────────────────────────────────────────
Write-Host "`n[1/4] Installing Python 3.12 via winget..." -ForegroundColor Cyan
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
    # -1978335189 = APPINSTALLER_ERROR_ALREADY_INSTALLED, safe to ignore
    Write-Error "winget failed with exit code $LASTEXITCODE"
    exit 1
}

# ── 2. Locate the real executable via the py launcher ────────────────────────
Write-Host "`n[2/4] Locating Python 3.12 executable..." -ForegroundColor Cyan
try {
    $pyExe = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
} catch {
    Write-Error "py launcher not found. Try reopening the terminal and running this script again."
    exit 1
}

if (-not $pyExe -or -not (Test-Path $pyExe)) {
    Write-Error "Could not locate Python 3.12 executable."
    exit 1
}

$pyDir     = Split-Path $pyExe
$scriptsDir = Join-Path $pyDir "Scripts"
Write-Host "  Found: $pyExe"

# ── 3. Add to user PATH if not already present ────────────────────────────────
Write-Host "`n[3/4] Updating user PATH..." -ForegroundColor Cyan
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
$toAdd = @($pyDir, $scriptsDir) | Where-Object { $currentPath -notlike "*$_*" }

if ($toAdd.Count -eq 0) {
    Write-Host "  PATH already contains Python directories, skipping."
} else {
    $newPath = ($currentPath.TrimEnd(";") + ";" + ($toAdd -join ";")).TrimStart(";")
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "  Added to PATH:"
    $toAdd | ForEach-Object { Write-Host "    $_" }
}

# ── 4. Verify ─────────────────────────────────────────────────────────────────
Write-Host "`n[4/4] Verification (in this process)..." -ForegroundColor Cyan
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "Machine")

$ver = & "$pyExe" --version
Write-Host "  $ver"
Write-Host "  Executable: $pyExe"

Write-Host "`nDone! Open a new terminal, then run:" -ForegroundColor Green
Write-Host "  python --version"
Write-Host "  pip --version"
