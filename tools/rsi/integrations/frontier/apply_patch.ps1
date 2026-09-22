# Apply the ve_policy bridge patch to a fresh Frontier checkout.
# Usage: .\apply_patch.ps1 -FrontierRepo C:\path\to\Frontier
param([Parameter(Mandatory = $true)][string]$FrontierRepo)

$patch = Join-Path $PSScriptRoot "ve_policy.patch"
if (-not (Test-Path $patch)) { Write-Error "patch not found: $patch"; exit 1 }
if (-not (Test-Path (Join-Path $FrontierRepo "frontier"))) {
    Write-Error "not a Frontier checkout: $FrontierRepo"; exit 1
}
Push-Location $FrontierRepo
git apply --check $patch 2>$null
if ($LASTEXITCODE -eq 0) {
    git apply $patch
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "patch apply failed after a clean check"
        exit 1
    }
    Pop-Location
    Write-Output "ve_policy bridge applied to $FrontierRepo"
    exit 0
}
git apply --check --reverse $patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Pop-Location
    Write-Output "ve_policy bridge already applied to $FrontierRepo"
    exit 0
}
Pop-Location
Write-Error "patch is neither applicable nor fully present; wrong commit or partial patch"
exit 1
