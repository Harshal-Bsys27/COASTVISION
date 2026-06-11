# Phase B–E smoke test — requires backend running on port 8000
param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Phase B–E smoke test against $BaseUrl" -ForegroundColor Cyan
python "$PSScriptRoot\phase_b_e_smoke_test.py" $BaseUrl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Phase B–E smoke test passed." -ForegroundColor Green
