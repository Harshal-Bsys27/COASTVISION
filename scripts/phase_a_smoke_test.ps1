# Phase A - end-to-end smoke test for CoastVision
# Run with backend already started: .\run_backend.ps1

param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root "venv\Scripts\python.exe"
$script = Join-Path $root "scripts\phase_a_smoke_test.py"

if (-not (Test-Path $python)) {
    Write-Host "FAIL: venv not found at $python" -ForegroundColor Red
    Write-Host "Create it: python -m venv venv; pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

$weights = @(
    (Join-Path $root "yolov8n.pt"),
    (Join-Path $root "models\yolov8n.pt")
) | Where-Object { Test-Path $_ }

if ($weights.Count -eq 0) {
    Write-Host "FAIL: No yolov8n.pt found in project root or models/" -ForegroundColor Red
    exit 1
}

Write-Host "Phase A smoke test - $BaseUrl" -ForegroundColor Cyan
Write-Host "Model weights: $($weights[0])" -ForegroundColor Cyan

& $python $script $BaseUrl
$code = $LASTEXITCODE

if ($code -eq 0) {
    Write-Host "Phase A automated checks: PASSED" -ForegroundColor Green
} else {
    Write-Host "Phase A automated checks: FAILED" -ForegroundColor Red
}

exit $code
