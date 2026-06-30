# Build standalone Android APK (requires Android SDK + USB device or emulator)
param(
    [switch]$PrebuildOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$mobile = Join-Path $root "frontend\mobile"

Write-Host "CoastVision Android build" -ForegroundColor Cyan
Set-Location $mobile

if (-not (Test-Path "node_modules")) {
    npm install
}

Write-Host "Running expo prebuild..." -ForegroundColor Cyan
npx expo prebuild --platform android
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($PrebuildOnly) {
    Write-Host "Prebuild complete. android/ folder ready." -ForegroundColor Green
    exit 0
}

Write-Host "Building Android app (expo run:android)..." -ForegroundColor Cyan
npx expo run:android
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build finished. Install the APK from android/app/build/outputs/apk/debug/" -ForegroundColor Green
