# Build standalone Android APK (no Expo Go required on demo day)
# Prerequisites: Android SDK, USB debugging or emulator, Java JDK
param(
    [switch]$PrebuildOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$mobile = Join-Path $root "frontend\mobile"

Set-Location $mobile
npm install

if ($PrebuildOnly) {
    npx expo prebuild --platform android
    Write-Host "Prebuild complete. Run without -PrebuildOnly to compile APK." -ForegroundColor Green
    exit 0
}

npx expo prebuild --platform android
npx expo run:android --variant release

Write-Host ""
Write-Host "APK build finished. Install the app from the android/app/build output or connected device." -ForegroundColor Green
