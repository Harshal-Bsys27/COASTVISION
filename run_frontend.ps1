$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$web = Join-Path $root "frontend\web"

if (!(Test-Path (Join-Path $web "package.json"))) {
  Write-Host "frontend/web/package.json not found at: $web" -ForegroundColor Red
  exit 1
}

Set-Location $web
Write-Host "Starting React dev server..." -ForegroundColor Cyan
npm install
npm run dev
