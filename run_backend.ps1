param(
  [switch]$Detach,
  [switch]$Stop
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "venv\Scripts\python.exe"

if (!(Test-Path $python)) {
  Write-Host "venv not found at: $python" -ForegroundColor Red
  Write-Host "Create it first: python -m venv venv" -ForegroundColor Yellow
  exit 1
}

$env:COASTVISION_VIDEO_DIR = Join-Path $root "frontend\dashboard\videos"

# Default to GPU when available (override by setting COASTVISION_DEVICE yourself)
if (-not $env:COASTVISION_DEVICE) { $env:COASTVISION_DEVICE = "cuda:0" }
if (-not $env:COASTVISION_HALF) { $env:COASTVISION_HALF = "1" }

if (-not $env:COASTVISION_MAX_SIDE) { $env:COASTVISION_MAX_SIDE = "960" }
if (-not $env:COASTVISION_IMGSZ) { $env:COASTVISION_IMGSZ = "640" }
if (-not $env:COASTVISION_FPS) { $env:COASTVISION_FPS = "12" }
if (-not $env:COASTVISION_INFER_EVERY) { $env:COASTVISION_INFER_EVERY = "2" }

# Smooth/clean defaults (can be tuned per machine)
if (-not $env:COASTVISION_DET_HOLD_S) { $env:COASTVISION_DET_HOLD_S = "0.9" }

Write-Host "Backend videos: $env:COASTVISION_VIDEO_DIR" -ForegroundColor Cyan
Write-Host "Starting backend on http://127.0.0.1:8000" -ForegroundColor Cyan

Set-Location $root

if ($Stop) {
  $logDir = Join-Path $root "data\logs"
  $jobFile = Join-Path $logDir "backend.job.txt"
  $pidFile = Join-Path $logDir "backend.pid.txt"

  # Prefer PID-based stop (newer, more reliable)
  if (Test-Path $pidFile) {
    try {
      $pidText = Get-Content $pidFile -ErrorAction SilentlyContinue
      if ($pidText) {
        Stop-Process -Id ([int]$pidText) -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped backend process PID $pidText" -ForegroundColor Yellow
      }
    } catch {
      # ignore
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }

  if (Test-Path $jobFile) {
    $jid = Get-Content $jobFile -ErrorAction SilentlyContinue
    if ($jid) {
      Stop-Job -Id $jid -ErrorAction SilentlyContinue
      Remove-Job -Id $jid -ErrorAction SilentlyContinue
      Write-Host "Stopped backend job $jid" -ForegroundColor Yellow
    }
    Remove-Item $jobFile -ErrorAction SilentlyContinue
  } else {
    Write-Host "No backend.job.txt found (nothing to stop)." -ForegroundColor Yellow
  }

  # Also kill any leftover python processes still bound to port 8000.
  try {
    $pids = @()
    $lines = netstat -ano | Select-String ":8000" | Select-String "LISTENING" | ForEach-Object { $_.ToString() }
    foreach ($ln in $lines) {
      $parts = ($ln -split "\s+") | Where-Object { $_ -ne "" }
      if ($parts.Count -ge 5) { $pids += $parts[$parts.Count - 1] }
    }
    $pids = $pids | Select-Object -Unique
    foreach ($procId in $pids) {
      Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue
    }
    if ($pids.Count -gt 0) {
      Write-Host "Killed listeners on :8000 (PIDs: $($pids -join ', '))" -ForegroundColor Yellow
    }
  } catch {
    # ignore
  }
  exit 0
}

if ($Detach) {
  $args = @(
    "-m", "waitress",
    "--listen=127.0.0.1:8000",
    "--threads=32",
    "backend.server:app"
  )

  $logDir = Join-Path $root "data\logs"
  if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
  $outLog = Join-Path $logDir "backend.out.log"
  $errLog = Join-Path $logDir "backend.err.log"

  $p = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  $pidFile = Join-Path $logDir "backend.pid.txt"
  Set-Content -Path $pidFile -Value $p.Id

  Write-Host "Backend started (background process). PID: $($p.Id)" -ForegroundColor Green
  Write-Host "Logs: $outLog , $errLog" -ForegroundColor Green
  Write-Host "Stop: .\\run_backend.ps1 -Stop" -ForegroundColor Yellow
  exit 0
}

& $python -m waitress --listen=127.0.0.1:8000 --threads=32 backend.server:app
