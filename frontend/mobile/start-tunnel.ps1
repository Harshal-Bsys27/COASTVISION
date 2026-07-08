param(
  [int]$Port = 8081,
  [int]$MaxRetries = 5,
  [int]$RetryDelaySeconds = 4
)

$ErrorActionPreference = "Stop"

function Stop-PortProcesses {
  param([int[]]$Ports)

  foreach ($p in $Ports) {
    $conns = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue
    if ($null -eq $conns) { continue }

    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
      try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        Write-Host "[tunnel] Killed PID $procId on port $p"
      } catch {
        Write-Host "[tunnel] Could not kill PID $procId on port $p"
      }
    }
  }
}

Set-Location $PSScriptRoot

# Expo tunnel uses ngrok under the hood. Without auth token, tunnel often fails.
$ngrokConfig = Join-Path $env:USERPROFILE ".ngrok2\ngrok.yml"
$hasNgrokConfig = Test-Path $ngrokConfig
$hasEnvToken = -not [string]::IsNullOrWhiteSpace($env:NGROK_AUTHTOKEN)

if (-not $hasNgrokConfig -and -not $hasEnvToken) {
  Write-Host "[tunnel] ngrok auth token is not configured." -ForegroundColor Red
  Write-Host "[tunnel] Run one of these commands, then retry:" -ForegroundColor Yellow
  Write-Host "  npx ngrok config add-authtoken <YOUR_TOKEN>" -ForegroundColor Yellow
  Write-Host "  ngrok config add-authtoken <YOUR_TOKEN>" -ForegroundColor Yellow
  Write-Host "  `$env:NGROK_AUTHTOKEN = '<YOUR_TOKEN>'" -ForegroundColor Yellow
  exit 1
}

# Clear stale environment variables that can break Expo CLI on Windows.
Remove-Item Env:CI -ErrorAction SilentlyContinue
Remove-Item Env:EXPO_CI -ErrorAction SilentlyContinue

# Clear stale Metro/Expo listeners before launch.
$portsToClear = @($Port, 8082, 8083, 8084, 19000, 19001, 19002)
for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
  Stop-PortProcesses -Ports $portsToClear

  Write-Host "[tunnel] Starting Expo tunnel on port $Port (attempt $attempt/$MaxRetries)"
  & npx expo start --go --tunnel --clear --port $Port
  $exitCode = $LASTEXITCODE

  # If Expo exits cleanly, keep that result.
  if ($exitCode -eq 0) {
    exit 0
  }

  if ($attempt -lt $MaxRetries) {
    Write-Host "[tunnel] Expo exited with code $exitCode. Retrying in $RetryDelaySeconds seconds..."
    Start-Sleep -Seconds $RetryDelaySeconds
  } else {
    Write-Host "[tunnel] Failed to keep tunnel session alive after $MaxRetries attempts."
    exit $exitCode
  }
}
