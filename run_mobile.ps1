param(
    [switch]$Android
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$MobileDir = Join-Path $Root "frontend\mobile"

if (-not (Test-Path (Join-Path $MobileDir "package.json"))) {
    Write-Error "Mobile app not found at $MobileDir"
}

function Stop-PortListener {
    param([int]$Port)
    $lines = netstat -ano | Select-String ":$Port\s"
    foreach ($line in $lines) {
        if ($line -match "LISTENING\s+(\d+)$") {
            $procId = [int]$Matches[1]
            if ($procId -gt 0) {
                Write-Host "Stopping process $procId on port $Port..."
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Push-Location $MobileDir
try {
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing mobile dependencies..."
        npm install
    }

    # Stale Metro on 8081/8082 causes Expo Go to hang or load the wrong bundle.
    Stop-PortListener -Port 8081
    Stop-PortListener -Port 8082

    if ($Android) {
        npx expo start --go --android --port 8081
    } else {
        npx expo start --go --port 8081
    }
} finally {
    Pop-Location
}
