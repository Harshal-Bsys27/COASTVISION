$out='C:\Users\HARSHAL BARHATE\OneDrive\Desktop\COASTVISION\tools'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$url='https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
$dest=Join-Path $out 'cloudflared.exe'

Write-Output "Downloading cloudflared to $dest..."
Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
Write-Output "Downloaded: $dest"

# Show version
& $dest --version
