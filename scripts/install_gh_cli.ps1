Param(
  [string]$Version = 'latest',
  [string]$Destination = '.tools/gh',
  [switch]$UpdatePath = $true
)

$ErrorActionPreference = 'Stop'

function Get-ArchSuffix {
  $arch = $env:PROCESSOR_ARCHITECTURE
  switch ($arch) {
    'AMD64' { return 'windows_amd64' }
    'ARM64' { return 'windows_arm64' }
    default { return 'windows_amd64' }
  }
}

function Ensure-Dir($path) {
  if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path | Out-Null }
}

function Get-LatestAssetUrl($suffix) {
  $release = Invoke-RestMethod -Headers @{ 'User-Agent'='PolliLib-Agent' } -Uri 'https://api.github.com/repos/cli/cli/releases/latest' -Method Get
  $asset = $release.assets | Where-Object { $_.name -like "*_${suffix}.zip" } | Select-Object -First 1
  if (-not $asset) { throw "Could not find asset matching *_${suffix}.zip" }
  return @{ url = $asset.browser_download_url; version = $release.tag_name }
}

function Get-TaggedAssetUrl($suffix, $version) {
  $release = Invoke-RestMethod -Headers @{ 'User-Agent'='PolliLib-Agent' } -Uri ("https://api.github.com/repos/cli/cli/releases/tags/{0}" -f $version) -Method Get
  $asset = $release.assets | Where-Object { $_.name -like "*_${suffix}.zip" } | Select-Object -First 1
  if (-not $asset) { throw "Could not find asset for $version matching *_${suffix}.zip" }
  return @{ url = $asset.browser_download_url; version = $release.tag_name }
}

$suffix = Get-ArchSuffix
$destRoot = Resolve-Path -LiteralPath $Destination -ErrorAction SilentlyContinue
if (-not $destRoot) {
  Ensure-Dir $Destination
  $destRoot = Resolve-Path -LiteralPath $Destination
}

if ($Version -eq 'latest') {
  $asset = Get-LatestAssetUrl -suffix $suffix
} else {
  if ($Version -notmatch '^v') { $Version = 'v' + $Version }
  $asset = Get-TaggedAssetUrl -suffix $suffix -version $Version
}

$zipPath = Join-Path $destRoot.Path 'gh.zip'
Write-Host ("Downloading gh CLI {0} for {1}" -f $asset.version, $suffix)
Invoke-WebRequest -Uri $asset.url -OutFile $zipPath

Write-Host "Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $destRoot.Path -Force
Remove-Item $zipPath -Force

# Find extracted bin/gh.exe
$binCandidate = Get-ChildItem -Path $destRoot.Path -Recurse -Filter gh.exe | Where-Object { $_.FullName -like "*\bin\gh.exe" } | Select-Object -First 1
if (-not $binCandidate) { throw "gh.exe not found after extraction" }

$targetBin = Join-Path $destRoot.Path 'bin'
Ensure-Dir $targetBin

# Copy bin folder contents to flattened .tools/gh/bin
$sourceBin = Split-Path -Parent $binCandidate.FullName
Copy-Item -Path (Join-Path $sourceBin '*') -Destination $targetBin -Recurse -Force

$binPath = (Resolve-Path $targetBin).Path

# Update current session PATH
if (-not ($env:PATH -split ';' | Where-Object { $_ -ieq $binPath })) {
  $env:PATH = "$binPath;" + $env:PATH
}

if ($UpdatePath) {
  # Persist to User PATH (non-admin)
  $currentUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (-not ($currentUserPath -split ';' | Where-Object { $_ -ieq $binPath })) {
    [Environment]::SetEnvironmentVariable('Path', ($currentUserPath + ';' + $binPath).Trim(';'), 'User')
    Write-Host "Added to User PATH: $binPath"
  }
}

& "$binPath/gh.exe" --version
if ($LASTEXITCODE -ne 0) { throw "gh did not run successfully" }

Write-Host "GitHub CLI installed at: $binPath"

