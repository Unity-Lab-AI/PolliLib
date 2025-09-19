Param(
  [Parameter(Mandatory=$true)][string]$Tag,
  [string]$Repo = 'Unity-Lab-AI/PolliLib',
  [string]$Target = 'main',
  [string]$Title,
  [string]$NotesFile,
  [switch]$Draft,
  [switch]$Prerelease,
  [string[]]$Assets
)

$ErrorActionPreference = 'Stop'

function Ensure-GH {
  $gh = Get-Command gh -ErrorAction SilentlyContinue
  if ($gh) { return $true }
  # Try local portable install location
  $localBin = Join-Path (Resolve-Path '.').Path '.tools/gh/bin/gh.exe'
  if (Test-Path $localBin) {
    $env:PATH = ((Split-Path $localBin -Parent) + ';' + $env:PATH)
    return $true
  }
  # Attempt to install portable gh if missing
  $installer = 'scripts/install_gh_cli.ps1'
  if (Test-Path $installer) {
    Write-Host 'gh not found; installing portable GitHub CLI...'
    & $installer
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install GitHub CLI' }
    return $true
  }
  return $false
}

if (-not $Title) { $Title = "PolliLib $Tag" }

if (-not (Ensure-GH)) { throw 'gh is not available. Please install GitHub CLI and re-run.' }

# Check auth
& gh auth status --hostname github.com | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host 'You are not authenticated with gh. Please run:'
  Write-Host '  gh auth login --hostname github.com --scopes repo'
  exit 1
}

# Prepare notes file
if (-not $NotesFile) {
  $defaultNotes = @"
# $Title
Stable cross-language release: Python + JavaScript clients, shared AST.
Highlights: images (flux, 512x512, nologo, 5-8 digit seed), text, chat (SSE + tools), vision, stt, public feeds, referrer/token, AST in /AST.
Tests: pytest python/tests; node --test javascript/tests
"@
  $tmp = New-TemporaryFile
  Set-Content -NoNewline -Path $tmp -Value $defaultNotes
  $NotesFile = $tmp
}

# Determine if release exists
& gh release view $Tag -R $Repo 1>$null 2>$null
$exists = ($LASTEXITCODE -eq 0)

if (-not $exists) {
  Write-Host "Creating release $Tag on $Repo"
  $args = @('release','create', $Tag, '-R', $Repo, '--target', $Target, '-t', $Title, '-F', $NotesFile)
  if ($Draft) { $args += '-d' }
  if ($Prerelease) { $args += '-p' }
  if ($Assets) { $args += $Assets }
  & gh @args
  if ($LASTEXITCODE -ne 0) { throw 'Failed to create release' }
}
else {
  Write-Host "Updating release $Tag on $Repo"
  $args = @('release','edit', $Tag, '-R', $Repo, '-t', $Title, '-F', $NotesFile)
  if ($Prerelease) { $args += '-p' } # only sets true if requested
  if ($Draft) { $args += '-d' }      # only sets true if requested
  & gh @args
  if ($LASTEXITCODE -ne 0) { throw 'Failed to update release' }
}

Write-Host ("release_url: https://github.com/{0}/releases/tag/{1}" -f $Repo, $Tag)

