[CmdletBinding()]
param(
    [string]$Branch = "phase-stable-icassp",
    [string]$CommitMessage = "Add phase-stable TI-DWT ICASSP experiment pipeline",
    [switch]$Commit
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Native {
    param([Parameter(Mandatory)][string]$Command, [Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

function Test-NativeCommand {
    param(
        [Parameter(Mandatory)][string]$Command,
        [Parameter(ValueFromRemainingArguments)][string[]]$Arguments
    )
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Command @Arguments *> $null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

foreach ($tool in @("git", "gh")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required command is missing: $tool"
    }
}

$inside = (& git rev-parse --is-inside-work-tree 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or $inside -ne "true") {
    throw "Not inside a Git worktree: $RepoRoot"
}

$isAuthenticated = Test-NativeCommand gh auth status --hostname github.com
if (-not $isAuthenticated) {
    Write-Host "GitHub browser authentication is required. No token is printed or stored by this script."
    Invoke-Native gh auth login --hostname github.com --git-protocol https --web
}

$currentBranch = (& git branch --show-current).Trim()
if ($currentBranch -ne $Branch) {
    $branchMatch = (& git branch --list $Branch) -join ""
    $branchExists = -not [string]::IsNullOrWhiteSpace($branchMatch)
    if ($branchExists) {
        Invoke-Native git switch $Branch
    } else {
        Invoke-Native git switch -c $Branch
    }
}

$status = @(& git status --porcelain)
if ($status.Count -gt 0) {
    if (-not $Commit) {
        Write-Host "Uncommitted files:" -ForegroundColor Yellow
        $status | ForEach-Object { Write-Host "  $_" }
        throw "Re-run with -Commit after reviewing the files above."
    }
    $publishPaths = @(
        ".gitignore",
        "README.md",
        "A100_QUICKSTART.md",
        "PHASE_STABLE_EXPERIMENTS.md",
        "configs/phase_stable_icassp.yaml",
        "phase_stable",
        "requirements-phase-stable.txt",
        "scripts",
        "tests"
    )
    Invoke-Native git add -- @publishPaths
    $staged = @(& git diff --cached --name-only)
    if ($staged.Count -eq 0) {
        throw "No intended experiment files were staged."
    }
    Write-Host "Files to commit:" -ForegroundColor Cyan
    $staged | ForEach-Object { Write-Host "  $_" }
    Invoke-Native git commit -m $CommitMessage
}

$remainingStatus = @(& git status --porcelain)
if ($remainingStatus.Count -gt 0) {
    throw "Worktree still has unrelated changes; refusing to publish a mixed state."
}

$login = (& gh api user --jq .login).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($login)) {
    throw "Could not determine the authenticated GitHub username."
}
$forkUrl = "https://github.com/$login/WFS-SB.git"
$remoteNames = @(& git remote)
if ($remoteNames -notcontains "fork") {
    $repoExists = $true
    if (-not (Test-NativeCommand gh repo view "$login/WFS-SB" --json nameWithOwner)) {
        $repoExists = $false
    }
    if (-not $repoExists) {
        Invoke-Native gh repo fork MAC-AutoML/WFS-SB --clone=false
    }
    Invoke-Native git remote add fork $forkUrl
} else {
    $forkRemote = (& git remote get-url fork).Trim()
    if ($forkRemote -ne $forkUrl) {
    throw "Remote 'fork' points to '$($forkRemote.Trim())', expected '$forkUrl'."
    }
}

Invoke-Native git push --set-upstream fork $Branch

Write-Host ""
Write-Host "Published successfully." -ForegroundColor Green
Write-Host "A100 clone URL : $forkUrl"
Write-Host "A100 branch    : $Branch"
Write-Host ""
Write-Host "Server command:"
Write-Host "  git clone --branch $Branch $forkUrl"
