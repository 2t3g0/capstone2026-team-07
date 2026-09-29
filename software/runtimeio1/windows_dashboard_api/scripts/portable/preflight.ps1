param(
    [switch]$Quiet,
    [switch]$AllowDisconnectedLan
)

. (Join-Path $PSScriptRoot "common.ps1")

$checks = [System.Collections.Generic.List[object]]::new()
function Add-Check([string]$Name, [bool]$Ok, [string]$Detail) {
    $checks.Add([pscustomobject]@{ Check = $Name; OK = $Ok; Detail = $Detail })
}

$python = Get-JolgwaVenvPython
Add-Check "Python venv" (Test-Path -LiteralPath $python) $python

$dashboard = Join-Path (Get-JolgwaProjectRoot) "frontend\dist\index.html"
Add-Check "Dashboard build" (Test-Path -LiteralPath $dashboard) $dashboard

$qgc = Get-QGroundControlPath
Add-Check "QGroundControl" ($null -ne $qgc) ([string]$qgc)

$key = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
if ([string]::IsNullOrWhiteSpace($key)) { $key = $env:GEMINI_API_KEY }
Add-Check "Gemini API key" (-not [string]::IsNullOrWhiteSpace($key)) "User environment"

$addresses = @(Get-LanIPv4Addresses)
$lanConnected = $addresses.Count -gt 0
$lanDetail = if ($lanConnected) {
    $addresses -join ", "
} elseif ($AllowDisconnectedLan) {
    "Not connected during installation; check again before flight"
} else {
    "No active physical LAN adapter"
}
Add-Check "LAN adapter" ($lanConnected -or $AllowDisconnectedLan) $lanDetail

if (-not $Quiet) { $checks | Format-Table -AutoSize }
if ($checks.OK -contains $false) { exit 1 } else { exit 0 }
