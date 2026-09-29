param([switch]$SkipQgc)

. (Join-Path $PSScriptRoot "common.ps1")

$results = [System.Collections.Generic.List[object]]::new()
function Add-Result([string]$Name, [bool]$Ok, [string]$Detail) {
    $results.Add([pscustomobject]@{ Component = $Name; OK = $Ok; Detail = $Detail })
}

try {
    Invoke-RestMethod -TimeoutSec 3 -Uri "http://127.0.0.1:9293/health" | Out-Null
    Add-Result "Operator API" $true "port 9293"
} catch { Add-Result "Operator API" $false $_.Exception.Message }

Add-Result "Dashboard" (Wait-JolgwaHttp -Uri "http://127.0.0.1:5173/" -TimeoutSeconds 2) "port 5173"
if (-not $SkipQgc) {
    Add-Result "QGroundControl" ($null -ne (Get-Process QGroundControl -ErrorAction SilentlyContinue)) "Windows process"
}

try {
    $control = Invoke-RestMethod -TimeoutSec 3 -Uri "http://127.0.0.1:9293/health/control"
    $gatewayDetail = if ($control.ros_connected) { "connected" } else { "waiting for Jetson" }
    Add-Result "Jetson gateway" $true $gatewayDetail
} catch { Add-Result "Jetson gateway" $false $_.Exception.Message }

$results | Format-Table -AutoSize
if ($results.OK -contains $false) { exit 1 } else { exit 0 }
