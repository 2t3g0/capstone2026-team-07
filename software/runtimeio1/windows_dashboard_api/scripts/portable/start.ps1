param(
    [switch]$NoQgc,
    [switch]$NoBrowser
)

. (Join-Path $PSScriptRoot "common.ps1")
$projectRoot = Get-JolgwaProjectRoot

& (Join-Path $PSScriptRoot "preflight.ps1") -Quiet
if ($LASTEXITCODE -ne 0) {
    throw "Preflight failed. Run SETUP_QGC_DASHBOARD.cmd first."
}

& (Join-Path $PSScriptRoot "stop.ps1") -Quiet
Initialize-JolgwaRuntime

$python = Get-JolgwaVenvPython
$apiScript = Join-Path $projectRoot "scripts\start_operator_api.py"
$dashboardScript = Join-Path $PSScriptRoot "serve_dashboard.py"
$dashboardDist = Join-Path $projectRoot "frontend\dist"

Write-Host "Starting Gemini API and dashboard..."
Start-LoggedProcess -Name "operator-api" -FilePath $python `
    -ArgumentList "`"$apiScript`"" | Out-Null
if (-not (Wait-JolgwaHttp -Uri "http://127.0.0.1:9293/health" -TimeoutSeconds 30)) {
    throw "Operator API did not become healthy. Check .runtime\qgc-dashboard\logs."
}

Start-LoggedProcess -Name "dashboard" -FilePath $python `
    -ArgumentList "`"$dashboardScript`" --directory `"$dashboardDist`" --port 5173" | Out-Null
if (-not (Wait-JolgwaHttp -Uri "http://127.0.0.1:5173/" -TimeoutSeconds 15)) {
    throw "Dashboard did not start. Check .runtime\qgc-dashboard\logs."
}

if (-not $NoQgc) {
    $qgc = Get-QGroundControlPath
    Start-LoggedProcess -Name "qgroundcontrol" -FilePath $qgc -Visible | Out-Null
}

Start-Sleep -Seconds 5
& (Join-Path $PSScriptRoot "verify.ps1") -SkipQgc:$NoQgc
if ($LASTEXITCODE -ne 0) {
    & (Join-Path $PSScriptRoot "stop.ps1") -Quiet
    throw "Startup verification failed. Check .runtime\qgc-dashboard\logs."
}

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:5173/"
}

Write-Host "QGC + field dashboard are running."
Write-Host "QGC can connect directly to the real PX4 vehicle."
Write-Host "Automatic missions become available when the Jetson ROS gateway connects to port 9293."
Write-Host "Stop it with STOP_QGC_DASHBOARD.cmd."
