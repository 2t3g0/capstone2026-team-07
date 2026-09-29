Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:ProjectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\..")
)
$script:RuntimeRoot = Join-Path $script:ProjectRoot ".runtime\qgc-dashboard"
$script:LogRoot = Join-Path $script:RuntimeRoot "logs"

function Get-JolgwaProjectRoot {
    return $script:ProjectRoot
}

function Get-JolgwaRuntimeRoot {
    return $script:RuntimeRoot
}

function Initialize-JolgwaRuntime {
    New-Item -ItemType Directory -Force -Path $script:RuntimeRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $script:LogRoot | Out-Null
}

function Test-ExternalCommand {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-QGroundControlPath {
    $candidates = @(
        (Join-Path $env:ProgramFiles "QGroundControl\bin\QGroundControl.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "QGroundControl\bin\QGroundControl.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\QGroundControl\bin\QGroundControl.exe")
    )
    return $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
        Select-Object -First 1
}

function Save-ProcessId {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$Id
    )
    Initialize-JolgwaRuntime
    [System.IO.File]::WriteAllText(
        (Join-Path $script:RuntimeRoot "$Name.pid"),
        [string]$Id,
        [System.Text.Encoding]::ASCII
    )
}

function Start-LoggedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string]$ArgumentList = "",
        [string]$WorkingDirectory = $script:ProjectRoot,
        [switch]$Visible
    )
    Initialize-JolgwaRuntime
    $parameters = @{
        FilePath = $FilePath
        WorkingDirectory = $WorkingDirectory
        PassThru = $true
    }
    if (-not [string]::IsNullOrWhiteSpace($ArgumentList)) {
        $parameters.ArgumentList = $ArgumentList
    }
    if (-not $Visible) {
        $parameters.WindowStyle = "Hidden"
        $parameters.RedirectStandardOutput = Join-Path $script:LogRoot "$Name.out.log"
        $parameters.RedirectStandardError = Join-Path $script:LogRoot "$Name.err.log"
    }
    $process = Start-Process @parameters
    Save-ProcessId -Name $Name -Id $process.Id
    return $process
}

function Wait-JolgwaHttp {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int]$TimeoutSeconds = 30
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri $Uri | Out-Null
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Get-JolgwaVenvPython {
    return Join-Path $script:ProjectRoot ".venv\Scripts\python.exe"
}

function Get-LanIPv4Addresses {
    $physicalIndexes = @(
        Get-NetAdapter -Physical -ErrorAction SilentlyContinue |
            Where-Object Status -eq "Up" |
            Select-Object -ExpandProperty ifIndex
    )
    return @(
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object {
                $physicalIndexes -contains $_.InterfaceIndex -and
                $_.IPAddress -notlike "127.*" -and
                $_.IPAddress -notlike "169.254.*"
            } |
            Select-Object -ExpandProperty IPAddress -Unique
    )
}
