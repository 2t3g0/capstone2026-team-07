param([switch]$Quiet)

. (Join-Path $PSScriptRoot "common.ps1")
$runtimeRoot = Get-JolgwaRuntimeRoot

if (Test-Path -LiteralPath $runtimeRoot) {
    Get-ChildItem -LiteralPath $runtimeRoot -Filter "*.pid" -File | ForEach-Object {
        $raw = [System.IO.File]::ReadAllText($_.FullName).Trim()
        $pidValue = 0
        if ([int]::TryParse($raw, [ref]$pidValue)) {
            $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($process) { Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue }
        }
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

if (-not $Quiet) { Write-Host "QGC + field dashboard stopped." }
