param(
    [switch]$SkipQgc,
    [switch]$SkipGeminiKey,
    [switch]$SkipFirewall
)

. (Join-Path $PSScriptRoot "common.ps1")
$projectRoot = Get-JolgwaProjectRoot

Write-Host "[1/5] Checking Windows prerequisites"
$pythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
$pythonExecutable = $null
if ($null -ne $pythonLauncher) {
    & py.exe -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $pythonExecutable = "py.exe" }
}
if ($null -eq $pythonExecutable) {
    $pythonInstaller = Join-Path $projectRoot "installers\python-3.11.9-amd64.exe"
    if (Test-Path -LiteralPath $pythonInstaller) {
        Write-Host "Installing the bundled Python 3.11 runtime..."
        Start-Process -FilePath $pythonInstaller -Wait -ArgumentList @(
            "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0", "Include_launcher=1"
        )
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
        if (-not (Test-Path -LiteralPath $candidate)) {
            throw "Python installation was not detected. Run setup again after opening a new terminal."
        }
        $pythonExecutable = $candidate
    } elseif (Test-ExternalCommand "winget.exe") {
        & winget.exe install --exact --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        throw "Python was installed. Open a new terminal and run setup again."
    } else {
        throw "Install Python 3.11 x64, then run setup again."
    }
}

Write-Host "[2/5] Creating the Windows Python environment"
$venvPython = Get-JolgwaVenvPython
if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($pythonExecutable -eq "py.exe") {
        & py.exe -3.11 -m venv (Join-Path $projectRoot ".venv")
    } else {
        & $pythonExecutable -m venv (Join-Path $projectRoot ".venv")
    }
}
$requirements = Join-Path $projectRoot "requirements-field.txt"
$wheelhouse = Join-Path $projectRoot "wheels"
if (Test-Path -LiteralPath $wheelhouse) {
    & $venvPython -m pip install --no-index --find-links $wheelhouse -r $requirements
} else {
    & $venvPython -m pip install -r $requirements
}
if ($LASTEXITCODE -ne 0) { throw "Dashboard Python dependency installation failed." }

Write-Host "[3/5] Checking QGroundControl"
if (-not $SkipQgc -and -not (Get-QGroundControlPath)) {
    $installer = Join-Path $projectRoot "installers\QGroundControl-installer.exe"
    if (-not (Test-Path -LiteralPath $installer)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $installer) | Out-Null
        $url = "https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-installer.exe"
        Write-Host "Downloading QGroundControl from the official distribution..."
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer
    }
    Write-Host "Installing the bundled QGroundControl package silently..."
    Start-Process -FilePath $installer -Wait -ArgumentList "/S"
    if (-not (Get-QGroundControlPath)) {
        throw "QGroundControl installation was not detected."
    }
}

Write-Host "[4/5] Checking the Gemini key"
$key = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
if (-not $SkipGeminiKey -and [string]::IsNullOrWhiteSpace($key)) {
    $secure = Read-Host "Gemini API key (stored in the current Windows user environment)" -AsSecureString
    $handle = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($handle)
        if ([string]::IsNullOrWhiteSpace($plain)) { throw "Gemini API key is empty." }
        [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $plain, "User")
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($handle)
        Remove-Variable plain -ErrorAction SilentlyContinue
    }
}

Write-Host "[5/5] Configuring local-network access for the Jetson gateway"
if (-not $SkipFirewall) {
    $firewallScript = Join-Path $PSScriptRoot "configure_firewall.ps1"
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
    if ($isAdmin) {
        & $firewallScript
    } else {
        Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList (
            "-NoProfile -ExecutionPolicy Bypass -File `"$firewallScript`""
        )
    }
}

& (Join-Path $PSScriptRoot "preflight.ps1") -AllowDisconnectedLan
if ($LASTEXITCODE -ne 0) { throw "Setup finished, but preflight checks still fail." }
Write-Host "Setup complete. Run START_QGC_DASHBOARD.cmd."
