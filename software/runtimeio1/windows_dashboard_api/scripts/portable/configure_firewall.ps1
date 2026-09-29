$ErrorActionPreference = "Stop"
$displayName = "Jolgwa Operator API (Private LAN)"
$existing = Get-NetFirewallRule -DisplayName $displayName -ErrorAction SilentlyContinue
if (-not $existing) {
    New-NetFirewallRule `
        -DisplayName $displayName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 9293 `
        -Profile Private `
        -RemoteAddress LocalSubnet | Out-Null
}
Write-Host "Private-LAN firewall access is ready on TCP port 9293."
