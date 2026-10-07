# Starts the local checkpoint-dashboard backend (dashboard/backend/server.py)
# using the project's own .venv, waits for it to report healthy, then opens
# the dashboard in the default browser. Binds to 127.0.0.1 only -- nothing
# here reaches outside this machine.
#
# Safe to run repeatedly: if a server is already listening on the port, this
# script detects it via the health check and just opens the browser.

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Server = Join-Path $ProjectRoot "dashboard\backend\server.py"
$Port = 8765
$HealthUrl = "http://127.0.0.1:$Port/api/health"
$DashboardUrl = "http://127.0.0.1:$Port/output/YieldMinx_dashboard_rev5.html"

if (-not (Test-Path $Python)) {
    Write-Error "Project virtual environment not found at $Python. Create it before running this launcher."
    exit 1
}

function Test-BackendHealthy {
    try {
        $resp = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (Test-BackendHealthy) {
    Write-Output "Backend already running at $HealthUrl -- reusing it."
} else {
    Write-Output "Starting backend: $Python $Server"
    Start-Process -FilePath $Python -ArgumentList "`"$Server`"" -WorkingDirectory $ProjectRoot `
        -WindowStyle Normal

    Write-Output "Waiting for backend to become healthy..."
    $deadline = (Get-Date).AddSeconds(30)
    $healthy = $false
    while ((Get-Date) -lt $deadline) {
        if (Test-BackendHealthy) { $healthy = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $healthy) {
        Write-Error "Backend did not become healthy within 30 seconds. Check the backend console window for errors."
        exit 1
    }
    Write-Output "Backend is healthy."
}

Write-Output "Opening dashboard: $DashboardUrl"
Start-Process $DashboardUrl
