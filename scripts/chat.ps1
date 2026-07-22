[CmdletBinding()]
param(
    [string]$Model
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:RASA_TELEMETRY_ENABLED = "false"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv-rasa\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw @"
Rasa environment was not found at .venv-rasa.
Create it with Python 3.10 and install the pinned dependencies:
  py -3.10 -m venv .venv-rasa
  .\.venv-rasa\Scripts\python.exe -m pip install -r requirements.txt
"@
}

Set-Location -LiteralPath $ProjectRoot

$EndpointsPath = Join-Path $ProjectRoot "endpoints.yml"
if (-not (Test-Path -LiteralPath $EndpointsPath -PathType Leaf)) {
    throw "Required project file is missing: endpoints.yml"
}

if ([string]::IsNullOrWhiteSpace($Model)) {
    $ModelsDirectory = Join-Path $ProjectRoot "models"
    if (-not (Test-Path -LiteralPath $ModelsDirectory -PathType Container)) {
        throw "No models directory was found. Run scripts/train.ps1 first."
    }

    $ModelFile = Get-ChildItem -LiteralPath $ModelsDirectory -Filter "*.tar.gz" -File |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1

    if ($null -eq $ModelFile) {
        throw "No trained model was found in models/. Run scripts/train.ps1 first."
    }

    $ModelPath = $ModelFile.FullName
}
else {
    $ModelPath = (Resolve-Path -LiteralPath $Model).Path
}

$ActionProcess = $null

try {
    Write-Host "Starting local Rasa action server on http://127.0.0.1:5055 ..."
    $ActionProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "rasa_sdk", "--actions", "actions.actions", "--port", "5055", "--quiet") `
        -WorkingDirectory $ProjectRoot `
        -PassThru `
        -WindowStyle Hidden

    $ActionServerReady = $false
    for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
        if ($ActionProcess.HasExited) {
            throw "The Rasa action server exited before becoming ready."
        }

        try {
            $HealthResponse = Invoke-WebRequest `
                -Uri "http://127.0.0.1:5055/health" `
                -UseBasicParsing `
                -TimeoutSec 1
            if ($HealthResponse.StatusCode -eq 200) {
                $ActionServerReady = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $ActionServerReady) {
        throw "The Rasa action server did not become ready within 30 seconds."
    }

    $RasaArguments = @(
        "shell",
        "--model", $ModelPath,
        "--endpoints", $EndpointsPath
    )

    Write-Host "Running: rasa shell --model <latest model> --endpoints <local endpoints>"
    & $Python -m rasa @RasaArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Rasa shell failed with exit code $LASTEXITCODE."
    }
}
finally {
    if (($null -ne $ActionProcess) -and (-not $ActionProcess.HasExited)) {
        Stop-Process -Id $ActionProcess.Id -Force -ErrorAction SilentlyContinue
        $ActionProcess.WaitForExit()
    }
}
