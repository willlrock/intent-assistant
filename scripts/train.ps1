[CmdletBinding()]
param(
    [switch]$Force
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

$RequiredFiles = @("config.yml", "domain.yml", "endpoints.yml")
foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $RequiredFile) -PathType Leaf)) {
        throw "Required project file is missing: $RequiredFile"
    }
}

$RasaArguments = @(
    "train",
    "--config", "config.yml",
    "--domain", "domain.yml",
    "--data", "data",
    "--out", "models",
    "--endpoints", "endpoints.yml"
)

if ($Force) {
    $RasaArguments += "--force"
}

Write-Host "Running: rasa $($RasaArguments -join ' ')"
& $Python -m rasa @RasaArguments
if ($LASTEXITCODE -ne 0) {
    throw "Rasa training failed with exit code $LASTEXITCODE."
}

$LatestModel = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "models") -Filter "*.tar.gz" -File |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

if ($null -eq $LatestModel) {
    throw "Training completed without producing a model in models/."
}

Write-Host "Model created: $($LatestModel.FullName)"
