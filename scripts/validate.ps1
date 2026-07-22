[CmdletBinding()]
param()

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
Create it with Python 3.10 and install the pinned development dependencies:
  py -3.10 -m venv .venv-rasa
  .\.venv-rasa\Scripts\python.exe -m pip install -r requirements-dev.txt
"@
}

Set-Location -LiteralPath $ProjectRoot

$RequiredFiles = @(
    "config.yml",
    "domain.yml",
    "endpoints.yml",
    "tests/static_validate.py"
)

foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $RequiredFile) -PathType Leaf)) {
        throw "Required project file is missing: $RequiredFile"
    }
}

Write-Host "Running: rasa data validate --domain domain.yml --data data --fail-on-warnings"
& $Python -m rasa data validate `
    --domain "domain.yml" `
    --data "data" `
    --fail-on-warnings
if ($LASTEXITCODE -ne 0) {
    throw "Rasa data validation failed with exit code $LASTEXITCODE."
}

Write-Host "Running: python tests/static_validate.py"
& $Python "tests/static_validate.py"
if ($LASTEXITCODE -ne 0) {
    throw "Static project validation failed with exit code $LASTEXITCODE."
}

Write-Host "Validation completed successfully."
