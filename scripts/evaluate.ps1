[CmdletBinding()]
param(
    [string]$Model,
    [string]$TestData = "tests/nlu_test.yml",
    [string]$OutputDirectory = "results/final"
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
Create it with Python 3.10 and install the pinned development dependencies:
  py -3.10 -m venv .venv-rasa
  .\.venv-rasa\Scripts\python.exe -m pip install -r requirements-dev.txt
"@
}

Set-Location -LiteralPath $ProjectRoot

$RequiredFiles = @(
    "config.yml",
    "domain.yml",
    "data/nlu.yml",
    "scripts/summarize_results.py",
    "scripts/evaluate_runtime_nlu.py",
    "tests/test_stories.yml"
)
foreach ($RequiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $RequiredFile) -PathType Leaf)) {
        throw "Required project file is missing: $RequiredFile"
    }
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
    $ModelCandidate = if ([System.IO.Path]::IsPathRooted($Model)) {
        $Model
    }
    else {
        Join-Path $ProjectRoot $Model
    }

    if (-not (Test-Path -LiteralPath $ModelCandidate -PathType Leaf)) {
        throw "Model file was not found: $Model"
    }

    $ModelPath = (Resolve-Path -LiteralPath $ModelCandidate).Path
}

if ([string]::IsNullOrWhiteSpace($TestData)) {
    throw "TestData must point to a Rasa NLU YAML file."
}

$TestDataCandidate = if ([System.IO.Path]::IsPathRooted($TestData)) {
    $TestData
}
else {
    Join-Path $ProjectRoot $TestData
}

if (-not (Test-Path -LiteralPath $TestDataCandidate -PathType Leaf)) {
    throw "NLU test data file was not found: $TestData"
}
$TestDataPath = (Resolve-Path -LiteralPath $TestDataCandidate).Path

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    throw "OutputDirectory must not be empty."
}

$OutputCandidate = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
}
else {
    Join-Path $ProjectRoot $OutputDirectory
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputCandidate)

if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
    throw "OutputDirectory points to a file: $OutputPath"
}
New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
$OutputPath = (Resolve-Path -LiteralPath $OutputPath).Path

$DomainPath = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "domain.yml")).Path
$ConfigPath = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "config.yml")).Path
$TrainDataPath = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "data/nlu.yml")).Path
$SummarizerPath = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "scripts/summarize_results.py")).Path
$RuntimeEvaluatorPath = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "scripts/evaluate_runtime_nlu.py")).Path
$CoreStoriesPath = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot "tests/test_stories.yml")).Path
$SummaryPath = Join-Path $OutputPath "summary.md"
$RuntimeReportPath = Join-Path $OutputPath "runtime_report.json"
$CoreOutputPath = Join-Path $OutputPath "core"
$CoreReportPath = Join-Path $CoreOutputPath "story_report.json"
$PublishedCoreReportPath = Join-Path $OutputPath "story_report.json"

$RasaArguments = @(
    "test",
    "nlu",
    "--model", $ModelPath,
    "--nlu", $TestDataPath,
    "--domain", $DomainPath,
    "--out", $OutputPath
)

Write-Host "Running held-out NLU evaluation against $TestDataPath"
Write-Host "Running: rasa $($RasaArguments -join ' ')"
& $Python -m rasa @RasaArguments
if ($LASTEXITCODE -ne 0) {
    throw "Rasa NLU evaluation failed with exit code $LASTEXITCODE."
}

Write-Host "Measuring the complete NLU pipeline including FallbackClassifier"
& $Python $RuntimeEvaluatorPath `
    $ModelPath `
    --nlu $TestDataPath `
    --config $ConfigPath `
    --output $RuntimeReportPath
if ($LASTEXITCODE -ne 0) {
    throw "Runtime NLU evaluation failed with exit code $LASTEXITCODE."
}

Write-Host "Running Rasa Core evaluation against $CoreStoriesPath"
$CoreArguments = @(
    "test",
    "core",
    "--model", $ModelPath,
    "--stories", $CoreStoriesPath,
    "--out", $CoreOutputPath
)
Write-Host "Running: rasa $($CoreArguments -join ' ')"
& $Python -m rasa @CoreArguments
if ($LASTEXITCODE -ne 0) {
    throw "Rasa Core evaluation failed with exit code $LASTEXITCODE."
}
if (-not (Test-Path -LiteralPath $CoreReportPath -PathType Leaf)) {
    throw "Rasa Core evaluation did not produce story_report.json."
}
Copy-Item -LiteralPath $CoreReportPath -Destination $PublishedCoreReportPath -Force

Write-Host "Building Markdown summary from the generated reports"
& $Python $SummarizerPath `
    $OutputPath `
    --train-data $TrainDataPath `
    --test-data $TestDataPath `
    --output $SummaryPath
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation summary generation failed with exit code $LASTEXITCODE."
}

Write-Host "Evaluation artifacts: $OutputPath"
