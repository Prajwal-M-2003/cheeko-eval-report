param(
    [int]$StartIndex = -1,
    [int]$Count = 20,
    [string]$TargetDevice = "v1"
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if ($StartIndex -lt 0) {
    $resolved = python .\tools\resolve_next_common_toxic_start.py --target-device $TargetDevice
    $resolved = "$resolved".Trim()
    if (-not $resolved) {
        throw "Could not resolve next shared toxic start index."
    }
    $StartIndex = [int]$resolved
    if ($StartIndex -lt 0) {
        throw "All toxic CSV cases are already aligned across Google and xAI."
    }
    Write-Host "Auto-selected next shared StartIndex: $StartIndex"
}

$env:TOXIC_START_INDEX = "$StartIndex"
$env:TOXIC_COUNT = "$Count"
$env:TARGET_DEVICE = "$TargetDevice"
$env:USE_SELECTED_CSV_CASES = "false"
$env:CASE_ID_BASE = "$($StartIndex + 1)"
if (Test-Path Env:RUN_MAX_PER_GROUP) { Remove-Item Env:RUN_MAX_PER_GROUP -ErrorAction SilentlyContinue }
$env:RUN_MAX_NEW_EVALS_PER_RUN = "$Count"

python .\local_csv_preview.py
python .\tools\inject_csv_to_eval.py --max $Count --replace-auto

foreach ($Provider in @("google", "xai")) {
    Write-Host ""
    Write-Host "=== Running provider: $Provider ==="
    $env:CHEEKO_PROVIDER = "$Provider"
    Remove-Item ".\state\cheeko_${TargetDevice}_${Provider}_eval_checkpoint.json" -ErrorAction SilentlyContinue
    python .\eval.py
    Write-Host "=== Completed provider: $Provider ==="
}

python .\tools\build_provider_comparison_report.py --device $TargetDevice

Write-Host ""
Write-Host "Gemini report:"
Write-Host "file:///$(Resolve-Path .\reports\cheeko_${TargetDevice}_google_eval_report.html)"
Write-Host ""
Write-Host "xAI report:"
Write-Host "file:///$(Resolve-Path .\reports\cheeko_${TargetDevice}_xai_eval_report.html)"
Write-Host ""
Write-Host "Comparison report:"
Write-Host "file:///$(Resolve-Path .\reports\cheeko_${TargetDevice}_provider_comparison_report.html)"
