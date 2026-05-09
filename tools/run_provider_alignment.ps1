param(
    [string]$TargetDevice = "v1",
    [int]$ChunkSize = 20
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$raw = python .\tools\provider_alignment_ranges.py --device $TargetDevice
$raw = "$raw".Trim()
if (-not $raw) {
    throw "Could not load provider alignment ranges."
}

$data = $raw | ConvertFrom-Json

function Invoke-RangeRuns {
    param(
        [string]$Provider,
        [object[]]$Ranges
    )

    foreach ($range in $Ranges) {
        $start = [int]$range[0]
        $end = [int]$range[1]
        $length = $end - $start + 1
        $offset = 0

        while ($offset -lt $length) {
            $currentStart = $start + $offset
            $currentCount = [Math]::Min($ChunkSize, $length - $offset)
            Write-Host ""
            Write-Host "=== Aligning $Provider | StartIndex=$currentStart Count=$currentCount ==="
            powershell -ExecutionPolicy Bypass -File .\tools\run_toxic_batch_from_csv.ps1 -StartIndex $currentStart -Count $currentCount -TargetDevice $TargetDevice -Provider $Provider
            $offset += $currentCount
        }
    }
}

Write-Host "Google total: $($data.google_total)"
Write-Host "xAI total: $($data.xai_total)"
Write-Host "Common total: $($data.common_total)"
Write-Host ""
Write-Host "xAI needs ranges: $($data.xai_needs_ranges | ConvertTo-Json -Compress)"
Write-Host "Google needs ranges: $($data.google_needs_ranges | ConvertTo-Json -Compress)"

Invoke-RangeRuns -Provider "xai" -Ranges $data.xai_needs_ranges
Invoke-RangeRuns -Provider "google" -Ranges $data.google_needs_ranges

python .\tools\build_provider_comparison_report.py --device $TargetDevice

Write-Host ""
Write-Host "Alignment complete."
Write-Host "Gemini report: file:///$(Resolve-Path .\reports\cheeko_${TargetDevice}_google_eval_report.html)"
Write-Host "xAI report: file:///$(Resolve-Path .\reports\cheeko_${TargetDevice}_xai_eval_report.html)"
Write-Host "Comparison report: file:///$(Resolve-Path .\reports\cheeko_${TargetDevice}_provider_comparison_report.html)"
