param(
    [int]$StartIndex = -1,
    [int]$Count = 50,
    [string]$TargetDevice = "v1",
    [string]$Provider = "xai",
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if ($StartIndex -lt 0) {
    $resolved = python .\tools\resolve_next_toxic_start.py --target-device $TargetDevice --provider $Provider
    $resolved = "$resolved".Trim()
    if (-not $resolved) {
        throw "Could not resolve next pending toxic start index."
    }
    $StartIndex = [int]$resolved
    if ($StartIndex -lt 0) {
        throw "All toxic CSV cases are already completed."
    }
    Write-Host "Auto-selected next pending StartIndex: $StartIndex"
}

$env:TOXIC_START_INDEX = "$StartIndex"
$env:TOXIC_COUNT = "$Count"
$env:TARGET_DEVICE = "$TargetDevice"
$env:CHEEKO_PROVIDER = "$Provider"
if ($Model -ne "") { $env:XAI_MODEL = "$Model" }

# CSV mode is handled by injecting into eval.py so keep dynamic CSV append off.
$env:USE_SELECTED_CSV_CASES = "false"

# Run-scoped cap: evaluate only this many NEW tests in this run.
# Do not cap MAX_PER_GROUP here, otherwise cumulative rendering can look truncated.
if (Test-Path Env:RUN_MAX_PER_GROUP) { Remove-Item Env:RUN_MAX_PER_GROUP -ErrorAction SilentlyContinue }
$env:RUN_MAX_NEW_EVALS_PER_RUN = "$Count"

python .\local_csv_preview.py

# Pre-check: how many selected rows are NEW vs already executed in Group 6.
@'
import json
import os
import re
import pandas as pd
from pathlib import Path

csv_sel = "selected_eval_questions.csv"
device = (os.getenv("TARGET_DEVICE", "v1") or "v1").strip().lower()
provider = (os.getenv("CHEEKO_PROVIDER", "google") or "google").strip().lower()
report_path = Path("reports") / f"cheeko_{device}_{provider}_eval_report.html"

if not os.path.exists(csv_sel):
    print("Selection file not found, skipping duplicate check.")
    raise SystemExit(0)

df = pd.read_csv(csv_sel)
selected_ids = set(df["conv_id"].astype(str).str.slice(0, 12).str.lower().tolist())

done_ids = set()
if report_path.exists():
    try:
        html = report_path.read_text(encoding="utf-8", errors="ignore")
        for m in re.findall(r"\[([0-9a-fA-F]{8,})\]", html):
            done_ids.add(m.lower())
    except Exception:
        pass

already = selected_ids.intersection(done_ids)
new = selected_ids.difference(done_ids)

print("")
print("=== Batch Duplicate Check ===")
print(f"Selected: {len(selected_ids)}")
print(f"New: {len(new)}")
print(f"Already done: {len(already)}")
if len(new) == 0 and len(selected_ids) > 0:
    print("WARNING: All selected prompts were already executed. Progress count will not increase.")
print("=============================")
print("")
'@ | python -

python .\tools\inject_csv_to_eval.py --max $Count --replace-auto

# Re-evaluate now-injected set and update cumulative report.
Remove-Item ".\state\cheeko_${TargetDevice}_${Provider}_eval_checkpoint.json" -ErrorAction SilentlyContinue

python .\eval.py

Write-Host ""
Write-Host "Latest report:"
$latestPath = ".\reports\cheeko_${TargetDevice}_${Provider}_eval_report_latest.html"
if (Test-Path $latestPath) {
  Write-Host "file:///$(Resolve-Path $latestPath)"
} else {
  Write-Host "(latest report not generated in this run)"
}
Write-Host ""
Write-Host "Cumulative report (includes previous + this batch):"
$cumPath = ".\reports\cheeko_${TargetDevice}_${Provider}_eval_report.html"
if (Test-Path $cumPath) {
  Write-Host "file:///$(Resolve-Path $cumPath)"
} else {
  Write-Host "(cumulative report not found)"
}
Write-Host ""
Write-Host "Run history archive (dated links):"
$arcPath = ".\reports\cheeko_run_links_archive.md"
if (Test-Path $arcPath) {
  Write-Host "file:///$(Resolve-Path $arcPath)"
} else {
  Write-Host "(run archive not found)"
}

# Private terminal-only CSV progress (not added to report HTML).
python .\tools\toxic_csv_progress.py --target-device $TargetDevice --provider $Provider
